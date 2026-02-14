import torch
import math
import warnings
warnings.filterwarnings('ignore')

from timm.models.layers import trunc_normal_
from monai.networks.blocks.mlp import MLPBlock
from typing import Optional, Tuple, Union
from torch import nn

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa

# 开关控制
GeoPriorGen3DOn = 0     # 0: 不使用3D几何先验，1: 使用3D几何先验
HybridSparseAttnOn = 0  # 0: 不使用混合稀疏注意力，1: 使用混合稀疏注意力

############################################################################################################
# 3D 混合稀疏索引生成器
############################################################################################################
class HybridSparseIndexGen3D(nn.Module):
    def __init__(self, window_size=2, max_pow=3):
        super().__init__()
        self.window_size = window_size
        self.max_pow = max_pow

    def forward(self, t: int, h: int, w: int, device):
        N = t * h * w
        L = N + 2 
        idx_t, idx_h, idx_w = torch.arange(t, device=device), torch.arange(h, device=device), torch.arange(w, device=device)
        grid = torch.meshgrid([idx_t, idx_h, idx_w], indexing='ij')
        patch_coords = torch.stack(grid, dim=-1).reshape(N, 3) 
        all_coords = torch.cat([torch.tensor([[-1, -1, -1]], device=device), patch_coords, torch.tensor([[t, h, w]], device=device)], dim=0)

        indices_list = []
        for i in range(L):
            q_coord = all_coords[i]
            diff = (patch_coords - q_coord).abs()
            local_mask = (diff <= self.window_size).all(dim=-1)
            sparse_mask = torch.zeros(N, dtype=torch.bool, device=device)
            for p in range(self.max_pow):
                offset = 2 ** p
                for d in range(3):
                    match = (diff[:, d] == offset)
                    for od in range(3):
                        if od != d: match &= (diff[:, od] == 0)
                    sparse_mask |= match
            sampled_patch_indices = torch.where(local_mask | sparse_mask)[0] + 1
            full_idx = torch.cat([torch.tensor([0, L-1], device=device), sampled_patch_indices]).unique()
            indices_list.append(full_idx)

        K = max(len(idx) for idx in indices_list)
        idx_tensor, valid_mask = torch.zeros((L, K), dtype=torch.long, device=device), torch.zeros((L, K), dtype=torch.bool, device=device)
        for i, row in enumerate(indices_list):
            idx_tensor[i, :len(row)], valid_mask[i, :len(row)] = row, True
        return idx_tensor, valid_mask, all_coords

############################################################################################################
# 3D 几何先验生成器
############################################################################################################
class GeoPriorGen3D(nn.Module):
    def __init__(self, num_heads, initial_value=2, heads_range=4):
        super().__init__()
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer("decay", decay)

    def forward_sparse(self, all_coords, idx_tensor):
        sampled_coords = all_coords[idx_tensor]
        dist = (all_coords.unsqueeze(1) - sampled_coords).abs().sum(dim=-1)
        return dist.unsqueeze(0) * self.decay[:, None, None]

    def forward_dense(self, t, h, w):
        N = t * h * w
        idx_t, idx_h, idx_w = torch.arange(t, device=self.decay.device), torch.arange(h, device=self.decay.device), torch.arange(w, device=self.decay.device)
        grid = torch.meshgrid([idx_t, idx_h, idx_w], indexing='ij')
        grid = torch.stack(grid, dim=-1).reshape(N, 3)
        dist = (grid[:, None, :] - grid[None, :, :]).abs().sum(dim=-1)
        return dist.unsqueeze(0) * self.decay[:, None, None]

class SinCosPosEmbed(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, t, h, w, embed_dim):
        grid_t, grid_h, grid_w = torch.arange(t).float(), torch.arange(h).float(), torch.arange(w).float()
        grid = torch.meshgrid(grid_t, grid_h, grid_w, indexing='ij')
        grid = torch.stack(grid, dim=0).reshape([3, 1, t, h, w])
        emb_t = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])
        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])
        return torch.concatenate([emb_t, emb_h, emb_w], dim=1)
    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        omega = 1.0 / 10000**(torch.arange(embed_dim // 2).float() / (embed_dim / 2.0))
        out = torch.einsum("m,d->md", pos.reshape(-1), omega)
        return torch.concatenate([torch.sin(out), torch.cos(out)], dim=1)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size, self.patch_size = (img_size,)*3 if isinstance(img_size, int) else img_size, (patch_size,)*3 if isinstance(patch_size, int) else patch_size
        self.grid_size = (self.img_size[0]//self.patch_size[0], self.img_size[1]//self.patch_size[1], self.img_size[2]//self.patch_size[2])
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
    def patchify(self, x):
        pt, ph, pw, (t, h, w) = *self.patch_size, self.grid_size
        x = x.reshape(x.shape[0], 1, t, pt, h, ph, w, pw)
        return torch.einsum('nctphqwr->nthwpqrc', x).reshape(x.shape[0], t * h * w, -1)
    def forward(self, x): return self.proj(x).flatten(2).transpose(1, 2)

class Attention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.num_heads, self.head_dim = config.num_attention_heads, config.hidden_size // config.num_attention_heads
        self.q_proj, self.k_proj, self.v_proj = [nn.Linear(config.hidden_size, config.hidden_size, bias=False) for _ in range(3)]
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, True)
    def forward(self, x, attention_mask=None, sparse_params=None):
        bsz, q_len, _ = x.size()
        q, k, v = [proj(x).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2) for proj in [self.q_proj, self.k_proj, self.v_proj]]
        if HybridSparseAttnOn == 1 and sparse_params is not None:
            idx, valid, geo_bias, K = sparse_params['idx'], sparse_params['valid'], sparse_params.get('geo_bias'), sparse_params['idx'].size(1)
            idx_exp = idx[None, None, :, :, None].expand(bsz, self.num_heads, q_len, K, self.head_dim)
            k_sel, v_sel = [torch.gather(m.unsqueeze(-2).expand(-1, -1, -1, K, -1), dim=2, index=idx_exp) for m in [k, v]]
            logits = (q.unsqueeze(-2) * k_sel).sum(-1) / math.sqrt(self.head_dim)
            if geo_bias is not None: logits = logits + geo_bias.unsqueeze(0)
            logits.masked_fill_(~(valid & (idx <= torch.arange(q_len, device=idx.device)[:, None]))[None, None, :, :], float("-inf"))
            attn = logits.softmax(dim=-1)
            out = (attn.unsqueeze(-1) * v_sel).sum(-2)
        else:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=attention_mask is None and q_len > 1)
        return self.o_proj(out.transpose(1, 2).reshape(bsz, q_len, -1))

class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn, self.mlp = Attention(config, layer_idx), MLPBlock(config.hidden_size, config.intermediate_size, 0.0)
        self.ln1, self.ln2 = nn.LayerNorm(config.hidden_size), nn.LayerNorm(config.hidden_size)
    def forward(self, x, attention_mask=None, sparse_params=None):
        x = x + self.self_attn(self.ln1(x), attention_mask=attention_mask, sparse_params=sparse_params)
        return x + self.mlp(self.ln2(x))

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_size, self.patch_size = config.img_size, config.patch_size
        self.embed_tokens, self.patchifier = nn.Embedding(4, config.hidden_size), PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, embed_dim=config.hidden_size)
        self.layers, self.norm = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)]), nn.LayerNorm(config.hidden_size)
        if GeoPriorGen3DOn == 1: self.geo_prior_gen = GeoPriorGen3D(config.num_attention_heads)
        if HybridSparseAttnOn == 1: self.sparse_idx_gen = HybridSparseIndexGen3D()

    def forward(self, input_ids, input_image, attention_mask=None, **kwargs):
        bsz, seq_len = input_ids.shape[:2]
        input_image = input_image.reshape(bsz, 1, *self.img_size)
        t, h, w = self.patchifier.grid_size
        img_embeds = self.patchifier(input_image)
        x = torch.cat([self.embed_tokens(input_ids[:, :1]), img_embeds, self.embed_tokens(input_ids[:, -1:])], 1)

        # 处理输入端传入的扁平化 attention_mask
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask.reshape(bsz, 1, seq_len, seq_len)
            attention_mask = torch.zeros_like(attention_mask, dtype=x.dtype).masked_fill_(~attention_mask.bool(), torch.finfo(x.dtype).min)

        sparse_params = None
        if HybridSparseAttnOn == 1:
            idx, valid, coords = self.sparse_idx_gen(t, h, w, input_ids.device)
            sparse_params = {'idx': idx, 'valid': valid}
            if GeoPriorGen3DOn == 1: sparse_params['geo_bias'] = self.geo_prior_gen.forward_sparse(coords, idx)
        elif GeoPriorGen3DOn == 1:
            geo_bias = self.geo_prior_gen.forward_dense(t, h, w)
            full_geo_bias = torch.zeros((geo_bias.shape[0], seq_len, seq_len), device=input_ids.device)
            full_geo_bias[:, 1:-1, 1:-1] = geo_bias
            attention_mask = (attention_mask if attention_mask is not None else 0) + full_geo_bias.unsqueeze(0)

        for layer in self.layers: x = layer(x, attention_mask=attention_mask, sparse_params=sparse_params)
        return (self.norm(x),)

class ReconModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = BaseModel(config)
        self.decoder_pred = nn.Linear(config.hidden_size, math.prod(self.model.patch_size))
    def forward(self, input_ids, input_image, prefix_mask, **kwargs):
        hidden_states = self.model(input_ids, input_image, **kwargs)[0]
        logits = self.decoder_pred(hidden_states).float()
        shift_logits, labels = logits[:, :-2, :], self.model.patchifier.patchify(input_image.reshape(input_ids.shape[0], 1, *self.model.img_size))
        loss = ((shift_logits - labels) ** 2)
        loss = loss[prefix_mask[:, 1:-1] > 0].mean()
        return (loss, logits)
