import torch
import math
import warnings
warnings.filterwarnings('ignore')

from timm.models.layers import trunc_normal_
from monai.networks.blocks.mlp import MLPBlock
from typing import Optional, Tuple, Union
from torch import nn

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa

GeoPriorGen3DOn = 0     # 0: 不使用3D几何先验，1: 使用3D几何先验
HybridSparseAttnOn = 0  # 0: 不使用混合稀疏注意力，1: 使用混合稀疏注意力

############################################################################################################
# 新增：3D 混合稀疏索引生成器 (Local Window + Global Dilated)
############################################################################################################
class HybridSparseIndexGen3D(nn.Module):
    def __init__(self, window_size=2, max_pow=3):
        super().__init__()
        self.window_size = window_size
        self.max_pow = max_pow

    def forward(self, t: int, h: int, w: int, device):
        N = t * h * w
        L = N + 2 # 包含 Start 和 End Token
        
        # 1. 生成 3D 坐标网格 (针对 N 个 Patch)
        idx_t = torch.arange(t, device=device)
        idx_h = torch.arange(h, device=device)
        idx_w = torch.arange(w, device=device)
        grid = torch.meshgrid([idx_t, idx_h, idx_w], indexing='ij')
        patch_coords = torch.stack(grid, dim=-1).reshape(N, 3) # (N, 3)
        
        # 为 Start (index 0) 和 End (index N+1) 定义虚拟坐标
        # 设置在边界之外，确保它们能通过几何逻辑进行区分
        all_coords = torch.cat([
            torch.tensor([[-1, -1, -1]], device=device),
            patch_coords,
            torch.tensor([[t, h, w]], device=device)
        ], dim=0) # (L, 3)

        indices_list = []
        for i in range(L):
            # 获取当前 Query 的坐标
            q_coord = all_coords[i]
            
            # 计算到所有 Patch 的距离 (只针对 Patch 部分进行采样)
            diff = (patch_coords - q_coord).abs()
            
            # 逻辑 A: 区域采样 (Local)
            local_mask = (diff <= self.window_size).all(dim=-1)
            
            # 逻辑 B: 全局稀疏采样 (Dilated)
            sparse_mask = torch.zeros(N, dtype=torch.bool, device=device)
            for p in range(self.max_pow):
                offset = 2 ** p
                for d in range(3):
                    match = (diff[:, d] == offset)
                    for od in range(3):
                        if od != d: match &= (diff[:, od] == 0)
                    sparse_mask |= match
            
            # 合并索引
            sampled_patch_indices = torch.where(local_mask | sparse_mask)[0] + 1 # 转换回全序列索引
            
            # 强制包含全局 Token (Start 和 End)
            full_idx = torch.cat([
                torch.tensor([0, L-1], device=device), 
                sampled_patch_indices
            ]).unique()
            indices_list.append(full_idx)

        # 对齐索引长度 K
        K = max(len(idx) for idx in indices_list)
        idx_tensor = torch.zeros((L, K), dtype=torch.long, device=device)
        valid_mask = torch.zeros((L, K), dtype=torch.bool, device=device)
        
        for i, row in enumerate(indices_list):
            idx_tensor[i, :len(row)] = row
            valid_mask[i, :len(row)] = True
            
        return idx_tensor, valid_mask, all_coords

############################################################################################################
# 3D 几何先验生成器 (优化：支持稀疏偏置计算)
############################################################################################################
class GeoPriorGen3D(nn.Module):
    def __init__(self, num_heads, initial_value=2, heads_range=4):
        super().__init__()
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        self.register_buffer("decay", decay)

    def forward_sparse(self, all_coords, idx_tensor):
        """
        all_coords: (L, 3)
        idx_tensor: (L, K)
        """
        L, K = idx_tensor.shape
        # 提取采样点的坐标: (L, K, 3)
        sampled_coords = all_coords[idx_tensor]
        
        # 计算曼哈顿距离: (L, K)
        dist = (all_coords.unsqueeze(1) - sampled_coords).abs().sum(dim=-1)
        
        # 应用衰减: (NumHeads, L, K)
        bias = dist.unsqueeze(0) * self.decay[:, None, None]
        return bias

    def forward_dense(self, t, h, w):
        # 原有的全图计算逻辑，保留用于非稀疏模式
        N = t * h * w
        idx_t = torch.arange(t, device=self.decay.device)
        idx_h = torch.arange(h, device=self.decay.device)
        idx_w = torch.arange(w, device=self.decay.device)
        grid = torch.meshgrid([idx_t, idx_h, idx_w], indexing='ij')
        grid = torch.stack(grid, dim=-1).reshape(N, 3)
        dist = (grid[:, None, :] - grid[None, :, :]).abs().sum(dim=-1)
        return dist.unsqueeze(0) * self.decay[:, None, None]


class SinCosPosEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t: int, h: int, w: int, embed_dim: int) -> torch.Tensor:
        assert embed_dim % 3 == 0, embed_dim
        grid_t = torch.arange(t).float()
        grid_h = torch.arange(h).float()
        grid_w = torch.arange(w).float()
        grid = torch.meshgrid(grid_t, grid_h, grid_w)
        grid = torch.stack(grid, dim=0).reshape([3, 1, t, h, w])
        emb_t = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])
        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])
        return torch.concatenate([emb_t, emb_h, emb_w], dim=1)

    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
        omega = 1.0 / 10000**(torch.arange(embed_dim // 2).float() / (embed_dim / 2.0))
        out = torch.einsum("m,d->md", pos.reshape(-1), omega)
        return torch.concatenate([torch.sin(out), torch.cos(out)], dim=1)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.grid_size = (self.img_size[0]//self.patch_size[0], self.img_size[1]//self.patch_size[1], self.img_size[2]//self.patch_size[2])
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def patchify(self, x):
        pt, ph, pw = self.patch_size
        t, h, w = self.grid_size
        x = x.reshape(x.shape[0], 1, t, pt, h, ph, w, pw)
        x = torch.einsum('nctphqwr->nthwpqrc', x)
        return x.reshape(x.shape[0], t * h * w, pt * ph * pw)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sparse_params: Optional[dict] = None # 传入稀疏计算所需的参数
    ):
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 稀疏采样注意力逻辑
        if HybridSparseAttnOn == 1 and sparse_params is not None:
            idx = sparse_params['idx']      # (L, K)
            valid = sparse_params['valid']  # (L, K)
            geo_bias = sparse_params.get('geo_bias') # (NumHeads, L, K)
            K = idx.size(1)

            # 1. Gather K, V: [B, H, L, K, Dh]
            # 扩展索引以匹配多头和 Batch
            idx_exp = idx[None, None, :, :, None].expand(bsz, self.num_heads, q_len, K, self.head_dim)
            k_sel = torch.gather(k.unsqueeze(-2).expand(-1, -1, -1, K, -1), dim=2, index=idx_exp)
            v_sel = torch.gather(v.unsqueeze(-2).expand(-1, -1, -1, K, -1), dim=2, index=idx_exp)

            # 2. 计算 Logits: [B, H, L, K]
            logits = (q.unsqueeze(-2) * k_sel).sum(-1) / math.sqrt(self.head_dim)

            # 3. 注入几何先验
            if geo_bias is not None:
                logits = logits + geo_bias.unsqueeze(0)

            # 4. 应用有效位掩码和因果遮挡
            # 因果遮挡: sampled_index <= query_index
            causal_mask = (idx <= torch.arange(q_len, device=idx.device)[:, None])
            final_valid = valid & causal_mask
            logits.masked_fill_(~final_valid[None, None, :, :], float("-inf"))

            # 5. Softmax & Output
            attn = logits.softmax(dim=-1)
            attn_output = (attn.unsqueeze(-1) * v_sel).sum(-2) # [B, H, L, Dh]
            
        else:
            # 标准稠密注意力逻辑
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, is_causal=attention_mask is None and q_len > 1
            )

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config=config, layer_idx=layer_idx)
        self.mlp = MLPBlock(config.hidden_size, config.intermediate_size, 0.0)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x, attention_mask=None, sparse_params=None):
        x = x + self.self_attn(self.ln1(x), attention_mask=attention_mask, sparse_params=sparse_params)
        x = x + self.mlp(self.ln2(x))
        return (x,)


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.embed_tokens = nn.Embedding(4, config.hidden_size)
        self.patchifier = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, embed_dim=config.hidden_size)
        
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        
        if GeoPriorGen3DOn == 1:
            self.geo_prior_gen = GeoPriorGen3D(config.num_attention_heads)
        if HybridSparseAttnOn == 1:
            self.sparse_idx_gen = HybridSparseIndexGen3D(window_size=2, max_pow=3)

    def forward(self, input_ids, input_image, attention_mask=None, **kwargs):
        bsz, seq_len = input_ids.shape[:2]
        t, h, w = self.patchifier.grid_size
        
        # Embeddings
        img_embeds = self.patchifier(input_image)
        img_embeds = torch.cat([self.embed_tokens(input_ids[:, :1]), img_embeds, self.embed_tokens(input_ids[:, -1:])], 1)
        
        # 稀疏计算准备
        sparse_params = None
        if HybridSparseAttnOn == 1:
            idx, valid, coords = self.sparse_idx_gen(t, h, w, input_ids.device)
            sparse_params = {'idx': idx, 'valid': valid}
            if GeoPriorGen3DOn == 1:
                sparse_params['geo_bias'] = self.geo_prior_gen.forward_sparse(coords, idx)
        elif GeoPriorGen3DOn == 1:
            # 稠密模式下的几何先验处理
            geo_bias = self.geo_prior_gen.forward_dense(t, h, w)
            full_geo_bias = torch.zeros((geo_bias.shape[0], seq_len, seq_len), device=input_ids.device)
            full_geo_bias[:, 1:-1, 1:-1] = geo_bias
            attention_mask = (attention_mask if attention_mask is not None else 0) + full_geo_bias.unsqueeze(0)

        x = img_embeds
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, sparse_params=sparse_params)[0]
        
        return (self.norm(x),)

class ReconModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = BaseModel(config)
        self.decoder_pred = nn.Linear(config.hidden_size, math.prod(self.model.patch_size))

    def forward(self, input_ids, input_image, prefix_mask, **kwargs):
        hidden_states = self.model(input_ids, input_image, **kwargs)[0]
        logits = self.decoder_pred(hidden_states).float()
        
        # Loss calculation (simplified)
        shift_logits = logits[:, 1:-1, :]
        labels = self.model.patchifier.patchify(input_image)
        loss = ((shift_logits - labels) ** 2)
        loss = loss[prefix_mask[:, 1:-1] > 0].mean()
        
        return (loss, logits)