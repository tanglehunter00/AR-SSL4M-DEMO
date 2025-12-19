import torch
import warnings
warnings.filterwarnings('ignore')

from timm.models.layers import trunc_normal_
from monai.networks.blocks.mlp import MLPBlock
from typing import Optional, Tuple, Union
from torch import nn

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa

GeoPriorGen3DOn = 1 # 0: 不使用3D几何先验，1: 使用3D几何先验


############################################################################################################
# 此处进行了修改，新增了3D几何先验生成器
############################################################################################################
class GeoPriorGen3D(nn.Module):
    def __init__(self, num_heads, initial_value=2, heads_range=4):
        super().__init__()
        # decay calculation based on DFormerV2
        # decay is a negative value: log(1 - 2^(-x)) < 0
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        self.register_buffer("decay", decay)

    def forward(self, t: int, h: int, w: int):
        # Generate 3D Grid
        idx_t = torch.arange(t).to(self.decay.device)
        idx_h = torch.arange(h).to(self.decay.device)
        idx_w = torch.arange(w).to(self.decay.device)
        
        # (t, h, w, 3)
        grid = torch.meshgrid([idx_t, idx_h, idx_w], indexing='ij')
        grid = torch.stack(grid, dim=-1).reshape(t * h * w, 3) # (N, 3)
        
        # Calculate 3D Relative Manhattan Distance
        # (N, 1, 3) - (1, N, 3) -> (N, N, 3)
        dist = grid[:, None, :] - grid[None, :, :]
        dist = dist.abs().sum(dim=-1) # (N, N)
        
        # Apply decay
        # self.decay: (NumHeads,)
        # dist: (N, N)
        # result: (NumHeads, N, N)
        # Since decay is negative, larger distance -> more negative bias
        bias = dist.unsqueeze(0) * self.decay[:, None, None]
        
        return bias


class SinCosPosEmbed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t: int, h: int, w: int, embed_dim: int) -> torch.Tensor:
        assert embed_dim % 3 == 0, embed_dim
        grid_t = torch.arange(t).float()
        grid_h = torch.arange(h).float()
        grid_w = torch.arange(w).float()

        grid = torch.meshgrid(grid_t, grid_h, grid_w)
        grid = torch.stack(grid, dim=0)
        grid = grid.reshape([3, 1, t, h, w])

        emb_t = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])
        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])
        pos_embed = torch.concatenate([emb_t, emb_h, emb_w], dim=1)

        return pos_embed

    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(
        embed_dim: int, pos: torch.Tensor
    ) -> torch.Tensor:
        omega = torch.arange(embed_dim // 2).float()
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

        emb_sin = torch.sin(out)  # (M, D/2)
        emb_cos = torch.cos(out)  # (M, D/2)

        emb = torch.concatenate([emb_sin, emb_cos], dim=1)  # (M, D)
        return emb


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]] = 64,
        patch_size: Union[int, Tuple[int, int, int]] = 8,
        in_chans: int = 1,
        embed_dim: int = 768,
    ):
        super().__init__()
        img_size = (
            (img_size, img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        )
        patch_size = (
            (patch_size, patch_size, patch_size)
            if isinstance(patch_size, int)
            else tuple(patch_size)
        )

        self.img_size, self.embed_dim = img_size, embed_dim
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def patchify(self, x):
        pt, ph, pw = self.patch_size
        t, h, w = self.grid_size
        x = x.reshape(shape=(x.shape[0], 1, t, pt, h, ph, w, pw))
        x = torch.einsum('nctphqwr->nthwpqrc', x)
        x = x.reshape(shape=(x.shape[0], t * h * w, pt * ph * pw * 1))
        return x

    def forward(self, x: torch.Tensor):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                ##########################################################################################################
                # 此处进行了修改，修改报错信息，不重要
                ##########################################################################################################
                #raise ValueError(
                #    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                #)
                if GeoPriorGen3DOn == 1:
                    if attention_mask.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                        raise ValueError(
                            f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)} or {(bsz, self.num_heads, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                        )
                else:
                    # 原报错信息在这里是不做进一步校验的，也就是说原报错更容易触发，新报错更不容易触发
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                ##########################################################################################################
                # 此处进行了修改，修改止步于此
                ##########################################################################################################

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config=config, layer_idx=layer_idx)
        self.mlp = MLPBlock(config.hidden_size, config.intermediate_size, 0.0)
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        return outputs


class BaseModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.pos_type = config.pos_type
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.embed_tokens = nn.Embedding(4, config.hidden_size)
        self.patchifier = PatchEmbed(embed_dim=config.hidden_size,
                                     img_size=(self.img_size[0], self.img_size[1], self.img_size[2]),
                                     patch_size=(self.patch_size[0], self.patch_size[1], self.patch_size[2]))
        if self.pos_type == 'sincos3d':
            self.pos_embed = SinCosPosEmbed()
        elif self.pos_type == 'learnable':
            self.pos_embed_learn = nn.Parameter(torch.zeros(self.img_size[0] * self.img_size[1] * self.img_size[2] //
                                                            self.patch_size[0] // self.patch_size[1] // self.patch_size[2] + 2, config.hidden_size))
            trunc_normal_(self.pos_embed_learn, std=.02)

        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.init_proj()
        ##########################################################################################################
        # 此处进行了修改，初始化3D几何先验生成器
        ##########################################################################################################
        # Init Geometry Prior Generator
        if GeoPriorGen3DOn == 1:
            self.geo_prior_gen = GeoPriorGen3D(config.num_attention_heads)
        ##########################################################################################################
        # 修改止步于此
        ##########################################################################################################
        self.apply(self._init_weights)

    def init_proj(self):
        if hasattr(self.patchifier, "proj"):
            w = self.patchifier.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_image=None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ):

        batch_size, seq_length = input_ids.shape[:2]
        past_key_values_length = 0
        input_image = input_image.reshape(batch_size, 1, self.img_size[0], self.img_size[1], self.img_size[2])
        image_embeds = self.patchifier(input_image)
        t, h, w = self.patchifier.grid_size
        embed_dim = self.patchifier.embed_dim

        starts_embeds = self.embed_tokens(input_ids[...,:1])
        ends_embeds = self.embed_tokens(input_ids[...,-1:])
        image_embeds = torch.cat((starts_embeds, image_embeds, ends_embeds), 1)
        if self.pos_type == 'sincos3d':
            pos_embed = self.pos_embed(t, h, w, embed_dim)
            pos_embed = torch.concatenate([torch.zeros([1, embed_dim]), pos_embed], dim=0)
            pos_embed = torch.concatenate([pos_embed, torch.zeros([1, embed_dim])], dim=0)
        elif self.pos_type == 'learnable':
            pos_embed = self.pos_embed_learn
        pos_embed = pos_embed.to(image_embeds.device)
        inputs_embeds = image_embeds + pos_embed[None, ...]

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(batch_size, 1, t * h * w + 2, t * h * w + 2)
            # Convert bool/int mask (1 for attend, 0 for mask) to float additive mask (0.0 for attend, min_float for mask)
            # This is required by transformers' _prepare_4d_causal_attention_mask_for_sdpa when input is 4D
            attention_mask_bool = attention_mask.bool()
            attention_mask = torch.zeros(
                attention_mask.shape, 
                dtype=inputs_embeds.dtype, 
                device=attention_mask.device
            )
            attention_mask.masked_fill_(~attention_mask_bool, torch.finfo(inputs_embeds.dtype).min)
        else:
            # Fallback: create a fully visible mask (0.0 everywhere)
            attention_mask = torch.zeros(
                batch_size, 1, t * h * w + 2, t * h * w + 2,
                dtype=inputs_embeds.dtype,
                device=input_image.device
            )

        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
        ##########################################################################################################
        # 此处进行了修改，不知道改的什么
        ##########################################################################################################
        # --- 应用 3D 几何先验 Bias ---
        if GeoPriorGen3DOn == 1:
            # 生成 3D 几何先验 Bias，形状为 (NumHeads, N, N)，其中 N = t*h*w 是 Patch 的总数
            geo_bias = self.geo_prior_gen(t, h, w)
        
            # 对 Bias 进行填充以匹配序列长度 (N+2)
            # 序列包含 [StartToken, Patch_1, ..., Patch_N, EndToken]
            # 我们只将几何衰减应用于 Patch-to-Patch 的注意力上，不影响特殊 Token
            full_geo_bias = torch.zeros(
                (geo_bias.shape[0], seq_length, seq_length),
                dtype=inputs_embeds.dtype,
                device=attention_mask.device
            )
            # 将生成的几何 Bias 填充到中间的 Patch 区域 [1:-1, 1:-1]
            full_geo_bias[:, 1:-1, 1:-1] = geo_bias
        
            # 将几何 Bias 加到现有的 Attention Mask 上
            # attention_mask 形状: (Batch, 1, SeqLen, SeqLen) - 包含了因果/前缀掩码
            # full_geo_bias 形状: (NumHeads, SeqLen, SeqLen) - 包含了距离衰减偏置
            # 利用广播机制合并: (Batch, NumHeads, SeqLen, SeqLen)
            # 最终效果：保留了原有的可见性约束(Mask)，同时为可见区域加上了随距离衰减的权重 Bias
            attention_mask = attention_mask + full_geo_bias.unsqueeze(0)
            # ------------------------------------
        ##########################################################################################################
        # 修改止步于此
        ##########################################################################################################

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return tuple(v for v in [hidden_states])


class ReconModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BaseModel(config)
        self.patch_size = self.model.patchifier.patch_size
        self.norm_pixel_loss = config.norm_pixel_loss
        self.img_size = config.img_size
        self.decoder_pred = nn.Linear(config.hidden_size, self.patch_size[0] * self.patch_size[1] * self.patch_size[2], bias=True)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_image=None,
        prefix_mask=None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ):

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            input_image=input_image,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs[0]
        logits = self.decoder_pred(hidden_states)
        logits = logits.float()
        loss = None

        shift_logits = logits[..., :-2, :].contiguous()
        input_image = input_image.reshape(input_image.shape[0], 1, self.img_size[0], self.img_size[1], self.img_size[2])
        shift_labels = self.model.patchifier.patchify(input_image)

        prefix_mask = prefix_mask[..., 1:-1, None]
        prefix_mask = prefix_mask.repeat_interleave(shift_labels.shape[-1], axis=-1)

        if self.norm_pixel_loss:
            mean = shift_labels.mean(dim=-1, keepdim=True)
            var = shift_labels.var(dim=-1, keepdim=True)
            shift_labels = (shift_labels - mean) / (var + 1.e-6) ** .5

        shift_labels = shift_labels.to(shift_logits.device)
        loss = (shift_logits - shift_labels) ** 2
        loss = loss[prefix_mask > 0]
        loss = loss.mean()  # [N, L], mean loss per patch

        output = (logits,)
        return (loss,) + output if loss is not None else output
