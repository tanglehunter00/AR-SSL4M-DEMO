import torch
import math

from typing import List, Optional, Tuple, Union
from torch import nn
from monai.networks.blocks.mlp import MLPBlock
from timm.models.layers import trunc_normal_

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa


class SinCosPosEmbed(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, t:int, h: int, w: int, embed_dim: int) -> torch.Tensor:
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
        pos_embed = torch.concatenate([emb_t, emb_h, emb_w], dim=1)  # (H*W, D)

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
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        pt, ph, pw = self.patch_size
        t, h, w = self.grid_size
        x = x.reshape(shape=(x.shape[0], 1, t, pt, h, ph, w, pw))
        x = torch.einsum('nctphqwr->nthwpqrc', x)
        x = x.reshape(shape=(x.shape[0], t * h * w, pt * ph * pw * 1))

        return x

    def forward(self, x: torch.Tensor):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class DilatedAttention(nn.Module):
    """
    膨胀注意力机制类
    
    作用：实现D-Former中的膨胀注意力，通过空洞机制扩大感受野
    原理：在全局自注意力计算中采用空洞方式，减少计算量同时保持长距离依赖
    """
    def __init__(self, config, layer_idx: Optional[int] = None, dilation_rate: int = 2):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dilation_rate = dilation_rate
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads

        # 检查隐藏层大小是否能被头数整除
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # 定义线性变换层
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        前向传播：计算膨胀多头自注意力
        
        参数：
        - hidden_states: 输入隐藏状态，形状为[batch_size, seq_len, hidden_size]
        - attention_mask: 注意力掩码，用于屏蔽某些位置
        
        返回：膨胀注意力输出，形状为[batch_size, seq_len, hidden_size]
        """
        bsz, q_len, _ = hidden_states.size()

        # 计算查询、键、值
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 重新整形为多头格式
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]

        # 应用膨胀机制：每隔dilation_rate个位置计算注意力
        if self.dilation_rate > 1:
            # 对query进行膨胀采样
            dilated_query_indices = torch.arange(0, q_len, self.dilation_rate, device=query_states.device)
            query_states = query_states[:, :, dilated_query_indices, :]
            
            # 对key和value进行膨胀采样
            dilated_kv_indices = torch.arange(0, kv_seq_len, self.dilation_rate, device=key_states.device)
            key_states = key_states[:, :, dilated_kv_indices, :]
            value_states = value_states[:, :, dilated_kv_indices, :]

        # 处理注意力掩码
        if attention_mask is not None:
            # 调整掩码以适应膨胀后的序列长度
            if self.dilation_rate > 1:
                dilated_mask_indices_q = torch.arange(0, attention_mask.size(-2), self.dilation_rate, device=attention_mask.device)
                dilated_mask_indices_kv = torch.arange(0, attention_mask.size(-1), self.dilation_rate, device=attention_mask.device)
                attention_mask = attention_mask[:, :, dilated_mask_indices_q, :]
                attention_mask = attention_mask[:, :, :, dilated_mask_indices_kv]
            
            if attention_mask.size() != (bsz, 1, query_states.size(-2), key_states.size(-2)):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, query_states.size(-2), key_states.size(-2))}, but is {attention_mask.size()}"
                )

        # 确保张量在GPU上是连续的
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # 计算缩放点积注意力
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0,
            is_causal=attention_mask is None and q_len > 1,
        )

        # 重新整形并应用输出投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # 如果使用了膨胀，需要将输出插值回原始长度
        if self.dilation_rate > 1:
            # 创建完整的输出张量
            full_output = torch.zeros(bsz, q_len, self.hidden_size, device=attn_output.device, dtype=attn_output.dtype)
            dilated_indices = torch.arange(0, q_len, self.dilation_rate, device=attn_output.device)
            full_output[:, dilated_indices, :] = attn_output.reshape(bsz, -1, self.hidden_size)
            attn_output = full_output
        else:
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        attn_output = self.o_proj(attn_output)
        return attn_output


class ResidualGating(nn.Module):
    """
    残差门控机制类（固定超参数版本）
    
    作用：控制原始注意力和膨胀注意力的权重比例
    原理：使用固定的超参数来强制约束两种注意力机制的权重比例
    
    参数说明：
    - hidden_size: 隐藏层大小
    - task_type: 任务类型（"pretrain"或"finetune"）
    - pretrain_dilated_ratio: 上游预训练任务中膨胀注意力的权重比例（默认0.01）
    - finetune_dilated_ratio: 下游微调任务中膨胀注意力的权重比例（默认0.01）
    """
    def __init__(self, hidden_size: int, task_type: str = "finetune", 
                 pretrain_dilated_ratio: float = 0.01, finetune_dilated_ratio: float = 0.01):
        super().__init__()
        self.task_type = task_type
        
        # 为不同任务设置固定的门控系数
        if task_type == "pretrain":
            # 上游预训练任务：使用固定超参数
            self.dilated_ratio = pretrain_dilated_ratio
            self.original_ratio = 1.0 - pretrain_dilated_ratio
        else:
            # 下游微调任务：使用固定超参数
            self.dilated_ratio = finetune_dilated_ratio
            self.original_ratio = 1.0 - finetune_dilated_ratio
        
        # 注册为buffer，这样不会参与梯度计算
        self.register_buffer('dilated_weight', torch.tensor(self.dilated_ratio))
        self.register_buffer('original_weight', torch.tensor(self.original_ratio))
    
    def forward(self, original_attention: torch.Tensor, dilated_attention: torch.Tensor):
        """
        前向传播：计算门控后的注意力输出
        
        参数：
        - original_attention: 原始注意力输出
        - dilated_attention: 膨胀注意力输出
        
        返回：门控后的注意力输出
        """
        # 使用固定的权重比例进行加权融合
        gated_output = self.dilated_weight * dilated_attention + self.original_weight * original_attention
        
        return gated_output


class Attention(nn.Module):
    """
    多头自注意力机制类（支持膨胀注意力）
    
    作用：让模型能够关注输入序列中的不同部分，理解它们之间的关系
    原理：通过计算查询(Q)、键(K)、值(V)之间的相似度来决定关注哪些位置
    新增：集成膨胀注意力机制和残差门控
    
    参数说明：
    - config: 模型配置，包含隐藏层大小、注意力头数等
    - layer_idx: 当前层在模型中的索引
    - task_type: 任务类型（"pretrain"或"finetune"）
    - use_dilated: 是否使用膨胀注意力
    """

    def __init__(self, config, layer_idx: Optional[int] = None, task_type: str = "finetune", use_dilated: bool = True,
                 pretrain_dilated_ratio: float = 0.01, finetune_dilated_ratio: float = 0.01):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.task_type = task_type
        self.use_dilated = use_dilated
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)
        
        # 如果使用膨胀注意力，添加膨胀注意力层和残差门控
        if self.use_dilated:
            self.dilated_attention = DilatedAttention(config, layer_idx, dilation_rate=2)
            self.residual_gating = ResidualGating(
                self.hidden_size, 
                task_type, 
                pretrain_dilated_ratio=pretrain_dilated_ratio,
                finetune_dilated_ratio=finetune_dilated_ratio
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        前向传播：计算多头自注意力（支持膨胀注意力）
        
        参数：
        - hidden_states: 输入隐藏状态，形状为[batch_size, seq_len, hidden_size]
        - attention_mask: 注意力掩码，用于屏蔽某些位置
        
        返回：注意力输出，形状为[batch_size, seq_len, hidden_size]
        """

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
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

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
        original_attn_output = self.o_proj(attn_output)
        
        # 如果使用膨胀注意力，计算膨胀注意力并应用残差门控
        if self.use_dilated:
            dilated_attn_output = self.dilated_attention(hidden_states, attention_mask)
            final_output = self.residual_gating(original_attn_output, dilated_attn_output)
            return final_output
        else:
            return original_attn_output


class DecoderLayer(nn.Module):
    """
    Transformer解码器层类（支持膨胀注意力）
    
    作用：构成Transformer模型的基本单元，包含自注意力机制和前馈网络
    原理：通过残差连接和层归一化，让信息在层间流动
    新增：支持膨胀注意力和残差门控
    
    参数说明：
    - config: 模型配置
    - layer_idx: 当前层索引
    - task_type: 任务类型（"pretrain"或"finetune"）
    - use_dilated: 是否使用膨胀注意力
    """
    def __init__(self, config, layer_idx: int, task_type: str = "finetune", use_dilated: bool = True,
                 pretrain_dilated_ratio: float = 0.01, finetune_dilated_ratio: float = 0.01):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.task_type = task_type
        self.use_dilated = use_dilated
        self.self_attn = Attention(
            config=config, 
            layer_idx=layer_idx, 
            task_type=task_type, 
            use_dilated=use_dilated,
            pretrain_dilated_ratio=pretrain_dilated_ratio,
            finetune_dilated_ratio=finetune_dilated_ratio
        )
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
    """
    基础Transformer模型类（支持膨胀注意力）
    
    作用：实现完整的Transformer架构，包含patch嵌入、位置编码、多层Transformer
    原理：将3D医学图像转换为序列，通过Transformer处理序列信息
    新增：支持膨胀注意力和残差门控
    
    参数说明：
    - config: 模型配置，包含所有超参数
    - task_type: 任务类型（"pretrain"或"finetune"）
    - use_dilated: 是否使用膨胀注意力
    """

    def __init__(self, config, task_type: str = "finetune", use_dilated: bool = True,
                 pretrain_dilated_ratio: float = 0.01, finetune_dilated_ratio: float = 0.01):
        super().__init__()
        self.pos_type = config.pos_type
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.task_type = task_type
        self.use_dilated = use_dilated
        self.embed_tokens = nn.Embedding(4, config.hidden_size)
        self.patchifier = PatchEmbed(embed_dim=config.hidden_size,
                                     in_chans=1,
                                     img_size=(self.img_size[0], self.img_size[1], self.img_size[2]),
                                     patch_size=(self.patch_size[0], self.patch_size[1], self.patch_size[2]))

        if self.pos_type == 'sincos3d':
            self.pos_embed = SinCosPosEmbed()
        elif self.pos_type == 'learnable':
            self.pos_embed = nn.Parameter(torch.zeros(self.img_size[0] * self.img_size[1] * self.img_size[2] //
                                                            self.patch_size[0] // self.patch_size[1] // self.patch_size[2] + 1, config.hidden_size))
            trunc_normal_(self.pos_embed, std=.02, a=-.02, b=.02)

        # Transformer层：多层解码器（支持膨胀注意力）
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx, task_type, use_dilated, pretrain_dilated_ratio, finetune_dilated_ratio) 
             for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        # self.init_proj()
        self.apply(self._init_weights)

    def init_proj(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d) following MAE
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

    def forward_patch_embedding(self, input_image):
        batch_size, seq_length = input_image.shape[0], 1 + (self.img_size[0] // self.patch_size[0]) * \
                                 (self.img_size[1] // self.patch_size[1]) * (self.img_size[2] // self.patch_size[2])

        input_ids = torch.empty(batch_size, 1, dtype=torch.int64, device=input_image.device).fill_(1)
        image_embeds = self.patchifier(input_image)
        t, h, w = self.patchifier.grid_size
        embed_dim = self.patchifier.embed_dim
        starts_embeds = self.embed_tokens(input_ids[..., :1])
        image_embeds = torch.cat((starts_embeds, image_embeds), 1)

        if self.pos_type == 'sincos3d':
            pos_embed = self.pos_embed(t, h, w, embed_dim)
            pos_embed = torch.concatenate([torch.zeros([1, embed_dim]), pos_embed], dim=0)
        elif self.pos_type == 'learnable':
            pos_embed = self.pos_embed

        pos_embed = pos_embed.to(image_embeds.device)
        inputs_embeds = image_embeds + pos_embed[None, ...]

        attention_mask = torch.ones(batch_size, 1, t * h * w + 1, t * h * w + 1, dtype=torch.bool).to(image_embeds.device)
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            0,
        )
        return attention_mask, inputs_embeds

    def forward(
        self,
        input_image,
    ):
        attention_mask, inputs_embeds = self.forward_patch_embedding(input_image)

        # embed positions
        hidden_states = inputs_embeds
        # decoder layers
        all_hidden_states = ()
        for decoder_layer in self.layers:
            all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs[0]
        hidden_states = self.norm(hidden_states)
        all_hidden_states += (hidden_states,)

        return hidden_states[:, 1:, :], [x[:, 1:, :] for x in all_hidden_states[1:]]
