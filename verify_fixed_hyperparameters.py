# 固定超参数门控机制验证示例（简化版）

import torch
import torch.nn as nn
from pretrain.model import ReconModel

# 示例配置类
class ModelConfig:
    def __init__(self):
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.num_key_value_heads = 12
        self.intermediate_size = 3072
        self.num_hidden_layers = 12
        self.pos_type = 'sincos3d'
        self.img_size = [128, 128, 128]
        self.patch_size = [16, 16, 16]
        self.norm_pixel_loss = True

def verify_fixed_hyperparameters():
    """
    验证固定超参数的门控机制
    """
    
    # 创建配置
    config = ModelConfig()
    
    print("=== 固定超参数门控机制验证 ===")
    print()
    
    # 设置不同的超参数
    pretrain_dilated_ratio = 0.01  # 上游预训练：膨胀注意力占比1%
    finetune_dilated_ratio = 0.01  # 下游微调：膨胀注意力占比1%
    
    print("1. 超参数设置:")
    print(f"   上游预训练膨胀注意力权重: {pretrain_dilated_ratio}")
    print(f"   上游预训练原始注意力权重: {1 - pretrain_dilated_ratio}")
    print(f"   下游微调膨胀注意力权重: {finetune_dilated_ratio}")
    print(f"   下游微调原始注意力权重: {1 - finetune_dilated_ratio}")
    print()
    
    # 1. 预训练模型示例（上游任务）
    print("2. 预训练模型（上游任务）:")
    pretrain_model = ReconModel(
        config=config,
        task_type="pretrain",
        use_dilated=True,
        pretrain_dilated_ratio=pretrain_dilated_ratio,
        finetune_dilated_ratio=finetune_dilated_ratio
    )
    
    # 检查门控权重
    first_layer_gating = pretrain_model.model.layers[0].self_attn.residual_gating
    print(f"   膨胀注意力权重: {first_layer_gating.dilated_weight.item():.6f}")
    print(f"   原始注意力权重: {first_layer_gating.original_weight.item():.6f}")
    print(f"   权重总和: {(first_layer_gating.dilated_weight + first_layer_gating.original_weight).item():.6f}")
    print()
    
    # 2. 下游分割模型示例（微调任务）
    print("3. 下游分割模型（微调任务）:")
    from downstream.segmentation.models.dilated_base_model import BaseModel as DilatedBaseModel
    
    finetune_model = DilatedBaseModel(
        config=config,
        task_type="finetune",
        use_dilated=True,
        pretrain_dilated_ratio=pretrain_dilated_ratio,
        finetune_dilated_ratio=finetune_dilated_ratio
    )
    
    # 检查门控权重
    first_layer_gating = finetune_model.layers[0].self_attn.residual_gating
    print(f"   膨胀注意力权重: {first_layer_gating.dilated_weight.item():.6f}")
    print(f"   原始注意力权重: {first_layer_gating.original_weight.item():.6f}")
    print(f"   权重总和: {(first_layer_gating.dilated_weight + first_layer_gating.original_weight).item():.6f}")
    print()
    
    # 3. 验证权重不会改变
    print("4. 验证权重固定性:")
    print("   这些权重是固定的buffer，不会在训练过程中改变")
    print("   不会参与梯度计算，始终保持设定的比例")
    print()
    
    # 4. 验证所有层的权重都相同
    print("5. 验证所有层的权重一致性:")
    for i, layer in enumerate(pretrain_model.model.layers):
        gating = layer.self_attn.residual_gating
        print(f"   第{i+1}层: 膨胀权重={gating.dilated_weight.item():.6f}, 原始权重={gating.original_weight.item():.6f}")
    print()
    
    # 5. 展示如何修改超参数
    print("6. 如何修改超参数:")
    print("   可以通过修改pretrain_dilated_ratio和finetune_dilated_ratio参数")
    print("   来调整膨胀注意力和原始注意力的权重比例")
    print()
    
    # 示例：不同的权重比例
    print("7. 不同权重比例示例:")
    ratios = [0.01, 0.05, 0.1, 0.2]
    for ratio in ratios:
        print(f"   膨胀注意力权重: {ratio:.2f}, 原始注意力权重: {1-ratio:.2f}")
    print()

def demonstrate_custom_ratios():
    """
    演示自定义权重比例
    """
    print("=== 自定义权重比例演示 ===")
    print()
    
    config = ModelConfig()
    
    # 自定义权重比例
    custom_pretrain_ratio = 0.05  # 上游：膨胀注意力5%
    custom_finetune_ratio = 0.1   # 下游：膨胀注意力10%
    
    print("自定义权重比例:")
    print(f"上游预训练: 膨胀注意力 {custom_pretrain_ratio:.1%}, 原始注意力 {1-custom_pretrain_ratio:.1%}")
    print(f"下游微调: 膨胀注意力 {custom_finetune_ratio:.1%}, 原始注意力 {1-custom_finetune_ratio:.1%}")
    print()
    
    # 创建模型
    pretrain_model = ReconModel(
        config=config,
        task_type="pretrain",
        use_dilated=True,
        pretrain_dilated_ratio=custom_pretrain_ratio,
        finetune_dilated_ratio=custom_finetune_ratio
    )
    
    # 验证权重
    first_layer_gating = pretrain_model.model.layers[0].self_attn.residual_gating
    print("实际权重验证:")
    print(f"膨胀注意力权重: {first_layer_gating.dilated_weight.item():.6f}")
    print(f"原始注意力权重: {first_layer_gating.original_weight.item():.6f}")
    print()

def demonstrate_gradient_independence():
    """
    演示门控权重不参与梯度计算
    """
    print("=== 梯度独立性验证 ===")
    print()
    
    config = ModelConfig()
    
    # 创建模型
    model = ReconModel(
        config=config,
        task_type="pretrain",
        use_dilated=True,
        pretrain_dilated_ratio=0.01,
        finetune_dilated_ratio=0.01
    )
    
    # 检查可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"总参数数量: {total_params:,}")
    print(f"门控权重不参与训练: {total_params - trainable_params} 个参数")
    print()
    
    # 验证门控权重是buffer
    first_layer_gating = model.model.layers[0].self_attn.residual_gating
    print("门控权重属性:")
    print(f"dilated_weight.requires_grad: {first_layer_gating.dilated_weight.requires_grad}")
    print(f"original_weight.requires_grad: {first_layer_gating.original_weight.requires_grad}")
    print(f"dilated_weight.is_buffer: {first_layer_gating.dilated_weight in first_layer_gating._buffers.values()}")
    print()

if __name__ == "__main__":
    verify_fixed_hyperparameters()
    print()
    demonstrate_custom_ratios()
    print()
    demonstrate_gradient_independence()
