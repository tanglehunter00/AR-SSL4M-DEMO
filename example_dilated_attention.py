# D-Former膨胀注意力集成示例
# 展示如何在AR-SSL4M项目中使用膨胀注意力和残差门控

import torch
import torch.nn as nn
from pretrain.model import ReconModel, BaseModel, Attention, DilatedAttention, ResidualGating

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

def example_usage():
    """
    展示如何使用集成了D-Former膨胀注意力的AR-SSL4M模型
    """
    
    # 创建配置
    config = ModelConfig()
    
    print("=== D-Former膨胀注意力集成示例 ===")
    print()
    
    # 1. 预训练模型示例（上游任务）
    print("1. 预训练模型（上游任务）:")
    print("   - 膨胀注意力占比: 0.01")
    print("   - 原始注意力占比: 0.99")
    print("   - 任务类型: pretrain")
    print()
    
    pretrain_model = ReconModel(
        config=config,
        task_type="pretrain",  # 上游预训练任务
        use_dilated=True       # 启用膨胀注意力
    )
    
    # 模拟输入数据
    batch_size = 2
    input_image = torch.randn(batch_size, 1, 128, 128, 128)
    input_ids = torch.tensor([[1, 3, 3, 3, 2], [1, 3, 3, 3, 2]])  # 开始token + 图像tokens + 结束token
    prefix_mask = torch.ones(batch_size, 5)  # 所有位置都需要预测
    
    # 前向传播
    with torch.no_grad():
        loss, logits = pretrain_model(
            input_ids=input_ids,
            input_image=input_image,
            prefix_mask=prefix_mask
        )
    
    print(f"   预训练模型输出形状: {logits.shape}")
    print(f"   重建损失: {loss.item():.4f}")
    print()
    
    # 2. 下游分割模型示例（微调任务）
    print("2. 下游分割模型（微调任务）:")
    print("   - 膨胀注意力占比: 0.01")
    print("   - 原始注意力占比: 0.99")
    print("   - 任务类型: finetune")
    print()
    
    # 导入下游模型
    from downstream.segmentation.models.dilated_base_model import BaseModel as DilatedBaseModel
    
    finetune_model = DilatedBaseModel(
        config=config,
        task_type="finetune",  # 下游微调任务
        use_dilated=True        # 启用膨胀注意力
    )
    
    # 前向传播
    with torch.no_grad():
        hidden_states, all_hidden_states = finetune_model(input_image)
    
    print(f"   下游模型输出形状: {hidden_states.shape}")
    print(f"   隐藏状态层数: {len(all_hidden_states)}")
    print()
    
    # 3. 膨胀注意力机制详细说明
    print("3. 膨胀注意力机制特点:")
    print("   - 膨胀率: 2（每隔2个位置计算注意力）")
    print("   - 计算复杂度: O(N²/4) 而不是 O(N²)")
    print("   - 感受野: 扩大2倍")
    print("   - 长距离依赖: 更好地捕获全局信息")
    print()
    
    # 4. 残差门控机制说明
    print("4. 残差门控机制:")
    print("   - 门控函数: Sigmoid")
    print("   - 上游门控系数: 0.01（可学习）")
    print("   - 下游门控系数: 0.01（可学习）")
    print("   - 输出: gate * dilated_attention + (1-gate) * original_attention")
    print()
    
    # 5. 模型参数统计
    print("5. 模型参数统计:")
    pretrain_params = sum(p.numel() for p in pretrain_model.parameters())
    finetune_params = sum(p.numel() for p in finetune_model.parameters())
    
    print(f"   预训练模型参数数量: {pretrain_params:,}")
    print(f"   下游模型参数数量: {finetune_params:,}")
    print()
    
    # 6. 使用建议
    print("6. 使用建议:")
    print("   - 预训练: 使用ReconModel进行自监督学习")
    print("   - 微调: 使用DilatedBaseModel进行下游任务")
    print("   - 门控系数: 可以根据任务调整膨胀注意力的权重")
    print("   - 膨胀率: 可以根据序列长度调整膨胀率")
    print()
    
    print("=== 示例完成 ===")

if __name__ == "__main__":
    example_usage()
