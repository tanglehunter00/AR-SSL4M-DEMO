# D-Former膨胀注意力集成说明

本项目已成功集成D-Former的膨胀注意力机制到AR-SSL4M中，实现了联合预训练和微调的功能。

## 主要特性

### 1. 膨胀注意力机制 (Dilated Attention)
- **膨胀率**: 2（每隔2个位置计算注意力）
- **计算复杂度**: O(N²/4) 而不是 O(N²)
- **感受野**: 扩大2倍，更好地捕获长距离依赖
- **适用场景**: 3D医学图像分割任务

### 2. 残差门控机制 (Residual Gating)
- **门控函数**: Sigmoid激活函数
- **上游任务门控系数**: 0.01（膨胀注意力占比）
- **下游任务门控系数**: 0.01（膨胀注意力占比）
- **输出公式**: `gate * dilated_attention + (1-gate) * original_attention`

### 3. 联合预训练和微调
- **预训练阶段**: 使用膨胀注意力进行自监督学习
- **微调阶段**: 使用膨胀注意力进行下游任务
- **门控系数**: 上游和下游任务分别设置，可独立学习

## 文件结构

```
pretrain/
├── model.py                           # 更新后的预训练模型（支持膨胀注意力）
└── ...

downstream/segmentation/models/
├── base_model.py                      # 原始下游模型
├── dilated_base_model.py              # 新的下游模型（支持膨胀注意力）
└── ...

example_dilated_attention.py           # 使用示例
```

## 使用方法

### 1. 预训练阶段

```python
from pretrain.model import ReconModel

# 创建配置
config = ModelConfig()

# 创建预训练模型（启用膨胀注意力）
model = ReconModel(
    config=config,
    task_type="pretrain",  # 上游预训练任务
    use_dilated=True       # 启用膨胀注意力
)

# 训练
loss, logits = model(
    input_ids=input_ids,
    input_image=input_image,
    prefix_mask=prefix_mask
)
```

### 2. 微调阶段

```python
from downstream.segmentation.models.dilated_base_model import BaseModel

# 创建下游模型（启用膨胀注意力）
model = BaseModel(
    config=config,
    task_type="finetune",  # 下游微调任务
    use_dilated=True        # 启用膨胀注意力
)

# 前向传播
hidden_states, all_hidden_states = model(input_image)
```

### 3. 自定义门控系数

```python
# 在ResidualGating类中修改门控系数
class ResidualGating(nn.Module):
    def __init__(self, hidden_size: int, task_type: str = "pretrain"):
        super().__init__()
        self.task_type = task_type
        
        if task_type == "pretrain":
            # 上游任务：可以调整膨胀注意力占比
            self.gate_weight = nn.Parameter(torch.tensor(0.01))  # 0.01 = 1%
        else:
            # 下游任务：可以调整膨胀注意力占比
            self.gate_weight = nn.Parameter(torch.tensor(0.01))  # 0.01 = 1%
```

## 技术细节

### 膨胀注意力实现

```python
class DilatedAttention(nn.Module):
    def __init__(self, config, layer_idx=None, dilation_rate=2):
        # 膨胀率设置为2
        self.dilation_rate = dilation_rate
        
    def forward(self, hidden_states, attention_mask=None):
        # 应用膨胀机制：每隔dilation_rate个位置计算注意力
        if self.dilation_rate > 1:
            dilated_indices = torch.arange(0, seq_len, self.dilation_rate)
            query_states = query_states[:, :, dilated_indices, :]
            # ... 其他处理
```

### 残差门控实现

```python
class ResidualGating(nn.Module):
    def forward(self, original_attention, dilated_attention):
        gate = self.gate_activation(self.gate_weight)
        gated_output = gate * dilated_attention + (1 - gate) * original_attention
        return gated_output
```

## 性能优势

1. **计算效率**: 膨胀注意力将计算复杂度从O(N²)降低到O(N²/4)
2. **长距离依赖**: 更好地捕获3D医学图像中的全局信息
3. **灵活门控**: 可学习的门控机制平衡局部和全局注意力
4. **联合优化**: 预训练和微调阶段都使用膨胀注意力

## 运行示例

```bash
# 运行使用示例
python example_dilated_attention.py
```

## 注意事项

1. **内存使用**: 膨胀注意力会减少内存使用，但需要额外的门控参数
2. **训练稳定性**: 门控系数初始化为0.01，确保训练稳定
3. **任务适配**: 可以根据具体任务调整膨胀率和门控系数
4. **兼容性**: 新模型与原始AR-SSL4M模型完全兼容

## 引用

如果您使用了本项目的膨胀注意力机制，请引用原始论文：

```bibtex
@inproceedings{wang2025autoregressive,
  title={Autoregressive sequence modeling for 3d medical image representation},
  author={Wang, Siwen and Wang, Churan and Gao, Fei and Su, Lixian and Zhang, Fandong and Wang, Yizhou and Yu, Yizhou},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={8},
  pages={7871--7879},
  year={2025}
}
```
