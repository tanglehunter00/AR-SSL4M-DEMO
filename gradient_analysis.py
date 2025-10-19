# 门控机制梯度传播详细分析

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class ResidualGating(nn.Module):
    def __init__(self, hidden_size: int, task_type: str = "pretrain"):
        super().__init__()
        self.task_type = task_type
        
        # 门控参数：初始化为0.01
        self.gate_weight = nn.Parameter(torch.tensor(0.01))
        self.gate_activation = nn.Sigmoid()
    
    def forward(self, original_attention, dilated_attention):
        # 计算门控权重
        gate = self.gate_activation(self.gate_weight)
        
        # 加权融合
        gated_output = gate * dilated_attention + (1 - gate) * original_attention
        
        return gated_output

def analyze_gradient_flow():
    """
    分析门控机制的梯度传播
    """
    print("=== 门控机制梯度传播分析 ===")
    print()
    
    # 创建门控模块
    gating = ResidualGating(hidden_size=768, task_type="pretrain")
    
    # 模拟输入
    batch_size, seq_len, hidden_size = 2, 100, 768
    original_attention = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    dilated_attention = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    
    print("1. 初始门控参数:")
    print(f"   gate_weight (原始值): {gating.gate_weight.item():.6f}")
    
    # 前向传播
    gate = gating.gate_activation(gating.gate_weight)
    print(f"   gate (Sigmoid后): {gate.item():.6f}")
    print(f"   膨胀注意力权重: {gate.item():.6f}")
    print(f"   原始注意力权重: {(1-gate).item():.6f}")
    print()
    
    # 计算门控输出
    gated_output = gating(original_attention, dilated_attention)
    
    # 模拟损失（简化）
    loss = gated_output.sum()
    
    print("2. 梯度计算:")
    print("   损失函数: L = sum(gated_output)")
    print("   门控输出: gated_output = gate * dilated_attention + (1-gate) * original_attention")
    print()
    
    # 反向传播
    loss.backward()
    
    print("3. 梯度分析:")
    print(f"   gate_weight梯度: {gating.gate_weight.grad.item():.6f}")
    print(f"   original_attention梯度范数: {original_attention.grad.norm().item():.6f}")
    print(f"   dilated_attention梯度范数: {dilated_attention.grad.norm().item():.6f}")
    print()
    
    # 分析梯度传播
    print("4. 梯度传播公式:")
    print("   ∂L/∂gate_weight = ∂L/∂gated_output * ∂gated_output/∂gate_weight")
    print("   ∂gated_output/∂gate_weight = sigmoid'(gate_weight) * (dilated_attention - original_attention)")
    print("   ∂L/∂original_attention = ∂L/∂gated_output * (1 - gate)")
    print("   ∂L/∂dilated_attention = ∂L/∂gated_output * gate")
    print()
    
    # 验证梯度公式
    sigmoid_derivative = gate * (1 - gate)  # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    expected_gate_grad = sigmoid_derivative * (dilated_attention - original_attention).sum()
    
    print("5. 梯度验证:")
    print(f"   理论gate_weight梯度: {expected_gate_grad.item():.6f}")
    print(f"   实际gate_weight梯度: {gating.gate_weight.grad.item():.6f}")
    print(f"   梯度误差: {abs(expected_gate_grad.item() - gating.gate_weight.grad.item()):.8f}")
    print()
    
    return gating, gate, gated_output

def demonstrate_training_dynamics():
    """
    演示训练过程中门控参数的变化
    """
    print("=== 训练过程中门控参数变化演示 ===")
    print()
    
    # 创建门控模块
    gating = ResidualGating(hidden_size=768, task_type="pretrain")
    optimizer = torch.optim.Adam([gating.gate_weight], lr=0.001)
    
    print("训练步骤 | gate_weight | gate值 | 膨胀权重 | 原始权重")
    print("-" * 60)
    
    for step in range(10):
        # 模拟训练数据
        original_attention = torch.randn(2, 100, 768)
        dilated_attention = torch.randn(2, 100, 768)
        
        # 前向传播
        gated_output = gating(original_attention, dilated_attention)
        
        # 模拟损失（假设膨胀注意力更好，所以损失更小）
        loss = (gated_output - dilated_attention).pow(2).mean()
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算当前门控值
        gate = gating.gate_activation(gating.gate_weight)
        
        print(f"   {step:2d}    | {gating.gate_weight.item():8.6f} | {gate.item():6.4f} | {gate.item():8.4f} | {(1-gate).item():8.4f}")
    
    print()
    print("观察：门控参数会根据损失函数自动调整，学习最优的权重比例")

def explain_parameter_update():
    """
    解释参数更新机制
    """
    print("=== 参数更新机制详解 ===")
    print()
    
    print("1. 门控参数更新:")
    print("   - gate_weight是可学习参数，通过反向传播更新")
    print("   - 初始值: 0.01")
    print("   - 实际门控值: Sigmoid(0.01) ≈ 0.5025")
    print("   - 这意味着初始时膨胀注意力和原始注意力权重几乎相等")
    print()
    
    print("2. 梯度传播路径:")
    print("   损失 → gated_output → gate → gate_weight")
    print("   损失 → gated_output → original_attention (权重: 1-gate)")
    print("   损失 → gated_output → dilated_attention (权重: gate)")
    print()
    
    print("3. 权重分配:")
    print("   - 不是固定的0.99和0.01比例")
    print("   - 而是可学习的动态权重")
    print("   - 模型会根据任务需求自动调整权重")
    print()
    
    print("4. 实际权重计算:")
    print("   - 膨胀注意力权重 = Sigmoid(gate_weight)")
    print("   - 原始注意力权重 = 1 - Sigmoid(gate_weight)")
    print("   - 两个权重之和始终为1")
    print()

if __name__ == "__main__":
    # 运行分析
    gating, gate, output = analyze_gradient_flow()
    print()
    demonstrate_training_dynamics()
    print()
    explain_parameter_update()
