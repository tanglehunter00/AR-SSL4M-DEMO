# AR-SSL4M 预训练项目 - 导师运行指南

## 项目简介
这是AR-SSL4M (Autoregressive Sequence Modeling for 3D Medical Image Representation) 的预训练项目复现。项目使用自回归序列建模来学习3D医学图像表示。

## 快速开始

### 方法1: 一键运行脚本 (推荐)
```bash
# 1. 创建并激活conda环境
conda env create -f environment.yml
conda activate base  # 或按照environment.yml中的环境名

# 2. 运行集成脚本
python setup_and_run.py
```

### 方法2: 手动步骤
```bash
# 1. 环境设置
conda env create -f environment.yml
conda activate base

# 2. 进入预训练目录
cd pretrain

# 3. 开始预训练
python main.py --output_dir save --batch_size_training 4
```

## 文件结构
```
AR-SSL4M-DEMO/
├── environment.yml              # Conda环境配置
├── setup_and_run.py            # 一键运行脚本
├── README_for_supervisor.md     # 本文档
├── pretrain/                    # 预训练代码
│   ├── main.py                 # 主训练脚本
│   ├── model.py                # 模型定义
│   ├── demodata/               # 数据目录 (137个.npy文件)
│   ├── demodata_list.txt       # 数据列表文件
│   ├── save/                   # 模型保存目录
│   └── configs/                # 配置文件
└── downstream/                  # 下游任务代码
```

## 环境要求
- **Python**: 3.8+ (当前使用3.13)
- **CUDA**: 支持CUDA的GPU
- **内存**: 建议16GB+ RAM
- **GPU内存**: 建议4GB+ VRAM

## 关键依赖包
- torch>=2.1.2
- transformers>=4.37.0
- monai>=1.3.0
- numpy
- fire
- tqdm

## 训练配置
- **模型大小**: 91.3M参数
- **图像尺寸**: 128×128×128
- **Patch尺寸**: 16×16×16
- **批大小**: 4
- **训练轮数**: 5 (测试用，可在configs/training.py中调整)
- **学习率**: 1e-4
- **优化器**: AdamW

## 数据说明
- **数据集**: RSNA-CSFD (137个预处理的3D医学图像)
- **格式**: .npy文件，每个文件128×128×128
- **位置**: `pretrain/demodata/`
- **大小**: 每个文件约8MB

## 预期输出
训练完成后，以下文件将保存在`pretrain/save/`目录：
- `checkpoint.pth`: 模型权重
- 训练日志和指标

## 性能监控
训练期间会显示：
- 每个step的损失值
- 内存使用情况
- 训练进度条
- 验证损失

## 故障排除

### 常见问题
1. **CUDA内存不足**: 减少batch_size (在configs/training.py中修改)
2. **数据文件未找到**: 确保demodata目录中有.npy文件
3. **依赖包缺失**: 运行`pip install -r requirements.txt`

### 调试模式
如果需要调试，可以：
```bash
# 只检查环境和数据
python setup_and_run.py --check-only

# 查看详细错误信息
cd pretrain
python main.py --output_dir save --batch_size_training 4 --verbose
```

## 预训练时间估算
- **单个epoch**: 约10-15分钟 (取决于硬件)
- **总训练时间**: 约1小时 (5个epochs)

## 联系信息
如有问题，请联系学生或查看原始论文：
- 原始项目: https://github.com/serena9525/AR-SSL4M
- 论文: "Autoregressive sequence modeling for 3d medical image representation" (AAAI 2025)

## 注意事项
1. 训练过程中请保持网络连接稳定
2. 建议在训练期间不要进行其他GPU密集型任务
3. 模型文件较大，请确保有足够的存储空间
4. 首次运行可能需要下载一些依赖包

---
*此项目仅用于学术研究目的，请尊重原作者版权*

