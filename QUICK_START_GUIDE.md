# AR-SSL4M 快速开始指南 - 给导师

## 🚀 最快启动方式

### Windows用户
1. 双击运行 `run_pretraining.bat`
2. 等待训练完成（约1小时）

### Linux/Mac用户 或 喜欢Python的用户
```bash
python setup_and_run.py
```

## 📋 文件清单
为导师准备的完整文件包:

### 环境配置
- `environment.yml` - Conda环境配置文件
- `requirements.txt` - pip依赖列表（备用）

### 运行脚本
- `setup_and_run.py` - Python集成脚本（推荐）
- `run_pretraining.bat` - Windows批处理脚本

### 文档
- `README_for_supervisor.md` - 详细说明文档
- `QUICK_START_GUIDE.md` - 本快速指南

### 项目文件
- `pretrain/` - 预训练代码目录
- `pretrain/demodata/` - 数据文件（137个.npy文件）

## ⚡ 一分钟快速检查

运行前请确认:
1. ✅ 有CUDA GPU
2. ✅ 有4GB+显存
3. ✅ 有网络连接
4. ✅ `pretrain/demodata/` 中有137个.npy文件

## 🎯 预期结果

训练完成后会看到:
- `pretrain/save/checkpoint.pth` - 训练好的模型
- 控制台显示训练完成信息
- GPU内存使用约2-3GB

## ❓ 如果出问题

1. **环境问题**: 运行 `python setup_and_run.py --check-only`
2. **CUDA问题**: 检查 `nvidia-smi`
3. **其他问题**: 查看 `README_for_supervisor.md`

---

**预计时间**: 环境设置5分钟 + 训练1小时 = 总共1小时5分钟

**一句话总结**: 运行 `python setup_and_run.py` 然后等1小时即可 ✨



