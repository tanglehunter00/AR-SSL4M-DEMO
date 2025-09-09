@echo off
echo ========================================
echo AR-SSL4M 预训练启动脚本
echo ========================================

REM 检查是否在正确的目录
if not exist "pretrain\main.py" (
    echo 错误: 请在项目根目录运行此脚本
    pause
    exit /b 1
)

REM 激活conda环境（如果需要）
echo 激活conda环境...
call conda activate base

REM 进入预训练目录
cd pretrain

REM 创建保存目录
if not exist "save" mkdir save

REM 开始预训练
echo 开始预训练...
echo ========================================
python main.py --output_dir save --batch_size_training 4

REM 训练完成
echo ========================================
echo 预训练完成！检查 pretrain\save 目录获取结果
pause


