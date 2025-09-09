#!/usr/bin/env python
"""
AR-SSL4M 预训练完整运行脚本
为导师准备的一键运行脚本

使用方法:
1. 创建conda环境: conda env create -f environment.yml
2. 激活环境: conda activate base (或者按照environment.yml中的名称)
3. 运行此脚本: python setup_and_run.py

或者直接运行: python setup_and_run.py --setup-env
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def setup_environment():
    """设置conda环境"""
    print("🔧 设置conda环境...")
    
    # 检查environment.yml是否存在
    if not os.path.exists("environment.yml"):
        print("❌ 错误: environment.yml文件不存在")
        return False
    
    try:
        # 创建conda环境
        print("创建conda环境...")
        subprocess.run(["conda", "env", "create", "-f", "environment.yml"], check=True)
        print("✅ Conda环境创建成功")
        return True
    except subprocess.CalledProcessError:
        print("❌ Conda环境创建失败")
        return False

def check_data():
    """检查数据是否存在"""
    print("📊 检查数据...")
    
    data_dir = Path("pretrain/demodata")
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return False
    
    npy_files = list(data_dir.glob("*.npy"))
    if len(npy_files) == 0:
        print("❌ 数据目录中没有.npy文件")
        return False
    
    print(f"✅ 找到 {len(npy_files)} 个数据文件")
    
    # 检查数据列表文件
    data_list_file = Path("pretrain/demodata_list.txt")
    if not data_list_file.exists():
        print("📝 创建数据列表文件...")
        with open(data_list_file, 'w', encoding='utf-8') as f:
            for npy_file in npy_files:
                f.write(f"demodata/{npy_file.name}\n")
        print("✅ 数据列表文件创建完成")
    
    return True

def check_environment():
    """检查Python环境和依赖"""
    print("🐍 检查Python环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查关键依赖
    required_packages = ['torch', 'numpy', 'transformers', 'monai', 'fire', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    
    return True

def run_pretraining():
    """运行预训练"""
    print("🚀 开始预训练...")
    
    # 确保在正确的目录
    os.chdir("pretrain")
    
    # 创建保存目录
    save_dir = Path("save")
    save_dir.mkdir(exist_ok=True)
    
    # 构建命令
    cmd = [
        sys.executable, "main.py",
        "--output_dir", "save",
        "--batch_size_training", "4"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        # 运行预训练
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 universal_newlines=True, bufsize=1)
        
        # 实时输出
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print("=" * 60)
            print("✅ 预训练完成!")
            print(f"模型保存在: pretrain/save/")
        else:
            print("=" * 60)
            print("❌ 预训练失败")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断训练")
        process.terminate()
        return False
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="AR-SSL4M 预训练运行脚本")
    parser.add_argument("--setup-env", action="store_true", help="设置conda环境")
    parser.add_argument("--check-only", action="store_true", help="只检查环境和数据")
    
    args = parser.parse_args()
    
    print("🎯 AR-SSL4M 预训练脚本")
    print("=" * 60)
    
    if args.setup_env:
        if not setup_environment():
            sys.exit(1)
        print("请激活环境后重新运行此脚本")
        return
    
    # 检查环境
    if not check_environment():
        print("\n💡 提示: 如果是首次运行，请先执行:")
        print("conda env create -f environment.yml")
        print("conda activate <环境名>")
        sys.exit(1)
    
    # 检查数据
    if not check_data():
        print("\n💡 请确保数据文件位于 pretrain/demodata/ 目录中")
        sys.exit(1)
    
    if args.check_only:
        print("✅ 所有检查通过!")
        return
    
    # 运行预训练
    print("=" * 60)
    input("按Enter键开始预训练，或Ctrl+C取消...")
    
    success = run_pretraining()
    
    if success:
        print("\n🎉 预训练成功完成!")
        print("📁 检查 pretrain/save/ 目录获取训练结果")
    else:
        print("\n❌ 预训练未成功完成")
        sys.exit(1)

if __name__ == "__main__":
    main()

