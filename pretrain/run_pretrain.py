#!/usr/bin/env python
"""
Windows兼容的预训练启动脚本
"""
import os
import sys
import subprocess

def main():
    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 切换到pretrain目录
    os.chdir("pretrain")
    
    # 构建命令
    cmd = [
        sys.executable, "main.py",
        "--output_dir", "save",
        "--batch_size_training", "4"
    ]
    
    print("启动预训练...")
    print(f"命令: {' '.join(cmd)}")
    print(f"工作目录: {os.getcwd()}")
    
    # 执行命令
    try:
        result = subprocess.run(cmd, check=True)
        print("预训练完成!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"预训练失败: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("用户中断了预训练")
        return 1

if __name__ == "__main__":
    sys.exit(main())
