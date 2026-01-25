import os
import sys
import random
import time
import numpy as np
import SimpleITK as sitk
import pydicom
from multiprocessing import Process
from monai.transforms import Compose, Resize, ScaleIntensityRangePercentiles

# 导入项目中已有的工具函数
from util import cut_patch

def load_and_scale_transforms(img, tar_img_size):
    """
    强度缩放处理，保持与项目其他预处理一致
    """
    transforms = Compose(
        [
            # 这里不使用 Resize，因为用户要求抛弃层数不足的，层数足够的直接切块
            ScaleIntensityRangePercentiles(lower=1., upper=99.9, b_min=0.0, b_max=1.0, clip=True, relative=False, channel_wise=False),
        ]
    )
    return transforms(img)

def is_ct_series(series_dir):
    """
    检查是否为 CT 模态且文件数是否足够
    """
    dicom_files = [f for f in os.listdir(series_dir) if f.endswith('.dcm')]
    if len(dicom_files) < 128:
        return False, 0
    
    # 读取第一张切片的元数据检查模态
    try:
        sample_path = os.path.join(series_dir, dicom_files[0])
        ds = pydicom.dcmread(sample_path, stop_before_pixels=True)
        if ds.Modality == 'CT':
            return True, len(dicom_files)
    except Exception:
        pass
    return False, 0

def process_one_series(series_dir, save_root, patch_num, tar_img_size):
    """
    处理单个 CT 序列：加载、缩放、切块、保存
    """
    try:
        # 使用 SimpleITK 读取整个 DICOM 序列以确保正确的 Z 轴顺序
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(series_dir)
        reader.SetFileNames(dicom_names)
        image_itk = reader.Execute()
        
        # 转换为 numpy (z, y, x)
        image_array = sitk.GetArrayFromImage(image_itk)
        
        # 检查最终形状是否满足 128x128x128
        if image_array.shape[0] < 128 or image_array.shape[1] < 128 or image_array.shape[2] < 128:
            return 0

        # 强度缩放 (针对整个 Volume)
        # Monai transforms 期望输入是 (C, H, W, D)
        image_array = np.expand_dims(image_array, 0) # -> (1, z, y, x)
        image_array = load_and_scale_transforms(image_array, tar_img_size).numpy()[0, ...] # -> (z, y, x)

        series_id = os.path.basename(series_dir)
        patient_id = os.path.basename(os.path.dirname(os.path.dirname(series_dir)))
        
        count = 0
        for i in range(patch_num):
            patch, _ = cut_patch(image_array, tar_img_size)
            
            # 检查 patch 形状确保万无一失 (cut_patch 在边缘可能返回不足尺寸的块)
            if patch.shape == tuple(tar_img_size):
                save_name = os.path.join(save_root, f"LIDC_{patient_id}_{series_id}_{i}.npy")
                np.save(save_name, patch)
                count += 1
        return count
    except Exception as e:
        print(f"Error processing {series_dir}: {e}")
        return 0

def main():
    if len(sys.argv) < 2:
        print("Usage: python proc_lidc_idri.py <lidc_idri_root_path>")
        return

    dataset_root = sys.argv[1]
    patch_num = 50
    tar_img_size = (128, 128, 128)
    
    # 设定输出目录
    # 尝试匹配项目结构，建议保存在 pretrain 数据目录下
    save_root = os.path.join(os.path.dirname(dataset_root), 'AR-SSL4M-DEMO', 'pretrain', 'data', 'patch_random_lidc')
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)

    print(f"Scanning dataset at: {dataset_root}")
    
    ct_series_dirs = []
    # 遍历 LIDC-IDRI 的三层目录结构
    for root, dirs, files in os.walk(dataset_root):
        if not dirs and files: # 叶子节点
            is_ct, num_slices = is_ct_series(root)
            if is_ct:
                ct_series_dirs.append(root)

    print(f"Found {len(ct_series_dirs)} qualified CT series (>=128 slices).")

    total_patches = 0
    start_time = time.time()
    
    for i, series_dir in enumerate(ct_series_dirs):
        t1 = time.time()
        num_saved = process_one_series(series_dir, save_root, patch_num, tar_img_size)
        total_patches += num_saved
        t2 = time.time()
        
        if i % 10 == 0:
            print(f"[{i+1}/{len(ct_series_dirs)}] Processed {series_dir}, generated {num_saved} patches. Time: {t2-t1:.2f}s")

    end_time = time.time()
    print(f"\nFinished! Total patches generated: {total_patches}")
    print(f"Total time: {(end_time - start_time) / 60:.2f} minutes")
    print(f"Patches saved to: {save_root}")

if __name__ == "__main__":
    main()
