#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import nncf
import openvino as ov
from openvino.runtime import Core
from nncf import quantize
import argparse

# 默认配置路径
DEFAULT_MODEL_PATH = "/home/guo/best_red.xml"  # 您的YOLOv8 XML模型路径
DEFAULT_DATASET_PATH = "/home/guo/Calibration_datasets/calibration"  # 修改为新的图像目录
DEFAULT_OUTPUT_DIR = "/home/guo/Calibration"      # 输出目录
DEFAULT_FP32_MODEL_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "best_red_fp32.xml")
DEFAULT_INT8_MODEL_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "best_red_int8.xml")

# 预处理函数 - 适用于YOLOv8
def preprocess_image(image_path, input_size=(640, 640)):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 保存原始图像尺寸用于后处理
    orig_h, orig_w = img.shape[:2]
    
    # 调整大小，保持长宽比，填充
    ratio = min(input_size[0] / orig_w, input_size[1] / orig_h)
    new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
    
    # 调整大小
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 创建画布并将调整大小后的图像放在中心
    canvas = np.full((input_size[1], input_size[0], 3), 114, dtype=np.uint8)
    offset_x, offset_y = (input_size[0] - new_w) // 2, (input_size[1] - new_h) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
    
    # BGR -> RGB并转换为float
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    
    # 归一化 [0, 255] -> [0, 1]
    canvas = canvas.astype(np.float32) / 255.0
    
    # 添加批次维度并转换为NCHW
    canvas = np.transpose(canvas, (2, 0, 1))
    canvas = np.expand_dims(canvas, 0)
    
    return canvas

# 创建数据加载器以用于模型量化（校准）
def get_calibration_dataset(dataset_path, input_size=(640, 640), num_samples=250, model_path=None):
    image_files = []
    
    # 如果没有指定模型路径，使用全局默认路径
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    
    # 检查路径是否存在
    if not os.path.exists(dataset_path):
        print(f"警告：数据集路径 {dataset_path} 不存在")
        return None
    
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(Path(dataset_path).rglob(f'*{ext}')))
    
    # 检查是否找到图像
    if len(image_files) == 0:
        print(f"警告：在 {dataset_path} 中未找到图像文件")
        return None
    
    # 限制校准图像的数量
    if len(image_files) > num_samples:
        import random
        random.shuffle(image_files)
        image_files = image_files[:num_samples]
    
    print(f"使用 {len(image_files)} 张图像进行校准")
    
    # 获取模型输入名称
    core = Core()
    model = core.read_model(model_path)
    input_name = model.input().get_any_name()
    
    # 定义转换函数
    def transform_fn(img_path):
        # 处理单个图像路径
        preprocessed_img = preprocess_image(str(img_path), input_size)
        return {input_name: preprocessed_img}
    
    # 返回NNCF数据集
    if len(image_files) > 0:
        return nncf.Dataset(image_files, transform_fn)
    else:
        return None

# 执行INT8量化
def quantize_model(model_path, dataset, output_path):
    print("开始INT8量化...")
    
    # 从XML中读取模型
    core = Core()
    model = core.read_model(model_path)
    
    # 检查数据集是否有效
    if dataset is None:
        print("校准数据集为空，无法进行量化")
        return None
    
    # 使用NNCF的quantize函数执行量化
    try:
        print(f"开始量化过程...")
        quantized_model = quantize(
            model=model,
            calibration_dataset=dataset,
            preset=nncf.QuantizationPreset.PERFORMANCE  # 性能优先预设
        )
        
        # 保存量化后的模型
        ov.save_model(quantized_model, output_path)
        print(f"量化模型已保存到 {output_path}")
        
        return quantized_model
    except Exception as e:
        print(f"量化过程发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='YOLOv8模型量化工具')
    
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH, 
                        help='原始模型路径 (.xml)')
    parser.add_argument('--dataset', default=DEFAULT_DATASET_PATH,
                        help='校准数据集图像目录')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR,
                        help='输出目录路径')
    parser.add_argument('--fp32', default=DEFAULT_FP32_MODEL_PATH,
                        help='FP32模型输出路径')
    parser.add_argument('--int8', default=DEFAULT_INT8_MODEL_PATH,
                        help='INT8模型输出路径')
    parser.add_argument('--num_samples', type=int, default=250,
                        help='校准使用的图像数量 (默认: 250)')
    
    args = parser.parse_args()
    
    # 打印参数
    print("=" * 50)
    print("YOLOv8模型量化")
    print("=" * 50)
    print(f"原始模型: {args.model}")
    print(f"校准数据集: {args.dataset}")
    print(f"输出目录: {args.output_dir}")
    print(f"FP32模型: {args.fp32}")
    print(f"INT8模型: {args.int8}")
    print("=" * 50)
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"错误：模型路径 {args.model} 不存在")
        print("请修改--model参数为您的YOLOv8 XML模型文件的正确路径")
        return
    
    # 复制FP32模型到输出目录
    import shutil
    shutil.copy(args.model, args.fp32)
    bin_path = args.model.replace('.xml', '.bin')
    if not os.path.exists(bin_path):
        print(f"错误：模型权重文件 {bin_path} 不存在")
        return
        
    shutil.copy(bin_path, args.fp32.replace('.xml', '.bin'))
    print(f"已复制FP32模型到 {args.fp32}")
    
    # 获取校准数据集
    calibration_dataset = get_calibration_dataset(
        args.dataset, 
        input_size=(640, 640), 
        num_samples=args.num_samples,
        model_path=args.model  # 传入模型路径参数，不使用全局变量
    )
    
    # 如果校准数据集有效，执行量化
    if calibration_dataset:
        quantized_model = quantize_model(args.fp32, calibration_dataset, args.int8)
        
        if quantized_model:
            # 检查模型大小
            if os.path.exists(args.int8.replace('.xml', '.bin')):
                fp32_size = os.path.getsize(args.fp32.replace('.xml', '.bin')) / (1024 * 1024)
                int8_size = os.path.getsize(args.int8.replace('.xml', '.bin')) / (1024 * 1024)
                size_reduction = (fp32_size - int8_size) / fp32_size * 100
                
                print(f"\n模型大小信息:")
                print(f"FP32模型大小: {fp32_size:.2f} MB")
                print(f"INT8模型大小: {int8_size:.2f} MB")
                print(f"大小减少: {size_reduction:.2f}%")
                
                print("\n量化完成! 请使用 accuracy_benchmark.py 脚本评估模型性能和精度:")
                print(f"python accuracy_benchmark.py --fp32 {args.fp32} --int8 {args.int8} --images {args.dataset} --labels /home/guo/Calibration_datasets/labels")
            else:
                print("量化异常: 无法获取INT8模型大小信息 - 文件不存在")
    else:
        print("警告：校准数据集无效，无法执行量化")
        print("请确保校准数据集目录中包含图像文件")

if __name__ == "__main__":
    main()
