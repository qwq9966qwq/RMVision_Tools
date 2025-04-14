#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import argparse
from openvino.runtime import Core

# 新增导入，用于精度评估
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 默认配置路径
DEFAULT_FP32_MODEL_PATH = "/home/guo/Calibration/best_red_fp32.xml"
DEFAULT_INT8_MODEL_PATH = "/home/guo/Calibration/best_red_int8.xml"
DEFAULT_IMAGES_DIR = "/home/guo/Calibration_datasets/image_red"
DEFAULT_LABELS_DIR = "/home/guo/Calibration_datasets/labels_red"
DEFAULT_OUTPUT_DIR = "/home/guo/Calibration/benchmark_results_red"

# 模型类别定义
CATEGORIES = [
    {"id": 0, "name": "armor_blue"},
    {"id": 1, "name": "armor_red"},
]

# 添加模型缓存
_model_cache = {}

def get_compiled_model(model_path, device="CPU"):
    """获取编译好的模型，使用缓存避免重复加载"""
    key = f"{model_path}_{device}"
    if key not in _model_cache:
        core = Core()
        model = core.read_model(model_path)
        _model_cache[key] = {
            "model": core.compile_model(model, device),
            "input_layer": None,
            "output_layer": None
        }
        _model_cache[key]["input_layer"] = _model_cache[key]["model"].input(0)
        _model_cache[key]["output_layer"] = _model_cache[key]["model"].output(0)
    
    return _model_cache[key]

# 从sub_openvino_blue.py复制的辅助函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2]/2
    y[..., 1] = x[..., 1] - x[..., 3]/2
    y[..., 2] = x[..., 0] + x[..., 2]/2
    y[..., 3] = x[..., 1] + x[..., 3]/2
    return y

# 预处理函数
def preprocess_image(image_path, input_size=(640, 640)):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 保存原始图像尺寸用于后处理
    orig_h, orig_w = img.shape[:2]
    img_height, img_width = orig_h, orig_w
    
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
    
    return canvas, (orig_h, orig_w), (offset_y, offset_x), ratio, img_height, img_width

# 模型推理函数
def inference(model_path, image_path, device="CPU"):
    # 获取编译好的模型（使用缓存）
    compiled_model_info = get_compiled_model(model_path, device)
    compiled_model = compiled_model_info["model"]
    output_layer = compiled_model_info["output_layer"]
    input_layer = compiled_model_info["input_layer"]
    
    # 读取和预处理图像
    preprocessed_img, orig_shape, pad, ratio, img_height, img_width = preprocess_image(image_path)
    
    # 测量推理时间
    start_time = time.time()
    result = compiled_model([preprocessed_img])[output_layer]
    inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
    
    # 获取输入尺寸
    input_height = input_layer.shape[2]
    input_width = input_layer.shape[3]
    
    return inference_time, result, orig_shape, pad, ratio, img_height, img_width, input_height, input_width

# 修改为与sub_openvino_blue.py一致的处理逻辑
def process_yolo_output(output, orig_shape, pad, ratio, img_height, img_width, input_height, input_width, conf_threshold=0.53, iou_threshold=0.5):
    """
    以sub_openvino_blue.py的方式处理YOLOv8输出
    """
    # 如同原始代码一样处理输出
    predictions = np.squeeze(output, axis=0).T
    
    # 提取框和类别信息
    boxes = predictions[:, :4]
    class_scores = sigmoid(predictions[:, 4:])
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)
    
    # 过滤低置信度检测
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    
    # 如果没有检测通过阈值，返回空列表
    if len(confidences) == 0:
        return [], [], []
    
    # 将中心坐标转换为角点坐标
    boxes_xyxy = xywh2xyxy(boxes)
    
    # 应用缩放比例
    scale_w = img_width / input_width
    scale_h = img_height / input_height
    boxes_xyxy[:, [0, 2]] *= scale_w
    boxes_xyxy[:, [1, 3]] *= scale_h
    
    # 转换为整数类型 - 与原始代码保持一致
    boxes_xyxy = boxes_xyxy.astype(np.int32)
    
    # 转换为列表用于处理
    boxes_list = boxes_xyxy.tolist()
    scores_list = confidences.tolist()
    
    # 初始化最终结果
    final_boxes = []
    final_confidences = []
    final_class_ids = []
    
    # 按类别应用NMS - 与sub_openvino_blue.py一致
    unique_classes = np.unique(class_ids)
    for cls in unique_classes:
        cls_mask = (class_ids == cls)
        cls_boxes = [boxes_list[i] for i in range(len(class_ids)) if cls_mask[i]]
        cls_scores = [scores_list[i] for i in range(len(class_ids)) if cls_mask[i]]
        
        if len(cls_boxes) == 0:
            continue
        
        # 转换为宽高格式用于NMS
        cls_boxes_xywh = []
        for box in cls_boxes:
            x1, y1, x2, y2 = box
            cls_boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
        
        # 应用NMS
        indices = cv2.dnn.NMSBoxes(cls_boxes_xywh, cls_scores, conf_threshold, iou_threshold)
        
        if len(indices) > 0:
            # 处理OpenCV不同版本的返回结果差异
            if isinstance(indices, tuple):
                indices = indices[0]
            elif isinstance(indices, np.ndarray) and indices.ndim == 2:
                indices = indices.flatten()
            elif isinstance(indices, list) and isinstance(indices[0], np.ndarray):
                indices = [i[0] for i in indices]
            
            # 添加到最终结果
            for i in indices:
                final_boxes.append(cls_boxes[i])
                final_confidences.append(cls_scores[i])
                final_class_ids.append(cls)
    
    return final_boxes, final_confidences, final_class_ids

# 推理并获取检测结果
def get_detections(model_path, image_path, conf_threshold=0.53, iou_threshold=0.5, device="CPU"):
    """
    在图像上运行推理并返回处理后的检测结果
    """
    # 运行推理
    inference_time, result, orig_shape, pad, ratio, img_height, img_width, input_height, input_width = inference(model_path, image_path, device)
    
    # 处理输出获取边界框、分数和类别
    boxes, scores, class_ids = process_yolo_output(
        result, orig_shape, pad, ratio, img_height, img_width, input_height, input_width,
        conf_threshold=conf_threshold, 
        iou_threshold=iou_threshold
    )
    
    return boxes, scores, class_ids

# 创建COCO格式的预测结果
def create_coco_predictions(image_list, model_path, output_file, conf_threshold=0.53, iou_threshold=0.5, device="CPU"):
    """创建COCO格式的模型预测结果"""
    predictions = []
    
    for img_id, img_path in enumerate(tqdm(image_list, desc=f"执行{os.path.basename(model_path)}推理")):
        # 获取检测结果
        boxes, scores, class_ids = get_detections(
            model_path, img_path, 
            conf_threshold=conf_threshold, 
            iou_threshold=iou_threshold, 
            device=device
        )
        
        # 转换为COCO格式
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # COCO格式使用[x,y,width,height]
            coco_box = [float(x1), float(y1), float(width), float(height)]
            
            prediction = {
                "image_id": img_id,
                "category_id": int(class_id),  # 保持与原始类别ID一致
                "bbox": coco_box,
                "score": float(score)
            }
            
            predictions.append(prediction)
    
    # 保存到文件
    with open(output_file, 'w') as f:
        json.dump(predictions, f)
    
    print(f"已保存预测结果到 {output_file}")
    return predictions

# 从YOLO格式标注创建COCO格式真实标签
def create_coco_ground_truth(image_list, labels_dir, output_file, categories):
    """从YOLO格式标注文件创建COCO格式真实标签"""
    images = []
    annotations = []
    ann_id = 0
    
    for img_id, img_path in enumerate(tqdm(image_list, desc="处理标注文件")):
        # 获取图像尺寸
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图像 {img_path}")
            continue
            
        height, width = img.shape[:2]
        
        # 获取文件名和基本名称
        filename = os.path.basename(img_path)
        basename = os.path.splitext(filename)[0]
        
        # 添加图像条目
        image_info = {
            "id": img_id,
            "file_name": filename,
            "height": height,
            "width": width
        }
        images.append(image_info)
        
        # 获取对应的标注文件
        ann_file = os.path.join(labels_dir, f"{basename}.txt")
        
        if not os.path.exists(ann_file):
            print(f"警告: 未找到标注文件 {ann_file}")
            continue
            
        # 解析标注文件 (YOLO格式: class_id cx cy w h)
        with open(ann_file, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    cx = float(parts[1]) * width
                    cy = float(parts[2]) * height
                    w = float(parts[3]) * width
                    h = float(parts[4]) * height
                    
                    # 转换为COCO格式 [x,y,width,height]
                    x = cx - w / 2
                    y = cy - h / 2
                    
                    annotation = {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": class_id,  # 使用原始类别ID
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    }
                    
                    annotations.append(annotation)
                    ann_id += 1
    
    # 创建COCO真实标签
    coco_gt = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    # 保存到文件
    with open(output_file, 'w') as f:
        json.dump(coco_gt, f)
    
    print(f"已保存真实标签到 {output_file}")
    return coco_gt

# 绘制性能对比图表 - 这是之前缺失的函数
def plot_performance_comparison(fp32_times, int8_times, output_dir):
    """绘制FP32和INT8的性能对比图表"""
    plt.figure(figsize=(10, 6))
    
    # 检查是否有有效的测量结果
    if not fp32_times or not int8_times:
        print("没有足够的数据来绘制性能对比图")
        return
    
    # 计算平均值用于条形图
    avg_fp32 = np.mean(fp32_times)
    avg_int8 = np.mean(int8_times)
    
    # 条形图 - 平均推理时间对比
    plt.subplot(1, 2, 1)
    plt.bar(['FP32', 'INT8'], [avg_fp32, avg_int8], color=['blue', 'green'])
    plt.title('Average Inference Time')
    plt.ylabel('Time (ms)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 在每个条形上方添加具体数值
    plt.text(0, avg_fp32 + 0.5, f'{avg_fp32:.2f} ms', ha='center')
    plt.text(1, avg_int8 + 0.5, f'{avg_int8:.2f} ms', ha='center')
    
    # 箱型图 - 展示分布
    plt.subplot(1, 2, 2)
    plt.boxplot([fp32_times, int8_times], labels=['FP32', 'INT8'])
    plt.title('Inference Time Distribution')
    plt.ylabel('Time (ms)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存图表
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'))
    plt.close()
    
    print(f"性能对比图已保存到 {os.path.join(output_dir, 'performance_comparison.png')}")

# 计算精度下降
def calculate_accuracy_drop(fp32_metrics, int8_metrics):
    """
    计算FP32和INT8模型之间的精度下降
    
    参数:
        fp32_metrics: FP32模型指标字典
        int8_metrics: INT8模型指标字典
        
    返回:
        精度下降百分比字典
    """
    drops = {}
    
    for k in fp32_metrics:
        if fp32_metrics[k] > 0:
            drops[k] = (fp32_metrics[k] - int8_metrics[k]) / fp32_metrics[k] * 100
        else:
            drops[k] = 0
    
    return drops

# 绘制精度对比图
def plot_accuracy_comparison(fp32_metrics, int8_metrics, output_dir):
    """
    绘制FP32和INT8模型的精度对比图
    
    参数:
        fp32_metrics: FP32模型指标字典
        int8_metrics: INT8模型指标字典
        output_dir: 输出目录
    """
    plt.figure(figsize=(12, 6))
    
    # 选择要绘制的关键指标
    key_metrics = ['AP', 'AP50', 'AP75']
    
    # 获取选定指标的值
    fp32_values = [fp32_metrics[k] for k in key_metrics]
    int8_values = [int8_metrics[k] for k in key_metrics]
    
    # 准备分组条形图的数据
    x = np.arange(len(key_metrics))
    width = 0.35
    
    # 创建图表
    plt.bar(x - width/2, fp32_values, width, label='FP32', color='blue')
    plt.bar(x + width/2, int8_values, width, label='INT8', color='green')
    
    # 添加标签和图例
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Accuracy Comparison: FP32 vs INT8')
    plt.xticks(x, key_metrics)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加文本标注
    for i, metric in enumerate(key_metrics):
        plt.text(i - width/2, fp32_values[i] + 0.02, f'{fp32_values[i]:.3f}', ha='center')
        plt.text(i + width/2, int8_values[i] + 0.02, f'{int8_values[i]:.3f}', ha='center')
    
    # 保存图表
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()
    
    print(f"精度对比图已保存到 {os.path.join(output_dir, 'accuracy_comparison.png')}")

# 绘制综合对比图
def plot_combined_comparison(fp32_metrics, int8_metrics, fp32_time, int8_time, output_dir):
    """创建一个综合的性能和精度对比图"""
    plt.figure(figsize=(12, 8))
    
    # 选择主要指标
    metrics_to_plot = ['AP50', 'AP75', 'AP']
    fp32_scores = [fp32_metrics[m] for m in metrics_to_plot]
    int8_scores = [int8_metrics[m] for m in metrics_to_plot]
    
    # 第一张子图：精度对比
    plt.subplot(2, 1, 1)
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    plt.bar(x - width/2, fp32_scores, width, label='FP32', color='blue', alpha=0.7)
    plt.bar(x + width/2, int8_scores, width, label='INT8', color='green', alpha=0.7)
    plt.title('Accuracy and Performance Comparison')
    plt.ylabel('Accuracy Score')
    plt.xticks(x, metrics_to_plot)
    plt.ylim(0, 1.0)
    
    # 添加文本标注
    for i, metric in enumerate(metrics_to_plot):
        plt.text(i - width/2, fp32_scores[i] + 0.02, f'{fp32_scores[i]:.3f}', ha='center')
        plt.text(i + width/2, int8_scores[i] + 0.02, f'{int8_scores[i]:.3f}', ha='center')
    plt.legend()
    
    # 第二张子图：性能对比
    plt.subplot(2, 1, 2)
    speedup = fp32_time / int8_time if int8_time > 0 else 0
    plt.bar(['FP32 Time', 'INT8 Time', 'Speedup (x)'], 
            [fp32_time, int8_time, speedup], 
            color=['blue', 'green', 'orange'], alpha=0.7)
    plt.ylabel('Time (ms) / Speedup Factor')
    
    # 添加文本标注
    plt.text(0, fp32_time + 0.5, f'{fp32_time:.2f} ms', ha='center')
    plt.text(1, int8_time + 0.5, f'{int8_time:.2f} ms', ha='center')
    plt.text(2, speedup + 0.1, f'{speedup:.2f}x', ha='center')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_comparison.png'))
    plt.close()
    
    print(f"综合对比图已保存到 {os.path.join(output_dir, 'combined_comparison.png')}")

# 使用COCO指标评估模型精度
def evaluate_coco(gt_file, pred_file):
    """使用COCO指标评估模型精度"""
    try:
        # 初始化COCO真实标签
        coco_gt = COCO(gt_file)
        
        # 初始化COCO预测
        coco_dt = coco_gt.loadRes(pred_file)
        
        # 初始化COCOeval
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        
        # 运行评估
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # 提取指标
        metrics = {
            "AP": coco_eval.stats[0],  # AP@IoU=0.50:0.95
            "AP50": coco_eval.stats[1],  # AP@IoU=0.50
            "AP75": coco_eval.stats[2],  # AP@IoU=0.75
            "AP_small": coco_eval.stats[3],
            "AP_medium": coco_eval.stats[4],
            "AP_large": coco_eval.stats[5],
            "AR_max1": coco_eval.stats[6],
            "AR_max10": coco_eval.stats[7],
            "AR_max100": coco_eval.stats[8],
            "AR_small": coco_eval.stats[9],
            "AR_medium": coco_eval.stats[10],
            "AR_large": coco_eval.stats[11]
        }
        
        return metrics
    except Exception as e:
        print(f"评估过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 生成对比报告
def generate_report(fp32_metrics, int8_metrics, avg_fp32_time, avg_int8_time, 
                   fp32_model_path, int8_model_path, output_dir):
    """生成对比报告文件"""
    # 计算精度下降
    accuracy_drops = calculate_accuracy_drop(fp32_metrics, int8_metrics)
    
    # 计算速度提升
    speedup = avg_fp32_time / avg_int8_time if avg_int8_time > 0 else 0
    
    # 计算模型大小
    fp32_size = os.path.getsize(fp32_model_path.replace('.xml', '.bin')) / (1024 * 1024)
    int8_size = os.path.getsize(int8_model_path.replace('.xml', '.bin')) / (1024 * 1024)
    size_reduction = (fp32_size - int8_size) / fp32_size * 100
    
    # 创建报告内容
    report = []
    report.append("# YOLOv8 模型量化对比报告")
    report.append("=" * 50)
    report.append("")
    report.append(f"## 模型信息")
    report.append(f"- 原始FP32模型: {os.path.basename(fp32_model_path)}")
    report.append(f"- 量化INT8模型: {os.path.basename(int8_model_path)}")
    report.append("")
    
    report.append("## 性能对比")
    report.append(f"- FP32推理时间: {avg_fp32_time:.2f} ms")
    report.append(f"- INT8推理时间: {avg_int8_time:.2f} ms")
    report.append(f"- 速度提升: {speedup:.2f}x")
    report.append("")
    
    report.append("## 模型大小")
    report.append(f"- FP32模型: {fp32_size:.2f} MB")
    report.append(f"- INT8模型: {int8_size:.2f} MB")
    report.append(f"- 大小减少: {size_reduction:.2f}%")
    report.append("")
    
    report.append("## 精度对比")
    report.append("| 指标 | FP32 | INT8 | 精度下降 |")
    report.append("|------|------|------|----------|")
    
    for k in ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large']:
        if k in fp32_metrics and k in int8_metrics:
            report.append(f"| {k} | {fp32_metrics[k]:.4f} | {int8_metrics[k]:.4f} | {accuracy_drops[k]:.2f}% |")
    report.append("")
    
    # 添加综合评分
    if 'AP50' in accuracy_drops:
        # 综合评分 (简单权衡: 速度、大小和精度)
        accuracy_weight = 0.5
        speed_weight = 0.3
        size_weight = 0.2
        
        # 精度损失越小越好，限制在0-100%之间
        accuracy_score = max(0, 100 - accuracy_drops['AP50']) / 100
        # 速度提升越大越好，限制在1-5x之间的归一化得分
        speed_score = min(speedup, 5) / 5
        # 大小减少越大越好，限制在0-75%之间的归一化得分
        size_score = min(size_reduction, 75) / 75
        
        total_score = (accuracy_weight * accuracy_score + 
                      speed_weight * speed_score + 
                      size_weight * size_score) * 100
        
        report.append("## 量化综合评分")
        report.append(f"- 精度权重 (50%): {accuracy_score * 100:.2f}%")
        report.append(f"- 速度权重 (30%): {speed_score * 100:.2f}%")
        report.append(f"- 大小权重 (20%): {size_score * 100:.2f}%")
        report.append(f"- **总分**: {total_score:.2f}/100")
    
    # 保存报告
    report_path = os.path.join(output_dir, 'benchmark_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"对比报告已保存到 {report_path}")
    return report_path

# 比较性能函数
def compare_performance(fp32_model_path, int8_model_path, test_images, output_dir, num_runs=10):
    fp32_times = []
    int8_times = []
    
    print("进行性能对比测试...")
    
    # 检查INT8模型是否存在
    if not os.path.exists(int8_model_path) or not os.path.exists(int8_model_path.replace('.xml', '.bin')):
        print(f"警告: INT8模型文件不存在 - {int8_model_path}")
        return 0, 0, 0
    
    # 检查测试图像列表
    if not test_images:
        print("警告: 没有测试图像")
        return 0, 0, 0
    
    for img_path in tqdm(test_images[:10], desc="测量推理性能"):  # 限制为前10张以加快测试速度
        # 检查图像文件是否存在
        if not os.path.exists(img_path):
            print(f"警告: 测试图像不存在 - {img_path}")
            continue
            
        # 多次运行以获得稳定的结果
        fp32_batch = []
        int8_batch = []
        
        try:
            for _ in range(num_runs):
                # FP32模型推理
                fp32_time, *_ = inference(fp32_model_path, img_path)
                fp32_batch.append(fp32_time)
                
                # INT8模型推理
                int8_time, *_ = inference(int8_model_path, img_path)
                int8_batch.append(int8_time)
            
            # 取平均值
            fp32_times.append(np.mean(fp32_batch))
            int8_times.append(np.mean(int8_batch))
        except Exception as e:
            print(f"推理过程发生错误 ({img_path}): {str(e)}")
    
    # 计算平均推理时间
    avg_fp32_time = np.mean(fp32_times) if fp32_times else 0
    avg_int8_time = np.mean(int8_times) if int8_times else 0
    speedup = avg_fp32_time / avg_int8_time if avg_int8_time > 0 else 0
    
    print(f"FP32 平均推理时间: {avg_fp32_time:.2f} ms")
    print(f"INT8 平均推理时间: {avg_int8_time:.2f} ms")
    print(f"加速比: {speedup:.2f}x")
    
    # 绘制性能对比图
    plot_performance_comparison(fp32_times, int8_times, output_dir)
    
    return avg_fp32_time, avg_int8_time, speedup

# 比较精度函数
def compare_accuracy(fp32_model_path, int8_model_path, test_images, labels_dir, categories, output_dir):
    """比较FP32和INT8模型的精度"""
    print("进行精度对比测试...")
    
    # 在输出目录中创建临时文件
    temp_dir = os.path.join(output_dir, 'temp_accuracy_eval')
    os.makedirs(temp_dir, exist_ok=True)
    
    # 设置一致的置信度和IOU阈值
    conf_threshold = 0.53
    iou_threshold = 0.5
    
    try:
        # 创建真实标签文件
        gt_file = os.path.join(temp_dir, 'ground_truth.json')
        create_coco_ground_truth(test_images, labels_dir, gt_file, categories)
        
        # 为两个模型创建预测文件
        fp32_pred_file = os.path.join(temp_dir, 'fp32_predictions.json')
        int8_pred_file = os.path.join(temp_dir, 'int8_predictions.json')
        
        # 运行推理并创建预测文件
        create_coco_predictions(test_images, fp32_model_path, fp32_pred_file, 
                               conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        create_coco_predictions(test_images, int8_model_path, int8_pred_file,
                               conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        
        # 评估模型
        print("评估FP32模型精度...")
        fp32_metrics = evaluate_coco(gt_file, fp32_pred_file)
        
        if fp32_metrics is None:
            print("FP32模型评估失败")
            return None, None, None
        
        print("评估INT8模型精度...")
        int8_metrics = evaluate_coco(gt_file, int8_pred_file)
        
        if int8_metrics is None:
            print("INT8模型评估失败")
            return fp32_metrics, None, None
        
        # 计算精度下降
        accuracy_drops = calculate_accuracy_drop(fp32_metrics, int8_metrics)
        
        # 打印指标
        print("\nFP32模型精度:")
        for k, v in fp32_metrics.items():
            print(f"{k}: {v:.4f}")
            
        print("\nINT8模型精度:")
        for k, v in int8_metrics.items():
            print(f"{k}: {v:.4f}")
            
        print("\n精度下降:")
        for k, v in accuracy_drops.items():
            print(f"{k}: {v:.2f}%")
        
        # 绘制精度对比图
        plot_accuracy_comparison(fp32_metrics, int8_metrics, output_dir)
        
        return fp32_metrics, int8_metrics, accuracy_drops
    except Exception as e:
        print(f"精度比较过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

# 主函数
def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='YOLOv8模型性能与精度对比工具')
    
    parser.add_argument('--fp32', default=DEFAULT_FP32_MODEL_PATH, 
                        help='FP32模型路径 (.xml)')
    parser.add_argument('--int8', default=DEFAULT_INT8_MODEL_PATH,
                        help='INT8模型路径 (.xml)')
    parser.add_argument('--images', default=DEFAULT_IMAGES_DIR,
                        help='测试图像目录')
    parser.add_argument('--labels', default=DEFAULT_LABELS_DIR,
                        help='测试标签目录')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR,
                        help='输出目录路径')
    parser.add_argument('--device', default='CPU',
                        help='推理设备 (CPU, GPU) (默认: CPU)')
    parser.add_argument('--num_runs', type=int, default=10,
                        help='性能测试重复运行次数 (默认: 10)')
    
    args = parser.parse_args()
    
    # 确保使用固定的阈值
    conf_threshold = 0.53  # 与sub_openvino_blue.py一致
    iou_threshold = 0.5    # 与sub_openvino_blue.py一致
    
    # 打印参数
    print("=" * 50)
    print("YOLOv8模型性能与精度对比")
    print("=" * 50)
    print(f"FP32模型: {args.fp32}")
    print(f"INT8模型: {args.int8}")
    print(f"图像目录: {args.images}")
    print(f"标签目录: {args.labels}")
    print(f"输出目录: {args.output_dir}")
    print(f"置信度阈值: {conf_threshold}")
    print(f"IOU阈值: {iou_threshold}")
    print(f"设备: {args.device}")
    print("=" * 50)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查模型文件是否存在
    if not os.path.exists(args.fp32) or not os.path.exists(args.int8):
        print(f"错误: 模型文件不存在。请确保FP32和INT8模型都可用。")
        return
    
    # 获取测试图像
    test_images = []
    for ext in ['.jpg', '.jpeg', '.png']:
        test_images.extend([str(p) for p in Path(args.images).rglob(f'*{ext}')])
    
    if not test_images:
        print(f"错误: 在图像目录 {args.images} 中未找到图像")
        return
    
    print(f"找到 {len(test_images)} 张测试图像")
    
    # 检查标签目录
    if not os.path.exists(args.labels):
        print(f"错误: 标签目录 {args.labels} 不存在")
        return
    
    # 比较性能
    avg_fp32_time, avg_int8_time, speedup = compare_performance(
        args.fp32,
        args.int8,
        test_images,
        args.output_dir,
        num_runs=args.num_runs
    )
    
    # 比较精度
    fp32_metrics, int8_metrics, accuracy_drops = compare_accuracy(
        args.fp32,
        args.int8,
        test_images,
        args.labels,
        CATEGORIES,
        args.output_dir
    )
    
    # 绘制综合对比图
    if fp32_metrics and int8_metrics:
        plot_combined_comparison(
            fp32_metrics, 
            int8_metrics, 
            avg_fp32_time, 
            avg_int8_time,
            args.output_dir
        )
        
        # 生成报告
        report_path = generate_report(
            fp32_metrics, 
            int8_metrics, 
            avg_fp32_time, 
            avg_int8_time,
            args.fp32,
            args.int8,
            args.output_dir
        )
        
        print("\n对比总结:")
        print(f"- 精度: FP32 AP50={fp32_metrics['AP50']:.4f}, INT8 AP50={int8_metrics['AP50']:.4f}")
        print(f"- 精度下降: AP50下降了 {accuracy_drops['AP50']:.2f}%")
        print(f"- 速度提升: {speedup:.2f}x")
        
        # 给出量化是否值得的建议
        if accuracy_drops['AP50'] < 1.0:
            print("\n✅ 建议: 精度损失很小 (< 1%), 量化非常值得。")
        elif accuracy_drops['AP50'] < 5.0:
            print("\n✅ 建议: 精度损失可接受 (< 5%), 如果速度至关重要，建议使用量化模型。")
        elif speedup > 3.0:
            print("\n⚠️ 建议: 虽然精度损失较大，但速度提升显著 (> 3x)，可能值得在速度敏感场景使用。")
        else:
            print("\n❌ 建议: 精度损失较大而速度提升有限，建议使用原始FP32模型。")
    else:
        print("由于精度评估失败，无法生成完整报告")

if __name__ == "__main__":
    main()
