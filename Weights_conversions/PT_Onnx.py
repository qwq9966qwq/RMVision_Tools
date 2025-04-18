#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_onnx(model_path, output_path=None, img_size=640, batch_size=1, simplify=True, opset=12):
    """将YOLOv8 PT模型转换为ONNX格式"""
    try:
        from ultralytics import YOLO
        
        logger.info(f"加载模型: {model_path}")
        model = YOLO(model_path)
        
        if output_path is None:
            output_path = Path(model_path).with_suffix('.onnx')
        
        logger.info(f"将模型导出为ONNX格式: {output_path}")
        
        # 使用Ultralytics的导出功能
        success = model.export(format="onnx", imgsz=img_size, batch=batch_size, 
                     simplify=simplify, opset=opset, verbose=False)
        
        if success:
            logger.info(f"成功将模型转换为ONNX格式，保存在: {output_path}")
            return output_path
        else:
            logger.error("ONNX转换失败")
            return None
    
    except Exception as e:
        logger.error(f"转换至ONNX时出错: {e}")
        return None

def main():
    # 预设参数 - 可根据需要手动修改
    model_path = "/home/guo/Weights_conversions/Weights/best_blue.pt"  # 输入模型路径
    output_path = None         # 输出路径，None表示自动生成
    img_size = 640            # 图像尺寸
    batch_size = 1            # 批处理大小
    simplify = True           # 是否简化模型
    opset = 12                # ONNX操作集版本
    
    # 调用转换函数
    convert_to_onnx(
        model_path=model_path,
        output_path=output_path,
        img_size=img_size,
        batch_size=batch_size,
        simplify=simplify,
        opset=opset
    )

if __name__ == "__main__":
    main()
