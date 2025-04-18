#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import subprocess
import logging
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_openvino(onnx_path, output_dir=None):
    """将ONNX模型转换为OpenVINO格式"""
    try:
        if output_dir is None:
            output_dir = Path(onnx_path).parent
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        output_model = Path(output_dir) / Path(onnx_path).stem
        
        logger.info(f"将ONNX模型转换为OpenVINO格式: {output_model}")
        
        # 使用OpenVINO命令行工具
        cmd = [
            "mo",
            "--input_model", str(onnx_path),
            "--output_dir", str(output_dir),
            "--model_name", Path(onnx_path).stem
        ]
        
        logger.info(f"执行命令: {' '.join(cmd)}")
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if process.returncode == 0:
            logger.info(f"成功将模型转换为OpenVINO格式，保存在: {output_model}.xml 和 {output_model}.bin")
            return f"{output_model}.xml"
        else:
            logger.error(f"OpenVINO转换失败: {process.stderr}")
            return None
    
    except Exception as e:
        logger.error(f"转换至OpenVINO时出错: {e}")
        return None

def main():
    # 预设参数 - 可根据需要手动修改
    onnx_path = "/home/guo/Weights_conversions/Weights/best_red.onnx"  # 输入ONNX模型路径
    output_dir = None         # 输出目录，None表示与输入模型相同
    
    convert_to_openvino(onnx_path, output_dir)

if __name__ == "__main__":
    main()
