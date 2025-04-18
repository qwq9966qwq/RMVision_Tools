#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_requirements():
    """检查所需的库是否已安装"""
    try:
        import torch
        import ultralytics
        import onnx
    except ImportError as e:
        logger.error(f"缺少必要的库: {e}")
        logger.info("请安装所需的库: pip install ultralytics onnx")
        sys.exit(1)
    
    logger.info("基本依赖检查通过")

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
            return Path(model_path).with_suffix('.onnx')
        else:
            logger.error("ONNX转换失败")
            return None
    
    except Exception as e:
        logger.error(f"转换至ONNX时出错: {e}")
        return None

def convert_to_openvino(onnx_path, output_dir=None):
    """将ONNX模型转换为OpenVINO格式"""
    try:
        import subprocess
        
        if output_dir is None:
            output_dir = Path(onnx_path).parent
        
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

def convert_to_tensorrt(onnx_path, output_path=None, fp16=True, int8=False, workspace=4):
    """将ONNX模型转换为TensorRT引擎"""
    try:
        import tensorrt as trt
        import cuda
        import pycuda.autoinit
        
        if output_path is None:
            output_path = Path(onnx_path).with_suffix('.engine')
        
        logger.info(f"将ONNX模型转换为TensorRT引擎: {output_path}")
        
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(f"TensorRT解析ONNX错误 {error}: {parser.get_error(error)}")
                return None
        
        config = builder.create_builder_config()
        config.max_workspace_size = workspace * 1 << 30  # 设置工作空间大小（以GB为单位）
        
        if fp16 and builder.platform_has_fast_fp16:
            logger.info("启用FP16精度")
            config.set_flag(trt.BuilderFlag.FP16)
        
        if int8 and builder.platform_has_fast_int8:
            logger.info("启用INT8精度")
            config.set_flag(trt.BuilderFlag.INT8)
            # 这里需要设置Int8校准器，这部分比较复杂，本示例代码中省略
        
        logger.info("开始构建TensorRT引擎，这可能需要一些时间...")
        engine = builder.build_engine(network, config)
        
        if engine:
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            logger.info(f"成功将模型转换为TensorRT引擎，保存在: {output_path}")
            return output_path
        else:
            logger.error("TensorRT引擎构建失败")
            return None
    
    except ImportError:
        logger.error("未找到TensorRT或CUDA相关库。请确保已安装TensorRT、CUDA和PyCUDA")
        return None
    except Exception as e:
        logger.error(f"转换至TensorRT时出错: {e}")
        return None

def main():
    # 预设参数 - 可根据需要手动修改
    model_path = "yolov8n.pt"  # 输入模型路径
    output_dir = None         # 输出目录，None表示与输入模型相同
    img_size = 640           # 图像尺寸
    batch_size = 1           # 批处理大小
    simplify = True          # 是否简化ONNX模型
    opset = 12               # ONNX操作集版本
    fp16 = True              # 启用TensorRT的FP16精度
    int8 = False             # 启用TensorRT的INT8精度(需要校准数据)
    workspace = 4            # TensorRT工作空间大小(GB)
    
    # 选择要执行的转换 (True表示执行)
    to_onnx = True           # 转换为ONNX格式
    to_openvino = True       # 转换为OpenVINO格式
    to_tensorrt = True       # 转换为TensorRT引擎格式
    
    # 检查必要的库
    check_requirements()
    
    # 设置输出目录
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = Path(model_path).parent
    
    onnx_path = None
    
    # 转换为ONNX（这是其他转换的基础）
    if to_onnx or to_openvino or to_tensorrt:
        onnx_path = convert_to_onnx(
            model_path, 
            Path(output_dir) / Path(model_path).with_suffix('.onnx').name,
            img_size,
            batch_size,
            simplify,
            opset
        )
        
        if onnx_path is None:
            logger.error("无法创建ONNX模型，终止后续转换")
            return
    
    # 转换为OpenVINO
    if to_openvino:
        openvino_path = convert_to_openvino(onnx_path, output_dir)
        if openvino_path is None:
            logger.warning("OpenVINO转换失败")
    
    # 转换为TensorRT
    if to_tensorrt:
        tensorrt_path = convert_to_tensorrt(
            onnx_path, 
            Path(output_dir) / Path(model_path).with_suffix('.engine').name,
            fp16,
            int8,
            workspace
        )
        if tensorrt_path is None:
            logger.warning("TensorRT转换失败")
    
    logger.info("转换过程完成")

if __name__ == "__main__":
    main()
