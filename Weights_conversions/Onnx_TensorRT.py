#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import logging
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_to_tensorrt(
    onnx_path,
    output_path=None,
    fp16=True,
    int8=False,
    workspace=4,  # 单位GB
    max_batch_size=1
):
    try:
        import tensorrt as trt
        import pycuda.autoinit
        import pycuda.driver as cuda
        
        # 设置输出路径
        if output_path is None:
            output_path = Path(onnx_path).with_suffix('.engine')
        os.makedirs(Path(output_path).parent, exist_ok=True)
        
        logger.info(f"开始转换: {onnx_path} -> {output_path}")
        
        # 1. 初始化TensorRT
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # 2. 解析ONNX模型
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                logger.error("ONNX解析失败:")
                for error in range(parser.num_errors):
                    logger.error(f"  Error {error}: {parser.get_error(error)}")
                return None
        
        # 3. 配置构建选项
        config = builder.create_builder_config()
        
        # 关键修改点：设置工作空间内存（新API）
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (1 << 30))  # 转换为字节
        
        # 设置精度标志
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("启用FP16模式")
        
        if int8 and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            logger.warning("INT8模式需要设置校准器，当前示例未实现此功能")
        
        # 4. 构建引擎
        logger.info("开始构建TensorRT引擎...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            logger.error("引擎构建失败")
            return None
            
        # 5. 保存引擎
        with open(output_path, 'wb') as f:
            f.write(serialized_engine)
        
        logger.info(f"转换成功! 保存至: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"转换失败: {str(e)}", exc_info=True)
        return None

def main():
    # 配置参数
    onnx_path = "/home/guo/Weights_conversions/Weights/best_red.onnx"  # 替换为你的ONNX路径
    output_path = None  # 自动生成.engine文件
    fp16 = True        # 启用FP16加速
    int8 = False       # INT8需要校准数据
    
    # 执行转换
    result = convert_to_tensorrt(
        onnx_path=onnx_path,
        output_path=output_path,
        fp16=fp16,
        int8=int8,
        workspace=4,     # 工作空间4GB
        max_batch_size=1 # 最大批处理大小
    )
    
    if not result:
        logger.error("转换失败，请检查错误日志")
        exit(1)

if __name__ == "__main__":
    main()
