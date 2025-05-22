# 视觉组的工具箱 (Vision Team Toolbox)

日常会用到的视觉处理工具组件集合。

## 目录

- [Camera_BiaoDing(相机标定)](#相机标定程序)
- [Calibration(模型量化压缩)](#模型量化工具)
- [Lidar_BiaoDing(雷达相机联合标定)](#雷达相机联合标定)
- [Weights_conversions(权重转化)](#权重转化程序)
- [Code_Analysis(代码文件结构分析——新增)](#代码文件结构分析)

## 代码文件结构分析

用于分析文件夹内的代码结构，生成对应的树状结构字符描述，清晰明朗

### 配置file_analyzer.conf

可配置分析文件夹的路径，支持text, json, xml, html以及终端直接输出（彩色）

示例：
![image](https://github.com/user-attachments/assets/c991632d-8ecd-4ac3-b07e-7a60dde6a760)

  
## 相机标定程序

用于相机内参标定，包含标定程序与标定后的海康与迈德威视相机内参。

### 资源链接

- **开源地址**：[GitHub - camera_calibration_tool](https://github.com/chenyr0021/camera_calibration_tool)
- **教程**：[CSDN - 相机标定详解](https://blog.csdn.net/qq_31417941/article/details/102952942)

### 使用方法

1. 按照教程准备棋盘格标定板
2. 使用相机在不同角度和高度拍摄标定板
3. 运行以下命令进行标定：

```bash
python3 calibration.py --image_size 1920x1080 --mode calibrate --corner 8x6 --square 20
```

参数说明：
- `--image_size`：图像分辨率
- `--mode`：运行模式
- `--corner`：棋盘格内角点数量（宽x高）
- `--square`：棋盘格方格尺寸（mm）

## 模型量化工具

针对Openvino部署，提供模型权重压缩量化程序，包含FP32与INT8权重及性能对比。

### 包含内容

- **Calibration**：包含量化前FP32与量化后INT8的权重，以及性能对比
- **Calibration_datasets**：量化测试数据集
- **Calibration_Code**：量化程序

### 使用方法

1. 准备FP32权重
2. 修改程序中的数据集与权重存放路径
3. 运行`Calibration.py`进行量化压缩
4. 运行`Compare.py`进行精度与性能对比

效果展示：
![微信图片_20250414202213](https://github.com/user-attachments/assets/cd61346c-29cb-4394-b4de-1311e7c40e96)
![微信图片_20250414202225](https://github.com/user-attachments/assets/9772ab87-e4ac-4c20-806b-94839a954dbe)


## 雷达相机联合标定

支持ROS2的雷达与相机自动联合标定工具。

### 资源链接

- **开源地址**：[GitHub - lidar_camera_calib](https://github.com/simi-asher/lidar_camera_calib)
- **标定教程**：
  - [CSDN - 标定教程1](https://blog.csdn.net/qq_37223654/article/details/144568843)
  - [CSDN - 标定教程2](https://blog.csdn.net/A6666686678/article/details/138096601)

### 使用流程

1. 按照教程顺序步骤运行
2. 在雷达进行bag包记录的同时，使用标定过的相机进行拍照
3. 按照教程运行标定程序完成联合标定

## 权重转化程序
主要包含PT权重转化为Onnx权重以及Onnx权重转化为openvino(bin和xml)权重以及TensorRT权重(engine)

### 包含内容

- **PT_conversions**：一键转化，包括所有转化，输入PT路径，输出onnx和openvino以及TensorRT的权重
- **Onnx_TensorRT**：Onxx权重转换为TensorRT权重程序
- **PT_Onnx**：PT转Onnx权重程序
- **PT_Openvino**：PT转Openvino权重程序，包含网格xml权重以及参数权重bin

### 推理引擎介绍
- **PyTorch**：PT原生的推理引擎，训练模型输出的就是这种权重
- **Onnxruntime**：跨平台通用推理引擎，转到Linux系统下搭建推理程序的时候的首选，操作简便，YOLO官方自带转化接口函数
- **Openvino**：Intel的推理引擎，为Intel专门推出的，可以使用Intel的CPU以及集成GPU进行推理，常规NUC或者迷你主机的首选，支持INT8量化(因为不具备NVIDA的GPU)
- **TensorRT**：NVIDA推出的推理引擎，性能最为强劲，但是受限成本无法实际部署，可以在Jeston nano上进行部署，支持FP16精度权重

### 效果展示
TensorRT转换：
![8fc381b50385cd2e33315f78a4b5200](https://github.com/user-attachments/assets/d05bbbc5-40d6-4d07-a002-41b3b5be0b38)

