# 视觉组的工具箱 (Vision Team Toolbox)

日常会用到的视觉处理工具组件集合。

## 目录

- [Camera_BiaoDing(相机标定)](#相机标定程序)
- [Calibration(模型量化压缩)](#模型量化工具)
- [Lidar_BiaoDing(雷达相机联合标定)](#雷达相机联合标定)

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
