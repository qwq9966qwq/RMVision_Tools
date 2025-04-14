from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # USB摄像头节点
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            output='screen',
            parameters=[
                {
                    'video_device': '/dev/video1',
                    'image_width': 1280,
                    'image_height': 1024,
                    'pixel_format': 'yuyv',
                    'color_format': 'yuv422p',
                    'camera_frame_id': 'usb_cam',
                    'io_method': 'mmap'
                }
            ]
        ),
        
        # RQT图像查看器
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='rqt_image_view',
            output='screen',
            arguments=['--force-discover']
        )
    ])
