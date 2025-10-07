#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class RealSenseCameraSubscriber(Node):
    """
    ROS2 node to subscribe to RealSense D405 camera topic
    """
    
    def __init__(self):
        super().__init__('realsense_camera_subscriber')
        
        # Initialize CV Bridge for converting ROS images to OpenCV format
        self.bridge = CvBridge()
        
        # Declare parameters
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('show_image', True)
        
        # Get parameter values
        camera_topic = self.get_parameter('camera_topic').value
        self.show_image = self.get_parameter('show_image').value
        
        # Create subscriber for camera images
        self.subscription = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )
        
        self.get_logger().info(f'Subscribed to camera topic: {camera_topic}')
        
        # Frame counter
        self.frame_count = 0
    
    def image_callback(self, msg):
        """
        Callback function that runs for every incoming frame
        """
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            self.frame_count += 1
            
            # Log every 30 frames to avoid spam
            if self.frame_count % 30 == 0:
                self.get_logger().info(
                    f'Received frame {self.frame_count} - '
                    f'Size: {cv_image.shape[1]}x{cv_image.shape[0]}'
                )
            
            # Display the image if enabled
            if self.show_image:
                cv2.imshow('RealSense D405 Camera', cv_image)
                cv2.waitKey(1)
            
            # Add your custom image processing here
            # Example: self.process_image(cv_image, msg.header)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
    
    def process_image(self, cv_image, header):
        """
        Custom processing for camera images
        Add your image processing logic here
        
        Args:
            cv_image: OpenCV image (numpy array)
            header: ROS message header with timestamp and frame_id
        """
        # Example: Access timestamp
        timestamp = header.stamp.sec + header.stamp.nanosec * 1e-9
        
        # Example: Simple image analysis
        height, width = cv_image.shape[:2]
        
        # Add your processing here
        # - Object detection
        # - Feature extraction
        # - Color segmentation
        # - etc.
        
        pass


def main(args=None):
    rclpy.init(args=args)
    
    camera_subscriber = RealSenseCameraSubscriber()
    
    try:
        rclpy.spin(camera_subscriber)
    except KeyboardInterrupt:
        camera_subscriber.get_logger().info("Camera subscriber stopped by user.")
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        camera_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()