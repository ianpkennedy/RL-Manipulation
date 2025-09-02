# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.srv import DeleteEntity
from sensor_msgs.msg import JointState

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Imu
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import MagneticField
from sensor_msgs.msg import Joy
from sensor_msgs.msg import MultiDOFJointState
from sensor_msgs.msg import BatteryState
from sensor_msgs.msg import RegionOfInterest      



from time import sleep

import cv2




from py_pubsub.my_lib import custom_function  # Importing from the custom library
from py_pubsub.my_lib import SimpleCNN
from py_pubsub.my_lib import SimpleDNN 
from py_pubsub.my_lib import YOLODetector
import numpy as np
import torch

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 7.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.timer2 = self.create_timer(4.0, self.timer_callback2)  # Additional timer for testing
        
        self.i = 2.0
        self.gazebo_spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_entity_client = self.create_client(DeleteEntity, '/delete_entity')
        print('starting MinimalPublisher node')
        self.delete = False
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        self.camera1 = self.create_subscription(
            Image,
            '/static_camera1/static_camera1/image_raw',
            self.camera1_callback,
            10
        )
        
        self.camera2 = self.create_subscription(
            Image,
            '/static_camera2/static_camera2/image_raw',
            self.camera2_callback,
            10
        )
        
        self.camera1_data = Image()
        self.camera2_data = Image()
        
        self.joint_angles = JointState()

        print(custom_function())  # Call the custom function from the library

        self.delete = True
        
        self.yolo_model = YOLODetector()
        print('YOLO model loaded successfully.')

    def timer_callback2(self):

        try:
            self.delete_box()
        except Exception as e:
            self.get_logger().error(f'Error deleting red box: {e}')

    
    def camera1_callback(self, msg):
        # Process the camera1 image message
        self.get_logger().info(f'Received camera1 image with width {msg.width} and height {msg.height}')
        # You can add more processing logic here if needed
        self.camera1_data = msg  # Store the received image data

        result = self.yolo_model.detect(msg)  # Perform detection using YOLO model

        print('yolo processing done')
        print(result)

    def camera2_callback(self, msg):
        # Process the camera2 image message
        self.get_logger().info(f'Received camera2 image with width {msg.width} and height {msg.height}')
        # You can add more processing logic here if needed
        self.camera2_data = msg  # Store the received image data

    def joint_state_callback(self, msg):
        # Process the joint state message
        self.get_logger().info(f'Received joint states: {msg.name} with positions {msg.position}')
        # You can add more processing logic here if needed
        self.joint_angles = msg  # Store the received joint states
        
    def spawn_red_box(self):
          
          
        red_box_sdf = '''<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="red_box">
    <static>true</static>
    <link name="link">
      <pose>0 0 0.5 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <box>
            <size>0.1 0.1 0.1</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>0.1 0.1 0.1</size>
          </box>
        </geometry>
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
'''
        self.delete = True
        print('configuring red box SDF')
        req = SpawnEntity.Request()
        req.name = 'red_box'
        req.xml = red_box_sdf
        req.robot_namespace = ''
        req.reference_frame = 'world'
        req.initial_pose.position.x = self.i
        self.i += 0.5  # Increment x position for each spawn
        req.initial_pose.position.y = 0.0
        req.initial_pose.position.z = 1.0
        req.initial_pose.orientation.x = 0.0
        req.initial_pose.orientation.y = 0.0
        req.initial_pose.orientation.z = 0.0
        req.initial_pose.orientation.w = 1.0
        future = self.gazebo_spawn_client.call_async(req)



        if future.result() is not None:
            self.get_logger().info('Red box spawned successfully!')
        else:
            self.get_logger().error('Failed to spawn red box.')

        
        
    def delete_box(self):
      
        self.delete = False    
        print('deleting red box')
        req = DeleteEntity.Request()
        req.name = 'red_box'
        future = self.delete_entity_client.call_async(req)
        
        if future.result() is not None:
            self.get_logger().info('Red box deleted successfully!')
        else:
            self.get_logger().error('Failed to delete red box.')


    def timer_callback(self):
        # if self.gazebo_spawn_client.service_is_ready():
        
        # if self.delete==False:
        #     self.spawn_red_box()
        
        custom_function()  # Call the custom function from the library
        try:
            self.spawn_red_box()
        except Exception as e:
            self.get_logger().error(f'Error spawning red box: {e}')
            
        # if self.delete==True:
        #     pass
            # self.delete_box()
            # self.timer.cancel()  # Only spawn once


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
