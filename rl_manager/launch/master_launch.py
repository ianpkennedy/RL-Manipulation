import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare





def generate_launch_description():
    
    robot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ur_simulation_gazebo"), "/launch", "/ur_sim_moveit.launch.py"]
        ),
        launch_arguments={}.items(),
    )
    # Initialize nodes
    
    # Spawn static camera in Gazebo

    camera_sdf_path = os.path.join(
        get_package_share_directory('rl_manager'),
        "description",
        "camera.sdf"
    )
    
    
    spawn_camera = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity", "static_camera",
            "-file", camera_sdf_path,
            "-x", "0", "-y", "1.0", "-z", "2",
            "-R", "0", "-P", "2.0", "-Y", "0"
        ],
        output="screen"
    )

    agent = Node(package='py_pubsub',
                 executable='talker',output='screen',)
    
    manager = Node(
        package='rl_manager',
        executable='rl_manager_node',
        name='rl_manager_node',
        output='screen',
    )

    return LaunchDescription([agent, spawn_camera, robot_launch, manager])