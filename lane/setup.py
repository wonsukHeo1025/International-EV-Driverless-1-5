from setuptools import setup

package_name = 'lane'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
        	'launch/yolo_lane_launch.py',
        	'launch/traditional_lane_launch.py']),
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='KTY',
    maintainer_email='kty@example.com',
    description='YOLOPv2 ROS 2 integration package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_lane_node = lane.lane_yolo:main',
            'traditional_lane_node = lane.lane_traditional:main',
            'custom_lane_node = lane.lane_custom:main',
            'path_node = lane.path:main',
            'highcontrol_node = lane.highcontrol:main',
        ],
    },
)

