#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import NavSatFix

def calculate_bearing(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)

    x = math.sin(delta_lon) * math.cos(phi2)
    y = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lon))

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360.0) % 360.0

class GPSToCoursePublisher(Node):
    def __init__(self):
        super().__init__('gps_course_publisher')
        self.course_pub = self.create_publisher(
            Float32,
            'ublox_gps/course_deg',
            10
        )
        self.fix_sub = self.create_subscription(
            NavSatFix,
            '/ublox_gps_node/fix',  
            self.fix_callback,
            10
        )

        self.prev_lat = None
        self.prev_lon = None

        self.get_logger().info('GPS Course Publisher node started, waiting for NavSatFix...')

    def fix_callback(self, msg: NavSatFix):
        lat = msg.latitude
        lon = msg.longitude

        if self.prev_lat is not None and self.prev_lon is not None:
            course = calculate_bearing(self.prev_lat, self.prev_lon, lat, lon)
            course_msg = Float32()
            course_msg.data = course
            self.course_pub.publish(course_msg)
            self.get_logger().info(f'Published course: {course:.2f}°')

        self.prev_lat = lat
        self.prev_lon = lon

def main(args=None):
    rclpy.init(args=args)
    node = GPSToCoursePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import math
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Float32
# from sensor_msgs.msg import NavSatFix
# from collections import deque # [추가] deque 라이브러리 임포트
# import numpy as np # [추가] numpy 라이브러리 임포트

# def calculate_bearing(lat1, lon1, lat2, lon2):
#     # ... (이 함수는 변경 없음) ...
#     phi1 = math.radians(lat1)
#     phi2 = math.radians(lat2)
#     delta_lon = math.radians(lon2 - lon1)
#     x = math.sin(delta_lon) * math.cos(phi2)
#     y = (math.cos(phi1) * math.sin(phi2)
#          - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lon))
#     bearing = math.degrees(math.atan2(x, y))
#     return (bearing + 360.0) % 360.0

# class GPSToCoursePublisher(Node):
#     def __init__(self):
#         super().__init__('gps_course_publisher')
#         self.course_pub = self.create_publisher(
#             Float32,
#             'ublox_gps/course_deg',
#             10
#         )
#         self.fix_sub = self.create_subscription(
#             NavSatFix,
#             '/ublox_gps_node/fix',
#             self.fix_callback,
#             10
#         )

#         self.prev_lat = None
#         self.prev_lon = None

#         # --- [추가] 이동 평균 필터(MAF)를 위한 변수 ---
#         self.window_size = 7  # 최근 5개의 데이터를 사용
#         self.course_history = deque(maxlen=self.window_size)
#         self.get_logger().info(f'Moving Average Filter window size set to: {self.window_size}')
#         # --- 추가 끝 ---

#         self.get_logger().info('GPS Course Publisher node started, waiting for NavSatFix...')

#     def fix_callback(self, msg: NavSatFix):
#         lat = msg.latitude
#         lon = msg.longitude

#         if self.prev_lat is not None and self.prev_lon is not None:
#             # 1. 원본(raw) course 계산
#             raw_course = calculate_bearing(self.prev_lat, self.prev_lon, lat, lon)
            
#             # 2. [수정] 계산된 course를 history에 추가
#             self.course_history.append(raw_course)

#             # 3. [수정] history가 꽉 찼을 때만 평균 계산 및 발행
#             if len(self.course_history) == self.window_size:
#                 # --- 각도 평균의 함정을 피하기 위한 벡터 계산 ---
#                 sum_x, sum_y = 0.0, 0.0
#                 for angle_deg in self.course_history:
#                     angle_rad = math.radians(angle_deg)
#                     sum_x += math.cos(angle_rad)
#                     sum_y += math.sin(angle_rad)
                
#                 avg_x = sum_x / self.window_size
#                 avg_y = sum_y / self.window_size
                
#                 averaged_course_rad = math.atan2(avg_y, avg_x)
#                 averaged_course_deg = (math.degrees(averaged_course_rad) + 360.0) % 360.0
#                 # --- 벡터 계산 끝 ---

#                 course_msg = Float32()
#                 course_msg.data = averaged_course_deg
#                 self.course_pub.publish(course_msg)
#                 self.get_logger().info(f'Raw: {raw_course:.2f}°, Smoothed: {averaged_course_deg:.2f}°')

#         self.prev_lat = lat
#         self.prev_lon = lon

# def main(args=None):
#     # ... (이 함수는 변경 없음) ...
#     rclpy.init(args=args)
#     node = GPSToCoursePublisher()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()