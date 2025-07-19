#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, UInt8
from sensor_msgs.msg import NavSatFix, PointCloud
from geometry_msgs.msg import Point32, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
import pandas as pd
import numpy as np
import math
import os
import cv2
import threading
import time

class MissionControlNode(Node):
    def __init__(self):
        super().__init__('mission_control_node')
        self.get_logger().info('Mission Control Node (GPS Checkpoint Mode) has been started.')
        
        # shutdown 플래그 추가
        self.is_shutdown = False

        self.declare_parameter('waypoints_path', '역.csv')#gnss
        self.declare_parameter('look_ahead_distance', 2.0)
        self.declare_parameter('wheel_base', 0.7)
        
        # GPS 체크포인트 설정
        self.declare_parameter('checkpoint_lat', 33.305500)       # 체크포인트 위도
        self.declare_parameter('checkpoint_lon', 126.314400)      # 체크포인트 경도
        self.declare_parameter('checkpoint_radius', 0.5)          # 체크포인트 반경 (미터)
        self.declare_parameter('new_ldr_velocity', 0.7)           # 변경할 속도
        self.declare_parameter('default_ldr_velocity', 0.45)      # 기본 속도
        
        # 충돌 관련 파라미터
        self.declare_parameter('caution_speed', 0.3)  
        self.declare_parameter('normal_speed', 0.5)   
        self.declare_parameter('emergency_stop_mode', 3)  

        waypoints_path = self.get_parameter('waypoints_path').get_parameter_value().string_value
        self.look_ahead_distance = self.get_parameter('look_ahead_distance').get_parameter_value().double_value
        self.wheel_base = self.get_parameter('wheel_base').get_parameter_value().double_value
        
        # 체크포인트 파라미터 가져오기
        self.checkpoint_lat = self.get_parameter('checkpoint_lat').get_parameter_value().double_value
        self.checkpoint_lon = self.get_parameter('checkpoint_lon').get_parameter_value().double_value
        self.checkpoint_radius = self.get_parameter('checkpoint_radius').get_parameter_value().double_value
        self.new_ldr_velocity = self.get_parameter('new_ldr_velocity').get_parameter_value().double_value
        self.default_ldr_velocity = self.get_parameter('default_ldr_velocity').get_parameter_value().double_value
        
        self.heading_history = []
        self.heading_filter_size = 5
        self.last_valid_heading = 0.0
        self.heading_jump_threshold = 180.0
        
        # 충돌 관련 파라미터
        self.caution_speed = self.get_parameter('caution_speed').get_parameter_value().double_value
        self.normal_speed = self.get_parameter('normal_speed').get_parameter_value().double_value
        self.emergency_stop_mode = self.get_parameter('emergency_stop_mode').get_parameter_value().integer_value

        if not os.path.exists(waypoints_path):
            self.get_logger().error(f"Waypoints file not found: {waypoints_path}")
            rclpy.shutdown()
            return

        self.waypoints = pd.read_csv(waypoints_path)
        self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints from {waypoints_path}")
        
        self.origin_lat = self.waypoints['latitude'].iloc[0]
        self.origin_lon = self.waypoints['longitude'].iloc[0]
        
        # 체크포인트 로컬 좌표 계산
        self.checkpoint_x, self.checkpoint_y = self.latlon_to_local_xy(self.checkpoint_lat, self.checkpoint_lon)
        
        self.get_logger().info(f"Checkpoint GPS: {self.checkpoint_lat:.6f}, {self.checkpoint_lon:.6f}")
        self.get_logger().info(f"Checkpoint Local: ({self.checkpoint_x:.2f}, {self.checkpoint_y:.2f})")
        self.get_logger().info(f"Checkpoint radius: {self.checkpoint_radius}m")
        self.get_logger().info(f"Speed change: {self.default_ldr_velocity} -> {self.new_ldr_velocity} m/s")

        # GPS 위치 (차량 위치로 직접 사용)
        self.gps_x = 0.0
        self.gps_y = 0.0
        self.current_heading_deg = 0.0
        self.gps_initialized = False
        
        # 충돌 상태 변수
        self.collision_flag = 0  
        self.current_speed = self.normal_speed
        self.current_mode = 1  
        
        # 체크포인트 상태
        self.current_ldr_velocity = self.default_ldr_velocity
        self.checkpoint_passed = False  # 체크포인트 통과 여부 (한 번만!)

        # Publishers
        self.gps_steering_publisher = self.create_publisher(Float32, '/gps_stragl', 10)
        self.alpha_publisher = self.create_publisher(Float32, '/alpha_angle_topic', 10)
        self.ldr_velocity_publisher = self.create_publisher(Float32, '/ldr_velocity_command', 10)
        
        # Vehicle-relative coordinate publishers
        self.vehicle_waypoints_pub = self.create_publisher(PointCloud, '/vehicle_relative_waypoints', 10)
        self.look_ahead_point_pub = self.create_publisher(Point32, '/look_ahead_point_xy', 10)
        
        # ========== RViz2 시각화용 퍼블리셔 ==========
        self.gps_path_pub = self.create_publisher(Path, '/gps_path', 10)
        self.vehicle_marker_pub = self.create_publisher(Marker, '/vehicle_marker', 10)
        self.lookahead_marker_pub = self.create_publisher(Marker, '/lookahead_marker', 10)
        self.heading_arrow_pub = self.create_publisher(Marker, '/heading_arrow', 10)
        self.status_text_pub = self.create_publisher(Marker, '/status_text', 10)
        self.checkpoint_marker_pub = self.create_publisher(Marker, '/checkpoint_marker', 10)
        # ===============================================
        
        # Subscribers
        self.create_subscription(NavSatFix, '/ublox_gps_node/fix', self.gnss_callback, 10)
        self.create_subscription(Float32, 'ublox_gps/course_deg', self.course_callback, 10)
        self.create_subscription(UInt8, '/collision_flag', self.collision_flag_callback, 10)
        
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.plot_thread = threading.Thread(target=self.plot_waypoints_cv2)
        self.plot_thread.daemon = True 
        self.plot_thread.start()
        self.get_logger().info("Visualization thread has been started.")

    def collision_flag_callback(self, msg):
        """충돌 플래그 콜백"""
        self.collision_flag = msg.data
        
        if self.collision_flag == 0:  # 안전
            self.current_speed = self.normal_speed
            self.current_mode = 1  
            
        elif self.collision_flag == 1:  # 주의
            self.current_speed = self.caution_speed
            self.current_mode = 1  
            self.get_logger().warn(f"Caution zone detected! Reducing speed to {self.caution_speed}")
            
        elif self.collision_flag == 2:  # 위험
            self.current_speed = 0.0
            self.current_mode = self.emergency_stop_mode  
            self.get_logger().error("Danger zone detected! Emergency stop activated!")

    def latlon_to_local_xy(self, lat, lon):
        dx = (lon - self.origin_lon) * 88000
        dy = (lat - self.origin_lat) * 111000
        return dx, dy

    def gnss_callback(self, msg):
        # GPS 위치를 차량 위치로 직접 사용
        self.gps_x, self.gps_y = self.latlon_to_local_xy(msg.latitude, msg.longitude)
        
        if not self.gps_initialized:
            self.gps_initialized = True
            self.get_logger().info(
                f"GPS Initialized! Vehicle Position: "
                f"X={self.gps_x:.2f}, Y={self.gps_y:.2f}, "
                f"Heading={self.current_heading_deg:.1f}°"
            )

    def check_checkpoint(self):
        """체크포인트 도달 여부 확인 및 속도 변경 (한 번만!)"""
        if self.checkpoint_passed:
            return  # 이미 통과했으면 더 이상 체크하지 않음
        
        # 체크포인트와의 거리 계산
        distance = math.hypot(self.gps_x - self.checkpoint_x, self.gps_y - self.checkpoint_y)
        
        if distance <= self.checkpoint_radius:
            # 체크포인트 통과!
            self.checkpoint_passed = True
            self.current_ldr_velocity = self.new_ldr_velocity
            
            self.get_logger().info(
                f"🎯 CHECKPOINT REACHED! "
                f"LDR velocity changed: {self.default_ldr_velocity} -> {self.new_ldr_velocity} m/s "
                f"(Distance: {distance:.2f}m)"
            )
            
            # LDR 속도 명령 퍼블리시
            ldr_vel_msg = Float32()
            ldr_vel_msg.data = self.current_ldr_velocity
            self.ldr_velocity_publisher.publish(ldr_vel_msg)
            
    def course_callback(self, msg: Float32):
        """헤딩 각도 콜백 - 각도 정규화 및 필터링 적용"""
        raw_heading = msg.data
        
        # 헤딩 점프 감지 및 보정
        if len(self.heading_history) > 0:
            diff = raw_heading - self.last_valid_heading
            
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            
            if abs(diff) < self.heading_jump_threshold:
                self.last_valid_heading = raw_heading
            else:
                if diff > 0:
                    self.last_valid_heading = raw_heading
                else:
                    self.last_valid_heading = raw_heading
        else:
            self.last_valid_heading = raw_heading
        
        self.heading_history.append(self.last_valid_heading)
        
        if len(self.heading_history) > self.heading_filter_size:
            self.heading_history.pop(0)
        
        if len(self.heading_history) > 0:
            sin_sum = sum(math.sin(math.radians(h)) for h in self.heading_history)
            cos_sum = sum(math.cos(math.radians(h)) for h in self.heading_history)
            filtered_heading = math.degrees(math.atan2(sin_sum, cos_sum))
            
            if filtered_heading < 0:
                filtered_heading += 360
            
            self.current_heading_deg = filtered_heading
        else:
            self.current_heading_deg = raw_heading
    
    def transform_to_vehicle_frame(self, waypoint):
        wx, wy = self.latlon_to_local_xy(waypoint['latitude'], waypoint['longitude'])
        
        theta = math.radians(self.current_heading_deg)
        
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        global_right = [cos_theta, -sin_theta]
        global_forward = [sin_theta, cos_theta]

        T_vehicle_to_global = np.array([
            [global_right[0],  global_forward[0], 0, self.gps_x],
            [global_right[1],  global_forward[1], 0, self.gps_y],
            [0,                0,                 1, 0],
            [0,                0,                 0, 1]
        ])
        T_global_to_vehicle = np.linalg.inv(T_vehicle_to_global)

        point_global = np.array([wx, wy, 0, 1])
        point_vehicle = T_global_to_vehicle.dot(point_global)
        return point_vehicle[0], point_vehicle[1]

    def publish_rviz_visualization(self, relative_waypoints, look_ahead_point):
        """RViz2 시각화를 위한 마커들 퍼블리시"""
        current_time = self.get_clock().now().to_msg()
        
        # 1. GPS 경로 (빨간색 점들) - Path로 표시
        path = Path()
        path.header.frame_id = "velodyne"
        path.header.stamp = current_time

        max_distance = 15.0
        min_dist = float('inf')
        closest_idx = 0
        
        for i, (x, y) in enumerate(relative_waypoints):
            dist = math.hypot(x, y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        accumulated_distance = 0.0
        
        for i in range(closest_idx, len(relative_waypoints)):
            x, y = relative_waypoints[i]
            
            if i == closest_idx:
                segment_distance = math.hypot(x, y)
            else:
                prev_x, prev_y = relative_waypoints[i-1]
                segment_distance = math.hypot(x - prev_x, y - prev_y)
            
            accumulated_distance += segment_distance
            
            if accumulated_distance > max_distance:
                break
            
            if y > 0:
                pose = PoseStamped()
                pose.header.frame_id = "velodyne"
                pose.header.stamp = current_time
                pose.pose.position.x = y
                pose.pose.position.y = -x
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0
                path.poses.append(pose)
        
        self.gps_path_pub.publish(path)
        
        # 2. 체크포인트 마커
        # 체크포인트를 차량 좌표계로 변환
        theta = math.radians(self.current_heading_deg)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        dx = self.checkpoint_x - self.gps_x
        dy = self.checkpoint_y - self.gps_y
        vehicle_x = dx * cos_theta + dy * sin_theta
        vehicle_y = -dx * sin_theta + dy * cos_theta
        
        checkpoint_marker = Marker()
        checkpoint_marker.header.frame_id = "base_link"
        checkpoint_marker.header.stamp = current_time
        checkpoint_marker.ns = "checkpoint"
        checkpoint_marker.id = 0
        checkpoint_marker.type = Marker.CYLINDER
        checkpoint_marker.action = Marker.ADD
        
        checkpoint_marker.pose.position.x = vehicle_x
        checkpoint_marker.pose.position.y = vehicle_y
        checkpoint_marker.pose.position.z = 0.0
        checkpoint_marker.pose.orientation.w = 1.0
        
        # 체크포인트 통과 여부에 따라 색상 변경
        if self.checkpoint_passed:
            checkpoint_marker.color.r = 0.5  # 회색 (통과함)
            checkpoint_marker.color.g = 0.5
            checkpoint_marker.color.b = 0.5
            checkpoint_marker.color.a = 0.3
        else:
            checkpoint_marker.color.r = 1.0  # 빨간색 (미통과)
            checkpoint_marker.color.g = 0.0
            checkpoint_marker.color.b = 0.0
            checkpoint_marker.color.a = 0.7
        
        checkpoint_marker.scale.x = self.checkpoint_radius * 2
        checkpoint_marker.scale.y = self.checkpoint_radius * 2
        checkpoint_marker.scale.z = 0.1
        
        self.checkpoint_marker_pub.publish(checkpoint_marker)
        
        # 3. 차량 위치 마커 (파란색)
        vehicle_marker = Marker()
        vehicle_marker.header.frame_id = "base_link"
        vehicle_marker.header.stamp = current_time
        vehicle_marker.ns = "vehicle"
        vehicle_marker.id = 0
        vehicle_marker.type = Marker.CUBE
        vehicle_marker.action = Marker.ADD
        vehicle_marker.pose.position.x = 0.0
        vehicle_marker.pose.position.y = 0.0
        vehicle_marker.pose.position.z = 0.0
        vehicle_marker.pose.orientation.w = 1.0
        vehicle_marker.scale.x = 1.0
        vehicle_marker.scale.y = 0.5
        vehicle_marker.scale.z = 0.3
        vehicle_marker.color.r = 0.0
        vehicle_marker.color.g = 0.0
        vehicle_marker.color.b = 1.0
        vehicle_marker.color.a = 1.0
        
        self.vehicle_marker_pub.publish(vehicle_marker)
        
        # 4. Look-ahead 포인트 (초록색)
        if look_ahead_point:
            la_marker = Marker()
            la_marker.header.frame_id = "base_link"
            la_marker.header.stamp = current_time
            la_marker.ns = "lookahead"
            la_marker.id = 0
            la_marker.type = Marker.SPHERE
            la_marker.action = Marker.ADD
            la_marker.pose.position.x = look_ahead_point[0]
            la_marker.pose.position.y = look_ahead_point[1]
            la_marker.pose.position.z = 0.0
            la_marker.pose.orientation.w = 1.0
            la_marker.scale.x = 0.4
            la_marker.scale.y = 0.4
            la_marker.scale.z = 0.4
            la_marker.color.r = 0.0
            la_marker.color.g = 1.0
            la_marker.color.b = 0.0
            la_marker.color.a = 1.0
            
            self.lookahead_marker_pub.publish(la_marker)
        
        # 5. 상태 텍스트
        status_marker = Marker()
        status_marker.header.frame_id = "base_link"
        status_marker.header.stamp = current_time
        status_marker.ns = "status"
        status_marker.id = 0
        status_marker.type = Marker.TEXT_VIEW_FACING
        status_marker.action = Marker.ADD
        status_marker.pose.position.x = 0.0
        status_marker.pose.position.y = 5.0
        status_marker.pose.position.z = 2.0
        status_marker.pose.orientation.w = 1.0
        
        # 상태 텍스트 생성
        collision_status = "Safe"
        if self.collision_flag == 1:
            collision_status = f"CAUTION (Speed: {self.current_speed})"
        elif self.collision_flag == 2:
            collision_status = "DANGER - STOP!"
        
        checkpoint_distance = math.hypot(self.gps_x - self.checkpoint_x, self.gps_y - self.checkpoint_y)
        
        status_text = f"GPS Checkpoint Control\n"
        status_text += f"LDR Velocity: {self.current_ldr_velocity} m/s\n"
        status_text += f"Checkpoint: {'PASSED' if self.checkpoint_passed else f'{checkpoint_distance:.1f}m'}\n"
        status_text += f"Collision: {collision_status}"
        
        status_marker.text = status_text
        status_marker.scale.z = 0.4
        
        # 체크포인트 통과 여부에 따라 색상 변경
        if self.checkpoint_passed:
            status_marker.color.r = 0.0
            status_marker.color.g = 1.0
            status_marker.color.b = 0.0
        else:
            status_marker.color.r = 1.0
            status_marker.color.g = 1.0
            status_marker.color.b = 0.0
        status_marker.color.a = 1.0
        
        self.status_text_pub.publish(status_marker)

    def timer_callback(self):
        # GPS 초기화 확인
        if not self.gps_initialized:
            self.get_logger().info("Waiting for initial GPS fix...", throttle_duration_sec=5)
            return

        # 체크포인트 확인 (중요!)
        self.check_checkpoint()

        # 디버깅용 상태 출력 (5초마다)
        current_time = self.get_clock().now()
        if int(current_time.nanoseconds / 1e9) % 5 == 0:
            checkpoint_distance = math.hypot(self.gps_x - self.checkpoint_x, self.gps_y - self.checkpoint_y)
            checkpoint_status = "PASSED" if self.checkpoint_passed else f"{checkpoint_distance:.1f}m"
            self.get_logger().info(
                f"[STATUS] GPS: ({self.gps_x:.2f}, {self.gps_y:.2f}), "
                f"LDR Vel: {self.current_ldr_velocity}, "
                f"Checkpoint: {checkpoint_status}, "
                f"Collision: {self.collision_flag}"
            )

        # Pure Pursuit 알고리즘
        relative_waypoints = [self.transform_to_vehicle_frame(row) for _, row in self.waypoints.iterrows()]
        
        # 차량 중심 웨이포인트 퍼블리시
        waypoint_cloud = PointCloud()
        waypoint_cloud.header.frame_id = "velodyne"
        waypoint_cloud.header.stamp = self.get_clock().now().to_msg()
        
        for x, y in relative_waypoints:
            point = Point32()
            point.x = float(x)
            point.y = float(y)
            point.z = 0.0
            waypoint_cloud.points.append(point)
        
        self.vehicle_waypoints_pub.publish(waypoint_cloud)
        
        forwards = [(x, y) for (x, y) in relative_waypoints if y > 0]
        
        look_ahead_point = None
        for (x, y) in sorted(forwards, key=lambda p: math.hypot(p[0], p[1])):
            if math.hypot(x, y) >= self.look_ahead_distance:
                look_ahead_point = (x, y)
                break
        
        steering_angle_rad = 0.0
        alpha_rad = 0.0  
        
        if look_ahead_point:
            lx, ly = look_ahead_point
            alpha_rad = math.atan2(lx, ly)  
            Ld = math.hypot(lx, ly)
            steering_angle_rad = math.atan2(2.0 * self.wheel_base * math.sin(alpha_rad), Ld)
            
            # Look-ahead 포인트 퍼블리시
            la_point = Point32()
            la_point.x = float(look_ahead_point[0])
            la_point.y = float(look_ahead_point[1])
            la_point.z = 0.0
            self.look_ahead_point_pub.publish(la_point)
        
        # Alpha 각도 퍼블리시 (MPC에서 사용)
        alpha_msg = Float32()
        alpha_msg.data = math.degrees(alpha_rad)
        self.alpha_publisher.publish(alpha_msg)
        
        # LDR 속도 명령 지속적으로 퍼블리시 (selector가 받을 수 있도록)
        ldr_vel_msg = Float32()
        ldr_vel_msg.data = self.current_ldr_velocity
        self.ldr_velocity_publisher.publish(ldr_vel_msg)
        
        # RViz2 시각화 퍼블리시
        self.publish_rviz_visualization(relative_waypoints, look_ahead_point)
        
        # Steering angle 퍼블리시
        steering_msg = Float32()
        steering_msg.data = math.degrees(steering_angle_rad) / 20.0
        self.gps_steering_publisher.publish(steering_msg)
            
    def plot_waypoints_cv2(self):
        window_name = "GPS Checkpoint Control - Vehicle View"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        img_width, img_height = 800, 600
        visualization_range = 20.0
        scale = 300 / visualization_range
        center_x = img_width // 2
        center_y = img_height - 50
        
        while rclpy.ok() and not self.is_shutdown:
            relative_waypoints = [self.transform_to_vehicle_frame(row) for _, row in self.waypoints.iterrows()]
            filtered_waypoints = [(x, y) for (x, y) in relative_waypoints if y > 0 and math.hypot(x, y) <= visualization_range]
            
            img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            
            # 웨이포인트 그리기 (빨간색)
            for (x, y) in filtered_waypoints:
                img_x = int(center_x + x * scale)
                img_y = int(center_y - y * scale)
                cv2.circle(img, (img_x, img_y), 2, (0, 0, 255), -1)
            
            # 체크포인트 그리기
            theta = math.radians(90 - self.current_heading_deg)
            dx = self.checkpoint_x - self.gps_x
            dy = self.checkpoint_y - self.gps_y
            vehicle_x = dx * math.cos(theta) - dy * math.sin(theta)
            vehicle_y = dx * math.sin(theta) + dy * math.cos(theta)
            
            if math.hypot(vehicle_x, vehicle_y) <= visualization_range:
                checkpoint_img_x = int(center_x + vehicle_x * scale)
                checkpoint_img_y = int(center_y - vehicle_y * scale)
                radius_px = int(self.checkpoint_radius * scale)
                
                # 체크포인트 통과 여부에 따라 색상 변경
                if self.checkpoint_passed:
                    color = (128, 128, 128)  # 회색 (통과함)
                    thickness = 2
                else:
                    color = (0, 0, 255)  # 빨간색 (미통과)
                    thickness = 3
                
                cv2.circle(img, (checkpoint_img_x, checkpoint_img_y), radius_px, color, thickness)
                cv2.putText(img, "CHECKPOINT", 
                           (checkpoint_img_x - 40, checkpoint_img_y - radius_px - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # 차량 위치 (파란색)
            cv2.circle(img, (center_x, center_y), 10, (255, 0, 0), -1)
            
            # 차량 진행 방향 화살표
            arrow_end = (center_x, center_y - int(4.0 * scale))
            cv2.arrowedLine(img, (center_x, center_y), arrow_end, (255, 0, 0), 3)
            
            # 정보 텍스트
            cv2.putText(img, f"GPS Checkpoint Control", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 현재 LDR 속도 표시
            speed_text = f"LDR Velocity: {self.current_ldr_velocity:.2f} m/s"
            speed_color = (0, 255, 0) if self.checkpoint_passed else (255, 255, 255)
            cv2.putText(img, speed_text, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, speed_color, 2)
            
            # 체크포인트 상태
            checkpoint_distance = math.hypot(self.gps_x - self.checkpoint_x, self.gps_y - self.checkpoint_y)
            if self.checkpoint_passed:
                checkpoint_text = "Checkpoint: PASSED ✓"
                checkpoint_color = (0, 255, 0)
            else:
                checkpoint_text = f"Checkpoint: {checkpoint_distance:.1f}m"
                checkpoint_color = (0, 0, 255)
            
            cv2.putText(img, checkpoint_text, (10, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, checkpoint_color, 2)
            
            # 충돌 상태 표시
            collision_text = "Safe"
            collision_color = (0, 255, 0)  
            if self.collision_flag == 1:
                collision_text = f"CAUTION"
                collision_color = (0, 165, 255)  
            elif self.collision_flag == 2:
                collision_text = "DANGER - EMERGENCY STOP!"
                collision_color = (0, 0, 255)  
            
            cv2.putText(img, f"Collision: {collision_text}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, collision_color, 1)

            cv2.imshow(window_name, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.1)
        cv2.destroyAllWindows()
    
    def destroy(self):
        """Override destroy to handle cleanup"""
        self.is_shutdown = True
        if hasattr(self, 'plot_thread'):
            self.plot_thread.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = MissionControlNode()
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            executor.shutdown()
            node.destroy_node()
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
