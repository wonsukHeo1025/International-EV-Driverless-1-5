#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, UInt8

# 기본 속도 설정
DEFAULT_LDR_VELOCITY = 1.2  # 기본 LiDAR 기반 주행 속도 (GPS 체크포인트에서 변경 가능)
LANE_VELOCITY = 1.0          # 차선 주행 속도
YELLOW_VELOCITY = 0.5        # 주의 속도
RED_VELOCITY = 0.0           # 정지

class Selector(Node):
    def __init__(self):
        super().__init__('Selector')

        # 초기 값
        self.lane_angle = 0.0
        self.ldr_angle = 0.0
        self.lane_detected = 0
        self.collision_flag = 0
        
        # ⭐ GPS 체크포인트 속도 제어 추가 ⭐
        self.ldr_velocity = DEFAULT_LDR_VELOCITY  # mission_control에서 업데이트됨

        # 디버깅용 변수
        self.current_mode = "LDR"  # 기본 모드는 LDR
        self.mode_change_count = 0

        # 구독
        self.create_subscription(Float32, '/lane_stragl', self.lane_callback, 10)
        self.create_subscription(Float32, '/ldr_stragl', self.ldr_callback, 10)
        self.create_subscription(UInt8, '/lane_detected', self.lane_detected_callback, 10)
        self.create_subscription(UInt8, '/collision_flag', self.collision_flag_callback, 10)
        
        # ⭐ GPS 체크포인트에서 오는 LDR 속도 명령 구독 ⭐
        self.create_subscription(Float32, '/ldr_velocity_command', self.ldr_velocity_callback, 10)

        # 퍼블리셔
        self.steering_pub = self.create_publisher(Float32, '/steering_angle', 10)
        self.velocity_pub = self.create_publisher(Float32, '/target_speed', 10)

        # 주기적 실행 (10Hz)
        self.create_timer(0.1, self.main_loop)
        
        self.get_logger().info(f'🚗 Selector with GPS Speed Control initialized')
        self.get_logger().info(f'Default LDR velocity: {self.ldr_velocity} m/s')

    def lane_callback(self, msg: Float32):
        self.lane_angle = msg.data
        self.get_logger().debug(f'Lane angle updated: {self.lane_angle:.2f}')

    def ldr_callback(self, msg: Float32):
        self.ldr_angle = msg.data
        self.get_logger().debug(f'LDR angle updated: {self.ldr_angle:.2f}')

    def lane_detected_callback(self, msg: UInt8):
        if self.lane_detected != msg.data:
            self.get_logger().info(f'Lane detection changed: {self.lane_detected} -> {msg.data}')
        self.lane_detected = msg.data

    def collision_flag_callback(self, msg: UInt8):
        if self.collision_flag != msg.data:
            self.get_logger().warn(f'Collision flag changed: {self.collision_flag} -> {msg.data}')
        self.collision_flag = msg.data
    
    def ldr_velocity_callback(self, msg: Float32):
        """⭐ GPS 체크포인트에서 오는 LDR 속도 명령 ⭐"""
        prev_velocity = self.ldr_velocity
        self.ldr_velocity = msg.data
        
        if abs(prev_velocity - self.ldr_velocity) > 0.01:  # 속도 변경 감지
            self.get_logger().info(f'🔄 LDR velocity updated by GPS checkpoint: {prev_velocity:.2f} -> {self.ldr_velocity:.2f} m/s')

    def main_loop(self):
        try:
            prev_mode = self.current_mode
            
            # 긴급정지가 최우선
            if self.collision_flag == 2:
                # 긴급정지 (어떤 상황에서도 적용)
                steering = 0.0#self.lane_angle  # 직진 상태로 정지
                velocity = RED_VELOCITY
                self.current_mode = "STOP (Red)"
            
            elif self.lane_detected == 1:
                # 차선이 감지된 경우
                if self.collision_flag == 1:
                    # 차선 모드에서 장애물 감지 → LDR로 전환
                    steering = self.ldr_angle
                    velocity = YELLOW_VELOCITY
                    self.current_mode = "LDR (Obstacle Avoidance)"
                else:
                    # 차선 모드 정상 주행
                    steering = self.lane_angle
                    velocity = LANE_VELOCITY
                    self.current_mode = "LANE"
            
            else:
                # 차선이 감지되지 않은 경우 → LDR 주행
                # ⭐ 여기서 GPS 체크포인트에서 설정한 동적 LDR 속도 사용! ⭐
                steering = self.ldr_angle
                velocity = self.ldr_velocity  # GPS 체크포인트에 의해 변경될 수 있는 속도!
                self.current_mode = f"LDR (v={self.ldr_velocity:.2f})"
                
                # LDR 모드에서 collision_flag=1은 정상 동작이므로 특별한 처리 불필요

            # 모드 변경 감지 및 로깅
            if prev_mode != self.current_mode:
                self.mode_change_count += 1
                self.get_logger().info(f'[MODE CHANGE #{self.mode_change_count}] {prev_mode} -> {self.current_mode}')

            # 현재 상태 로깅 (2초에 한 번)
            if int(self.get_clock().now().nanoseconds / 1e9) % 2 == 0:
                self.get_logger().info(
                    f'[STATUS] Mode: {self.current_mode} | '
                    f'Steering: {steering:.2f} | Velocity: {velocity:.2f} | '
                    f'Collision: {self.collision_flag} | Lane: {self.lane_detected}'
                )

            # 디버그 레벨 상세 로깅
            self.get_logger().debug(
                f'[DEBUG] Angles - Lane: {self.lane_angle:.2f}, '
                f'LDR: {self.ldr_angle:.2f}, LDR_Vel: {self.ldr_velocity:.2f}'
            )

            # 조향각 및 속도 퍼블리시
            steer_msg = Float32()
            steer_msg.data = steering
            self.steering_pub.publish(steer_msg)

            vel_msg = Float32()
            vel_msg.data = velocity
            self.velocity_pub.publish(vel_msg)
            
        except Exception as e:
            self.get_logger().error(f"Main loop error: {e}")
            # 에러 발생 시 안전하게 정지
            steer_msg = Float32()
            steer_msg.data = 0.0
            self.steering_pub.publish(steer_msg)
            
            vel_msg = Float32()
            vel_msg.data = 0.0
            self.velocity_pub.publish(vel_msg)

def main(args=None):
    try:
        rclpy.init(args=args)
        node = Selector()
        
        # 시작 메시지
        node.get_logger().info('🚗 Selector with GPS Speed Control started successfully')
        node.get_logger().info(f'Initial mode: {node.current_mode}')
        node.get_logger().info('Listening for GPS checkpoint speed commands...')
        
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info('Node stopped by user')
        finally:
            node.destroy_node()
            rclpy.shutdown()
            
    except Exception as e:
        print(f"Fatal error in main: {e}")
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
