#!/usr/bin/env python3
import rclpy
import math
import numpy as np
import cv2
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray
from my_lane_msgs.msg import LanePoints
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

#----주요 상수 정의----
CAR_POSITION = (320, 359)
WHEELBASE = 70.0
LOOKAHED_DISTANCE = 170
MAX_STEERING_ANGLE = 1.0
VELOCITY = 0.2 #고정 상수설정(필요시)
MIN_SPEED = 0.35
MAX_SPEED = 0.4

#----타겟 좌표 선정----
def find_target_point(path_points, lookahead_distance):
    global CAR_POSITION
    near_point = None
    min_distance = float('inf')
    for (x, y) in path_points:
        distance = math.sqrt((x-CAR_POSITION[0])**2 + (y-CAR_POSITION[1])**2)
        if distance >= lookahead_distance and distance < min_distance:
            near_point = (x, y)
            min_distance = distance
    return near_point

# ---- 조향각 계산 ----
def compute_steering_angle(target_x, target_y, wheelbase, max_steering_angle, car_position):
    dx = target_x - car_position[0]
    dy = -(target_y - car_position[1])
    # print(f"dx : {dx}")
    # print(f"dy : {dy}")
    # print(f"target : ({target_x , target_y})")
    alpha = math.atan2(dx, dy)
    print(f"alpha : {alpha}")
    if abs(alpha) < 1e-6:
        return 0.0
    steering_angle = math.atan2(2.0 * wheelbase * math.sin(alpha), math.sqrt((dx)**2 + (dy)**2))
    print(f"steering angle : {steering_angle}")
    return np.clip(steering_angle, -max_steering_angle, max_steering_angle), alpha

# ---- 속도 계산 ----
def compute_velocity(alpha, min_speed = MIN_SPEED, max_speed = MAX_SPEED):
    temp_alpha = abs(alpha)
    constrained_alpha = (np.pi/2) - temp_alpha

    # if abs(alpha) < 0.3:
    #     velocity = 0.5
    # else:
    #     velocity = 0.35
    if constrained_alpha > 1:
        constrained_alpha = 1
    if constrained_alpha < 0 :
        constrained_alpha = 0 

    velocity = min_speed + (max_speed - min_speed) * constrained_alpha
    #velocity = VELOCITY
    return velocity

# ---- 속도 퍼블리시 ----
def publish_velocity(velocity, pub, node):
    vel_msg = Float32()
    vel_msg.data = velocity
    pub.publish(vel_msg)
    node.get_logger().info(f"Published velocity: {velocity:.2f}")

# ---- 조향각 퍼블리시 ----
def publish_steering_angle(steering_angle, pub, node):
    steer_msg = Float32()
    steer_msg.data = steering_angle
    pub.publish(steer_msg)
    node.get_logger().info(f"Published steering angle: {steering_angle:.2f}")

# ---- 차선 좌표 Subscribe (LanePoints 메시지 구독) ----
class LaneSubscriber(Node):
    def __init__(self):
        super().__init__('lane_subscriber')
        self.lane_path_subscriber = self.create_subscription(
            LanePoints,
            'lane_points',
            self.lane_path_callback,
            10
        )
        # 구독한 데이터를 저장할 변수 초기화
        self.left_lane_x = []
        self.left_lane_y = []
        self.right_lane_x = []
        self.right_lane_y = []

    def lane_path_callback(self, msg):
        self.left_lane_x = msg.left_x
        self.left_lane_y = msg.left_y
        self.right_lane_x = msg.right_x
        self.right_lane_y = msg.right_y
        self.get_logger().info("-------------------------------------------------------")
        self.get_logger().info(
            f"Received LanePoints: left: {len(msg.left_x)} points, right: {len(msg.right_x)} points"
        )

# ---- 경로 좌표 Subscribe (Float32MultiArray 메시지 구독) ----
class PathSubscriber_HighControl(Node):
    def __init__(self):
        super().__init__('path_subscriber_highcontrol')
        self.planned_path_subscriber = self.create_subscription(
            Float32MultiArray,
            'path_publish',
            self.path_callback,
            10
        )
        # 경로 좌표 저장 리스트 (각 원소: (x, y))
        self.PATH_POINTS = []

    def path_callback(self, msg: Float32MultiArray):
        data = msg.data
        if len(data) < 2:
            self.get_logger().warn("Received empty path data!")
            return
        # flat한 리스트를 (x, y) 튜플 리스트로 변환
        self.PATH_POINTS = [(data[i], data[i+1]) for i in range(0, len(data), 2) if i+1 < len(data)]
        self.get_logger().info(f"Received {len(self.PATH_POINTS)} path points.")

# ---- Pure Pursuit Controller ----
class PurePursuitController:
    def __init__(self, lookahead_distance, wheelbase, max_steering_angle):
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase
        self.max_steering_angle = max_steering_angle

    def compute_PurePursuit(self, pathpoints):
        # 목표점 선택 및 제어 계산
        target = find_target_point(pathpoints, self.lookahead_distance)
        if target is None:
            print("No valid target point found.")
            return None
        target_x, target_y = target
        target = (int(target_x), int(target_y))
        steering_angle, alpha = compute_steering_angle(target_x, target_y, self.wheelbase,
                                                self.max_steering_angle, CAR_POSITION)
        velocity = compute_velocity(alpha)
        return steering_angle, velocity, target

# ---- 시각화 이미지 퍼블리셔 ----
class VisualizationPublisher(Node):
    def __init__(self):
        super().__init__('visualization_publisher')
        
        # Image 퍼블리셔 생성
        self.image_pub = self.create_publisher(Image, '/pure_pursuit_visualization', 10)
        
        # CvBridge 생성
        self.bridge = CvBridge()
        
    def create_visualization_image(self, lane_path, path_points, target, controller):
        # 빈 이미지 생성 (검은색 배경)
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        
        # 차선 플로팅
        for i in range(len(lane_path.left_lane_x)):
            cv2.circle(img, (int(lane_path.left_lane_x[i]), int(lane_path.left_lane_y[i])),
                       5, (255, 0, 0), -1)  # 파란색
        for i in range(len(lane_path.right_lane_x)):
            cv2.circle(img, (int(lane_path.right_lane_x[i]), int(lane_path.right_lane_y[i])),
                       5, (0, 0, 255), -1)  # 빨간색
        
        # 경로 플로팅
        for i in range(len(path_points) - 1):
            cv2.line(img, (int(path_points[i][0]), int(path_points[i][1])),
                     (int(path_points[i+1][0]), int(path_points[i+1][1])),
                     (0, 255, 0), 2)  # 초록색 선
        for point in path_points:
            cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 255, 255), -1)  # 노란색 점
        
        # 차량을 중심으로 lookahead 거리를 표시하는 원
        cv2.circle(img, CAR_POSITION, int(controller.lookahead_distance), (0, 0, 255), 1)

        # 차량 위치
        cv2.circle(img, CAR_POSITION, 5, (0, 0, 255), -1)  # 빨간색

        # 타겟 및 조향각 플로팅
        if target:
            cv2.circle(img, target, 5, (0, 0, 255), -1)  # 타겟 포인트
            cv2.line(img, CAR_POSITION, target, (0, 255, 0), 4)  # 차량에서 타겟까지 선
        
        # 정보 텍스트 추가
        cv2.putText(img, "Pure Pursuit Controller", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Lookahead: {controller.lookahead_distance}px", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
    
    def publish_image(self, cv_image):
        # OpenCV 이미지를 ROS Image 메시지로 변환
        try:
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.image_pub.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f'Failed to publish image: {e}')

def main(args=None):
    global CAR_POSITION, LOOKAHED_DISTANCE, WHEELBASE, MAX_STEERING_ANGLE
    rclpy.init(args=args)
    
    #----노드 생성----
    lane_path = LaneSubscriber()
    path_subscriber = PathSubscriber_HighControl()
    control_publisher_node = Node("Control_Publisher")
    visualization_publisher = VisualizationPublisher()
    
    #----퍼블리셔 생성----
    # velocity_publisher = control_publisher_node.create_publisher(Float32, '/throttle_topic', 10)
    steering_angle_publisher = control_publisher_node.create_publisher(Float32, '/lane_stragl', 10)
    
    #----컨트롤러 생성----
    controller = PurePursuitController(LOOKAHED_DISTANCE, WHEELBASE, MAX_STEERING_ANGLE)
    
    try:
        while rclpy.ok():
            # 두 노드의 콜백을 주기적으로 처리
            rclpy.spin_once(path_subscriber, timeout_sec=0.1)
            rclpy.spin_once(lane_path, timeout_sec=0.1)
            rclpy.spin_once(visualization_publisher, timeout_sec=0.1)
            
            path_points = path_subscriber.PATH_POINTS
            if path_points:
                result = controller.compute_PurePursuit(path_points)
                if result is None:
                    continue  # 타겟이 없으면 건너뛰기
                steering_angle, velocity, target = result
                
                # publish_velocity(velocity, velocity_publisher, control_publisher_node)
                publish_steering_angle(steering_angle, steering_angle_publisher, control_publisher_node)
                
                # 시각화 이미지 생성 및 퍼블리시
                vis_image = visualization_publisher.create_visualization_image(
                    lane_path, path_points, target, controller
                )
                visualization_publisher.publish_image(vis_image)
                
                # 로컬에서도 OpenCV 창으로 보기 (선택사항)
                # cv2.imshow("PATH PLAN AND CONTROL", vis_image)
                # cv2.waitKey(1)

    except KeyboardInterrupt:
        print("Shutting down Pure Pursuit Controller")
    finally:
        lane_path.destroy_node()
        path_subscriber.destroy_node()
        control_publisher_node.destroy_node()
        visualization_publisher.destroy_node()
        # cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()