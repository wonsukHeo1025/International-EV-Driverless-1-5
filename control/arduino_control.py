#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, String
import serial
import time

class ArduinoSerialNode(Node):
    def __init__(self):
        super().__init__('arduino_serial_node')

        # 개별 제어값을 구독하는 서브스크라이버 생성
        self.steering_subscription = self.create_subscription(
            Float32, '/steering_angle', self.steering_callback, 10)

        self.throttle_subscription = self.create_subscription(
            Float32, '/throttle_topic', self.throttle_callback, 10)

        self.mode_subscription = self.create_subscription(
            Int32, '/mode_topic', self.mode_callback, 10)

        # 아두이노 응답을 발행할 퍼블리셔 생성
        self.response_publisher = self.create_publisher(String, '/arduino_response_topic', 10)

        # 시리얼 포트 설정 (포트와 보드레이트는 환경에 맞게 수정)
        port = '/dev/ttyUSB0'
        baud_rate = 115200
        try:
            self.ser = serial.Serial(port, baud_rate, timeout=0.1)
            time.sleep(2)  # 아두이노 리셋 대기 시간
            self.get_logger().info(f"Serial port {port} opened.")
        except Exception as e:
            self.get_logger().error(f"Failed to open serial port: {e}")
            self.ser = None

        # 초기 제어값 설정 (초기값은 0)
        self.steering_angle = 0.0
        self.throttle = float(0.5)#0.3
        self.current_mode = 1

        # 타이머를 이용해 논블로킹 방식으로 아두이노 응답을 폴링
        self.timer = self.create_timer(0.1, self.serial_poll_callback)

    def steering_callback(self, msg):
        """ 스티어링 값 갱신 """
        self.steering_angle = -msg.data
        #if self.steering_angle < 0.01 and self.steering_angle >-0.01:
        #    msg.data = 0.0;
        self.send_to_arduino()

    def throttle_callback(self, msg):
        """ 스로틀 값 갱신 """
        self.throttle = round(float(msg.data), 1)
        #self.throttle = float(0.4)
        self.send_to_arduino()

    def mode_callback(self, msg):
        """ 모드 값 갱신 """
        self.current_mode = msg.data
        self.send_to_arduino()

    def send_to_arduino(self):
        """ 갱신된 제어값을 아두이노로 전송 """
        if self.ser is None:
            self.get_logger().error("Serial port not available.")
            return

        # 메시지 구성 및 전송
        send_str = f"{self.steering_angle},{self.throttle},{self.current_mode}\n"
        self.ser.write(send_str.encode('utf-8'))
        self.get_logger().info(f"Sent to Arduino: {send_str.strip()}")

    def serial_poll_callback(self):
        """ 타이머 콜백을 통해 시리얼 포트에서 아두이노 응답을 주기적으로 확인 """
        if self.ser is None:
            return

        try:
            response = self.ser.readline().decode('utf-8').strip()
            if response:
                self.get_logger().info(f"Received from Arduino: {response}")
                # 응답을 토픽에 발행
                response_msg = String()
                response_msg.data = response
                self.response_publisher.publish(response_msg)
        except Exception as e:
            self.get_logger().error(f"Serial read error: {e}")

    def destroy_node(self):
        if self.ser is not None:
            self.ser.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ArduinoSerialNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

