import cv2
import numpy as np
import matplotlib as plt
from scipy.interpolate import splrep, splev
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from my_lane_msgs.msg import LanePoints

#----전역변수 지정----
LEFT_LANE = None
RIGHT_LANE = None
OFFSET = 185

#----차선좌표 Subscribe (LanePoints 메시지 구독)----
class LaneSubscriber(Node):
    def __init__(self):
        super().__init__('lane_subscriber')
        self.lane_path_data = None
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
        self.lane_path_data = msg
        self.left_lane_x = msg.left_x
        self.left_lane_y = msg.left_y
        self.right_lane_x = msg.right_x
        self.right_lane_y = msg.right_y
        self.get_logger().info(
            f"Received LanePoints: left: {len(msg.left_x)} points, right: {len(msg.right_x)} points"
        )

#----경로좌표 Publish----
class pathpublisher(Node):
    def __init__(self, pathpoints):
        super().__init__('Path_Publish')
        self.pathpoints = pathpoints
        self.publisher_ = self.create_publisher(Float32MultiArray, 'path_publish', 10)
        self.timer = self.create_timer(1.0, self.publish_tuple_list)
    
    def publish_tuple_list(self):
        self.publishing_path()

    def publishing_path(self):
        msg = Float32MultiArray()
        flat_path = [float(value) for tup in self.pathpoints for value in tup]
        msg.data = flat_path
        self.get_logger().info(f'Publishing path: {self.pathpoints}')
        self.publisher_.publish(msg)

#----좌우측 차선이 모두 존재할 때----
def lane_both(leftx, lefty, rightx, righty):    
    if len(leftx) >= len(rightx):
        division = len(leftx) // len(rightx)
        midx = []
        midy = []
        for i in range(len(rightx)):
            for j in range(i*division, (i+1)*division):
                midx.append((rightx[i] + leftx[j]) / 2)
                midy.append((righty[i] + lefty[j]) / 2)
    else:
        division = len(rightx) // len(leftx)
        midx = []
        midy = []
        for i in range(len(leftx)):
            for j in range(i*division, (i+1)*division):
                midx.append((leftx[i] + rightx[j]) / 2)
                midy.append((lefty[i] + righty[j]) / 2)
    return midx, midy 

#----좌측 차선만 존재할 때----
def lane_left(lanex, laney, offset=-OFFSET, num_points=50):
    if len(lanex) < 2 or len(laney) < 2:
        return lanex, laney

    xs = np.array(lanex)
    ys = np.array(laney)
    order = np.argsort(ys)
    ys_sorted = ys[order]
    xs_sorted = xs[order]

    coeffs = np.polyfit(ys_sorted, xs_sorted, 3)
    poly = np.poly1d(coeffs)
    dpoly = poly.deriv()

    y_vals = np.linspace(np.min(ys_sorted), np.max(ys_sorted), num=num_points)
    x_vals = poly(y_vals)
    slopes = dpoly(y_vals)
    norms = np.sqrt(1 + slopes**2)

    normal_x = 1.0 / norms
    normal_y = -slopes / norms

    path_x = x_vals - offset * normal_x
    path_y = y_vals - offset * normal_y

    return path_x.tolist(), path_y.tolist()

#----우측 차선만 존재할 때----
def lane_right(lanex, laney, offset=-OFFSET, num_points=50):
    if len(lanex) < 2 or len(laney) < 2:
        return lanex, laney

    xs = np.array(lanex)
    ys = np.array(laney)
    order = np.argsort(ys)
    ys_sorted = ys[order]
    xs_sorted = xs[order]

    coeffs = np.polyfit(ys_sorted, xs_sorted, 3)
    poly = np.poly1d(coeffs)
    dpoly = poly.deriv()

    y_vals = np.linspace(np.min(ys_sorted), np.max(ys_sorted), num=num_points)
    x_vals = poly(y_vals)
    slopes = dpoly(y_vals)
    norms = np.sqrt(1 + slopes**2)

    normal_x = 1.0 / norms
    normal_y = -slopes / norms

    path_x = x_vals + offset * normal_x
    path_y = y_vals + offset * normal_y

    return path_x.tolist(), path_y.tolist()

#----계산 플래그----
def what_def_use(leftx, rightx):
    if not rightx: return 1
    if not leftx: return 2
    return 0

#----중점 선형화 및 재좌표화----
# def mid_linear(pathx, pathy, pixel_interval=10):
#     path_points = []
#     if not pathx or not pathy:
#         print("Warning: mid_linear received empty path data. Returning empty path.")
#         return path_points

#     coeffs_mid = np.polyfit(pathx, pathy, 3)
#     poly_mid_function = np.poly1d(coeffs_mid)

#     y_max = max(pathy)
#     y_min = min(pathy)

#     for y_val in np.arange(y_max, y_min - 1, -pixel_interval):
#         points_coeffs_mid = coeffs_mid.copy()
#         points_coeffs_mid[-1] -= y_val

#         xfromy = np.roots(points_coeffs_mid)
#         valid_xfromy = [r.real for r in xfromy if np.abs(r.imag) < 1e-6 and min(pathx) <= r.real <= max(pathx)]
#         if valid_xfromy:
#             x_val = min(valid_xfromy, key=lambda r: abs(r - np.mean(pathx)))
#             path_points.append((x_val, y_val))

#     return path_points

def splining(pathx, pathy, pixel_interval=10):
    path_points = []
    if not pathx or not pathy:
        print("Warning: splining received empty path data.")
        return path_points
    
    # Numpy 배열로 변환하고 y값 기준으로 정렬
    pathx = np.array(pathx, dtype=float)
    pathy = np.array(pathy, dtype=float)
    sorted_indices = np.argsort(pathy)
    sorted_y = pathy[sorted_indices]
    sorted_x = pathx[sorted_indices]

    # 서로 다른 y 좌표가 최소 4개 이상 있는지 확인
    unique_ys = np.unique(sorted_y)
    if len(unique_ys) < 4:
        print("Not enough distinct points for cubic spline, skipping.")
        return path_points

    # Cubic spline(3차) 생성
    tck = splrep(sorted_y, sorted_x, s=0, k=3)
    
    # y_min ~ y_max 범위에서만 일정 간격으로 평가
    y_min, y_max = np.min(sorted_y), np.max(sorted_y)
    for y_val in np.arange(y_min, y_max, pixel_interval):
        x_val = splev(y_val, tck)  # 기본(ext=0) => 범위 밖은 NaN
        path_points.append((x_val, y_val))

    return path_points


#----메인함수----
def main(args=None):
    rclpy.init(args=args)
    lane_subscriber = LaneSubscriber()
    path_publisher = pathpublisher([])
    try:
        while rclpy.ok():
            rclpy.spin_once(lane_subscriber, timeout_sec=0.1)

            # 토픽 저장
            recieved_left_lane_x = lane_subscriber.left_lane_x
            recieved_left_lane_y = lane_subscriber.left_lane_y
            recieved_right_lane_x = lane_subscriber.right_lane_x
            recieved_right_lane_y = lane_subscriber.right_lane_y

            #경로 list 선언
            path_coord_x = []
            path_coord_y = []

            #좌우측 차선 모두 존재할 때
            if what_def_use(recieved_left_lane_x, recieved_right_lane_x) == 0:
                path_coord_x, path_coord_y =  lane_both(recieved_left_lane_x, 
                                                        recieved_left_lane_y, 
                                                        recieved_right_lane_x, 
                                                        recieved_right_lane_y)
            #좌측 차선만 존재할 때
            elif what_def_use(recieved_left_lane_x, recieved_right_lane_x) == 1:
                path_coord_x, path_coord_y =  lane_left(recieved_left_lane_x, 
                                                        recieved_left_lane_y)
            #우측 차선만 존재할 때
            elif what_def_use(recieved_left_lane_x, recieved_right_lane_x) == 2:
                path_coord_x, path_coord_y =  lane_right(recieved_right_lane_x, 
                                                        recieved_right_lane_y)
            
            final_path = splining(path_coord_x, path_coord_y, 10)
            #final_path = mid_linear(path_coord_x, path_coord_y, 10)

            path_publisher.pathpoints = final_path
            path_publisher.publishing_path()

            # #---굳이 필요없음----
            # # 시각화
            # img = np.zeros((360, 640, 3), dtype=np.uint8)
            # # LanePoints로 수신한 데이터를 시각화 (좌측 차선: 파란색, 우측 차선: 빨간색)
            # for i in range(len(lane_subscriber.left_lane_x)):
            #     cv2.circle(img, (int(lane_subscriber.left_lane_x[i]), int(lane_subscriber.left_lane_y[i])),
            #                5, (255, 0, 0), -1)
            # for i in range(len(lane_subscriber.right_lane_x)):
            #     cv2.circle(img, (int(lane_subscriber.right_lane_x[i]), int(lane_subscriber.right_lane_y[i])),
            #                5, (0, 0, 255), -1)
            # # 경로(final_path)를 초록색 선으로 시각화
            # for i in range(len(final_path) - 1):
            #     cv2.line(img, (int(final_path[i][0]), int(final_path[i][1])),
            #              (int(final_path[i+1][0]), int(final_path[i+1][1])),
            #              (0, 255, 0), 2)
            # for point in final_path:
            #     cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 255, 255), -1)
            
            # cv2.imshow('Lane and Path Visualization', img)
            # cv2.waitKey(1)

    except KeyboardInterrupt:
        pass

    lane_subscriber.destroy_node()
    rclpy.shutdown()    

if __name__ == '__main__':
    main()
