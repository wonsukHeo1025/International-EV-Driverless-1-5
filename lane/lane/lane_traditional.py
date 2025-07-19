#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from my_lane_msgs.msg import LanePoints

# --------------------- 상수 설정 --------------------- #
MIN_LANE_SEPARATION = 300
LEFT_HIST_THRESHOLD = 1200
RIGHT_HIST_THRESHOLD = 1200

# --------------------- 칼만 필터 클래스 --------------------- #
class LaneKalmanFilter:
    """차선 추적을 위한 칼만 필터
    
    상태 벡터: [x, vx, ax] - 위치, 속도, 가속도
    측정 벡터: [x] - 위치만 측정
    """
    def __init__(self):
        # 상태 차원: 3 (위치, 속도, 가속도)
        # 측정 차원: 1 (위치만)
        self.kalman = cv2.KalmanFilter(3, 1)
        
        # 시간 간격 (프레임 간격)
        dt = 1.0
        
        # 상태 전이 행렬 (등가속도 운동 모델)
        self.kalman.transitionMatrix = np.array([
            [1, dt, 0.5*dt*dt],
            [0, 1, dt],
            [0, 0, 1]
        ], np.float32)
        
        # 측정 행렬 (위치만 측정)
        self.kalman.measurementMatrix = np.array([[1, 0, 0]], np.float32)
        
        # 프로세스 노이즈 공분산
        self.kalman.processNoiseCov = np.array([
            [0.01, 0, 0],
            [0, 0.01, 0],
            [0, 0, 0.01]
        ], np.float32)
        
        # 측정 노이즈 공분산
        self.kalman.measurementNoiseCov = np.array([[0.1]], np.float32)
        
        # 초기 추정 오차 공분산
        self.kalman.errorCovPost = np.eye(3, dtype=np.float32)
        
        self.initialized = False
        self.lost_track_count = 0
        
    def update(self, measurement):
        """새로운 측정값으로 칼만 필터 업데이트"""
        if measurement is None:
            # 측정값이 없으면 예측만 수행
            self.lost_track_count += 1
            if self.lost_track_count > 5:
                # 5프레임 이상 추적 실패 시 초기화
                self.initialized = False
                return None
            prediction = self.kalman.predict()
            return prediction[0][0]
        
        self.lost_track_count = 0
        measurement_array = np.array([[measurement]], np.float32)
        
        if not self.initialized:
            # 첫 측정값으로 초기화
            self.kalman.statePre = np.array([[measurement], [0], [0]], np.float32)
            self.kalman.statePost = np.array([[measurement], [0], [0]], np.float32)
            self.initialized = True
            return measurement
            
        # 예측
        self.kalman.predict()
        
        # 보정
        corrected = self.kalman.correct(measurement_array)
        
        return corrected[0][0]

# --------------------- 차선별 칼만 필터 관리 클래스 --------------------- #
class LaneKalmanManager:
    """각 윈도우별로 칼만 필터를 관리하는 클래스"""
    def __init__(self, num_windows):
        self.filters = [LaneKalmanFilter() for _ in range(num_windows)]
        
    def update_window(self, window_idx, measurement):
        """특정 윈도우의 칼만 필터 업데이트"""
        if window_idx < len(self.filters):
            return self.filters[window_idx].update(measurement)
        return measurement

# --------------------- RANSAC, DBSCAN, 전처리 함수 --------------------- #
def ransac_filter_points(points_x, points_y, degree=2):
    if len(points_y) < 5 or len(points_x) < 5:
        return points_x, points_y
    
    # 포인트 개수 불일치 확인
    if len(points_x) != len(points_y):
        min_len = min(len(points_x), len(points_y))
        points_x = points_x[:min_len]
        points_y = points_y[:min_len]
    
    try:
        X = np.array(points_y).reshape(-1, 1)
        y = np.array(points_x)
        model = make_pipeline(PolynomialFeatures(degree), RANSACRegressor(residual_threshold=25, max_trials=500))
        model.fit(X, y)
        inlier_mask = model.named_steps['ransacregressor'].inlier_mask_
        return y[inlier_mask].tolist(), X[inlier_mask].flatten().tolist()
    except Exception as e:
        print(f"RANSAC error: {e}")
        return points_x, points_y

def filter_lane_points_dbscan(points_x, points_y, eps=15, base_min_samples=4):
    if len(points_x) == 0 or len(points_y) == 0:
        return [], []
    
    # 포인트 개수 불일치 확인
    if len(points_x) != len(points_y):
        min_len = min(len(points_x), len(points_y))
        points_x = points_x[:min_len]
        points_y = points_y[:min_len]
    
    try:
        dynamic_min_samples = base_min_samples
        if len(points_x) < 50:
            dynamic_min_samples = max(2, len(points_x) // 10)
        
        data = np.column_stack((points_x, points_y))
        clustering = DBSCAN(eps=eps, min_samples=dynamic_min_samples).fit(data)
        labels = clustering.labels_
        valid_labels = labels[labels != -1]
        
        if len(valid_labels) == 0:
            return points_x, points_y
        
        unique, counts = np.unique(valid_labels, return_counts=True)
        largest_cluster = unique[np.argmax(counts)]
        filtered_data = data[labels == largest_cluster]
        return filtered_data[:, 0].tolist(), filtered_data[:, 1].tolist()
    except Exception as e:
        print(f"DBSCAN error: {e}")
        return points_x, points_y

def preprocess_lane(image):
    try:
        hls_img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        l_channel = hls_img[:, :, 1]
        _, mask = cv2.threshold(l_channel, 230, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask
    except Exception as e:
        print(f"Preprocess error: {e}")
        return np.zeros_like(image[:,:,0])

def apply_birds_eye_view(binary_mask, src_points, dst_points):
    try:
        h, w = binary_mask.shape[:2]
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(binary_mask, M, (w, h), flags=cv2.INTER_LINEAR)
        return warped
    except Exception as e:
        print(f"Birds eye view error: {e}")
        return binary_mask

def apply_gabor_filter(image):
    try:
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        angles = np.arctan2(sobely, sobelx)
        nonzero_angles = angles[np.abs(angles) > 1e-3]
        theta = np.median(nonzero_angles) if nonzero_angles.size > 0 else 0
        kernel = cv2.getGaborKernel((21, 21), sigma=5, theta=theta, lambd=10, gamma=0.5, psi=0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
        return cv2.convertScaleAbs(filtered)
    except Exception as e:
        print(f"Gabor filter error: {e}")
        return image

# =================================================================================
# ===== 칼만 필터가 적용된 Sliding Window 함수 =====
# =================================================================================
def sliding_window_with_kalman(mask, left_kalman_manager, right_kalman_manager, 
                              right_nwindows=20, left_nwindows=20,
                              right_margin=60, left_margin=60, minpix=10):
    """
    칼만 필터가 적용된 히스토그램 기반 Sliding Window 차선 탐지
    """
    # 히스토그램 계산 전 마스크 검증
    if mask is None or mask.size == 0:
        vis_img = np.zeros((360, 640, 3), dtype=np.uint8)  # 기본 크기
        return vis_img, [], [], [], []
    
    # 히스토그램을 이용한 차선 기초 탐색
    base_limit_height = mask.shape[0] * 4 // 5
    histogram = np.sum(mask[base_limit_height:, :], axis=0)
    
    # 히스토그램이 비어있는 경우 처리
    if np.sum(histogram) == 0:
        vis_img = np.dstack((mask, mask, mask))
        return vis_img, [], [], [], []
    
    midpoint = histogram.shape[0] // 2
    
    # 각 영역에서 히스토그램 최대값 찾기 (안전하게)
    left_hist = histogram[:midpoint]
    right_hist = histogram[midpoint:]
    
    meas_left = np.argmax(left_hist) if np.sum(left_hist) > 0 else midpoint // 2
    meas_right = np.argmax(right_hist) + midpoint if np.sum(right_hist) > 0 else midpoint + midpoint // 2

    left_hist_sum = np.sum(left_hist)
    right_hist_sum = np.sum(right_hist)

    left_lane_missing = False
    right_lane_missing = False

    # 차선이 감지되지 않는 경우 예외 처리
    if left_hist_sum < LEFT_HIST_THRESHOLD:
        left_lane_missing = True
        meas_left = meas_right - MIN_LANE_SEPARATION
    if right_hist_sum < RIGHT_HIST_THRESHOLD:
        right_lane_missing = True
        meas_right = meas_left + MIN_LANE_SEPARATION

    if meas_right <= meas_left + MIN_LANE_SEPARATION:
        meas_right = meas_left + MIN_LANE_SEPARATION

    # 슬라이딩 윈도우를 이용한 차선 포인트 탐색
    leftx_base = meas_left
    rightx_base = meas_right

    left_window_height = int(mask.shape[0] // left_nwindows)
    right_window_height = int(mask.shape[0] // right_nwindows)
    leftx_current = leftx_base
    rightx_current = rightx_base

    nonzero = mask.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_points_x = []
    left_lane_points_y = []
    right_lane_points_x = []
    right_lane_points_y = []

    # 각 윈도우의 박스 좌표를 저장할 리스트
    left_window_rects = []
    right_window_rects = []

    # 슬라이딩 윈도우 - 칼만 필터 적용
    for window in range(max(right_nwindows, left_nwindows)):
        # 왼쪽 윈도우
        if not left_lane_missing and window < left_nwindows:
            win_y_low_left = mask.shape[0] - (window + 1) * left_window_height
            win_y_high_left = mask.shape[0] - window * left_window_height
            
            # 윈도우 x 범위를 이미지 경계 내로 제한
            win_xleft_low = max(0, leftx_current - left_margin)
            win_xleft_high = min(mask.shape[1], leftx_current + left_margin)
            
            # 유효한 윈도우인지 확인
            if win_xleft_high > win_xleft_low:
                left_window_rects.append((int(win_xleft_low), int(win_y_low_left), 
                                        int(win_xleft_high), int(win_y_high_left)))
                
                good_left_inds = ((nonzeroy >= win_y_low_left) & (nonzeroy < win_y_high_left) &
                                  (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                
                if len(good_left_inds) > minpix:
                    leftx_mean = int(np.mean(nonzerox[good_left_inds]))
                    lefty_mean = int(np.mean(nonzeroy[good_left_inds]))
                    # 칼만 필터로 x 좌표 업데이트
                    filtered_x = left_kalman_manager.update_window(window, leftx_mean)
                    if filtered_x is not None:
                        leftx_current = int(filtered_x)
                        left_lane_points_x.append(int(filtered_x))
                        left_lane_points_y.append(lefty_mean)
                    else:
                        # 칼만 필터가 실패하면 원래 값 사용
                        leftx_current = leftx_mean
                        left_lane_points_x.append(leftx_mean)
                        left_lane_points_y.append(lefty_mean)

        # 오른쪽 윈도우
        if not right_lane_missing and window < right_nwindows:
            win_y_low_right = mask.shape[0] - (window + 1) * right_window_height
            win_y_high_right = mask.shape[0] - window * right_window_height
            
            # 윈도우 x 범위를 이미지 경계 내로 제한
            win_xright_low = max(0, rightx_current - right_margin)
            win_xright_high = min(mask.shape[1], rightx_current + right_margin)
            
            # 유효한 윈도우인지 확인
            if win_xright_high > win_xright_low:
                right_window_rects.append((int(win_xright_low), int(win_y_low_right), 
                                         int(win_xright_high), int(win_y_high_right)))
                
                good_right_inds = ((nonzeroy >= win_y_low_right) & (nonzeroy < win_y_high_right) &
                                   (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                
                if len(good_right_inds) > minpix:
                    rightx_mean = int(np.mean(nonzerox[good_right_inds]))
                    righty_mean = int(np.mean(nonzeroy[good_right_inds]))
                    # 칼만 필터로 x 좌표 업데이트
                    filtered_x = right_kalman_manager.update_window(window, rightx_mean)
                    if filtered_x is not None:
                        rightx_current = int(filtered_x)
                        right_lane_points_x.append(int(filtered_x))
                        right_lane_points_y.append(righty_mean)
                    else:
                        # 칼만 필터가 실패하면 원래 값 사용
                        rightx_current = rightx_mean
                        right_lane_points_x.append(rightx_mean)
                        right_lane_points_y.append(righty_mean)

    # DBSCAN -> RANSAC 순서로 후처리
    left_lane_points_x, left_lane_points_y = filter_lane_points_dbscan(left_lane_points_x, left_lane_points_y)
    right_lane_points_x, right_lane_points_y = filter_lane_points_dbscan(right_lane_points_x, right_lane_points_y)

    left_lane_points_x, left_lane_points_y = ransac_filter_points(left_lane_points_x, left_lane_points_y)
    right_lane_points_x, right_lane_points_y = ransac_filter_points(right_lane_points_x, right_lane_points_y)

    # 다항식 보정 (곡선에 맞게 재배열)
    if len(left_lane_points_y) >= 3:
        try:
            poly_left = np.polyfit(left_lane_points_y, left_lane_points_x, 2)
            p_left = np.poly1d(poly_left)
            y_left_new = np.linspace(min(left_lane_points_y), max(left_lane_points_y), num=len(left_lane_points_y))
            left_lane_points_x = p_left(y_left_new).tolist()
            left_lane_points_y = y_left_new.tolist()
        except Exception as e:
            print(f"Left polyfit error: {e}")
        
    if len(right_lane_points_y) >= 3:
        try:
            poly_right = np.polyfit(right_lane_points_y, right_lane_points_x, 2)
            p_right = np.poly1d(poly_right)
            y_right_new = np.linspace(min(right_lane_points_y), max(right_lane_points_y), num=len(right_lane_points_y))
            right_lane_points_x = p_right(y_right_new).tolist()
            right_lane_points_y = y_right_new.tolist()
        except Exception as e:
            print(f"Right polyfit error: {e}")
    
    # 기울기 기반 차선 무시 로직
    def calculate_slope(x_points, y_points):
        if len(x_points) >= 2 and len(y_points) >= 2:
            try:
                fit = np.polyfit(y_points, x_points, 1)
                return fit[0]
            except:
                return None
        return None

    slope_threshold = 3.0
    left_slope = calculate_slope(left_lane_points_x, left_lane_points_y)
    right_slope = calculate_slope(right_lane_points_x, right_lane_points_y)
    
    if len(left_lane_points_x) > len(right_lane_points_x):
        if left_slope is not None and abs(left_slope) < slope_threshold:
            right_lane_points_x, right_lane_points_y = [], []
    elif len(right_lane_points_x) > len(left_lane_points_x):
        if right_slope is not None and abs(right_slope) < slope_threshold:
            left_lane_points_x, left_lane_points_y = [], []
    
    # 시각화용 이미지 (마스크 3채널)
    vis_img = np.dstack((mask, mask, mask))

    # 왼쪽 차선 시각화 - 안전한 처리
    if left_lane_points_x and left_lane_points_y and left_window_rects:
        for i in range(min(len(left_window_rects), len(left_lane_points_x))):
            x_low, y_low, x_high, y_high = left_window_rects[i]
            cv2.rectangle(vis_img, (x_low, y_low), (x_high, y_high), (255, 0, 0), 2)
            
    # 오른쪽 차선 시각화 - 안전한 처리
    if right_lane_points_x and right_lane_points_y and right_window_rects:
        for i in range(min(len(right_window_rects), len(right_lane_points_x))):
            x_low, y_low, x_high, y_high = right_window_rects[i]
            cv2.rectangle(vis_img, (x_low, y_low), (x_high, y_high), (0, 0, 255), 2)

    # 검출된 차선 포인트 시각화
    for x, y in zip(left_lane_points_x, left_lane_points_y):
        try:
            cv2.circle(vis_img, (int(x), int(y)), 5, (255, 0, 0), -1)
        except:
            pass
            
    for x, y in zip(right_lane_points_x, right_lane_points_y):
        try:
            cv2.circle(vis_img, (int(x), int(y)), 5, (0, 0, 255), -1)
        except:
            pass

    return vis_img, left_lane_points_x, left_lane_points_y, right_lane_points_x, right_lane_points_y

# --------------------- 인자 파싱 및 메인 함수 --------------------- #
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='/dev/video2', 
                       help='비디오 소스 (파일 경로 또는 웹캠 장치)')
    return parser.parse_args()

def main():
    rclpy.init()
    node = Node("traditional_lane_detection_node")
    publisher = node.create_publisher(LanePoints, "lane_points", 10)
    args = parse_args()
    
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: 비디오 소스를 열 수 없습니다. 경로 확인: {args.source}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
    # 칼만 필터 매니저 초기화
    left_kalman_manager = LaneKalmanManager(20)   # 왼쪽 차선용
    right_kalman_manager = LaneKalmanManager(20)  # 오른쪽 차선용
    
    # 프레임 카운터 초기화
    lane_missing_counter = 0

    while rclpy.ok():
        try:
            time.sleep(0.01)
            ret, frame = cap.read()
            if not ret:
                if 'dev' not in args.source:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("프레임을 읽을 수 없습니다. 스트림 종료")
                    break
            
            # 전처리 및 차선 검출
            lane_mask = preprocess_lane(frame)
            src_points = np.float32([[130, 360], [510, 360], [205, 240], [435, 240]])
            dst_points = np.float32([[130, 360], [510, 360], [130, 0],   [510, 0]])
            birdseye_mask = apply_birds_eye_view(lane_mask, src_points, dst_points)
            gabor_mask = apply_gabor_filter(birdseye_mask)
            enhanced_mask = cv2.bitwise_and(birdseye_mask, gabor_mask)
            
            sobelx = cv2.Sobel(enhanced_mask, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(enhanced_mask, cv2.CV_64F, 0, 1, ksize=3)
            gradient = cv2.magnitude(sobelx, sobely)
            
            if np.max(gradient) > 0:
                gradient = np.uint8(255 * gradient / np.max(gradient))
            else:
                gradient = np.zeros_like(gradient, dtype=np.uint8)
                
            _, edge_mask = cv2.threshold(gradient, 50, 255, cv2.THRESH_BINARY)
            inv_edge_mask = cv2.bitwise_not(edge_mask)
            final_mask = cv2.bitwise_and(enhanced_mask, inv_edge_mask)

            # 칼만 필터가 적용된 슬라이딩 윈도우 사용
            sw_img, left_pts_x, left_pts_y, right_pts_x, right_pts_y = sliding_window_with_kalman(
                final_mask, left_kalman_manager, right_kalman_manager)

            # 안전한 메시지 퍼블리싱
            lane_msg = LanePoints()
            lane_msg.left_x = [float(x) for x in left_pts_x] if left_pts_x else []
            lane_msg.left_y = [float(y) for y in left_pts_y] if left_pts_y else []
            lane_msg.right_x = [float(x) for x in right_pts_x] if right_pts_x else []
            lane_msg.right_y = [float(y) for y in right_pts_y] if right_pts_y else []
            publisher.publish(lane_msg)

            # 차선 감지 카운터 업데이트
            lane_detected = (len(left_pts_x) > 0 or len(right_pts_x) > 0)
            if lane_detected:
                lane_missing_counter = 0
            else:
                lane_missing_counter += 1

            # 화면 표시
            cv2.imshow("Original Frame", frame)
            cv2.imshow("Sliding Window with Kalman Filter", sw_img)
            
            # ESC 키로 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
        except Exception as e:
            print(f"Error in main loop: {e}")
            # 에러 발생 시 빈 메시지 퍼블리싱
            lane_msg = LanePoints()
            lane_msg.left_x = []
            lane_msg.left_y = []
            lane_msg.right_x = []
            lane_msg.right_y = []
            publisher.publish(lane_msg)
            continue

    cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
