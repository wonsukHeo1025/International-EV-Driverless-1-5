#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, UInt8
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from my_lane_msgs.msg import LanePoints

from ultralytics import YOLO
import torch

# CUDA 사용 가능 여부 확인 및 디바이스 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
current_device = device  # 실제 사용 중인 디바이스 추적
print(f"Using device: {device}")
if device == 'cuda':
    try:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # CUDA 메모리 캐시 정리
        torch.cuda.empty_cache()
        # CUDA 메모리 할당 설정
        torch.cuda.set_per_process_memory_fraction(0.8)  # 전체 메모리의 80%만 사용
    except Exception as e:
        print(f"CUDA initialization error: {e}")
        device = 'cpu'
        current_device = 'cpu'
        print("Falling back to CPU")

# YOLO 모델을 디바이스에 로드
try:
    model = YOLO('/home/kai/Desktop/day1/src/lane/data/weights/best.pt')
    model.to(current_device)
    # 모델을 평가 모드로 설정 (메모리 효율성)
    model.eval()
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    if current_device == 'cuda':
        try:
            torch.cuda.empty_cache()
            current_device = 'cpu'
            print("Retrying with CPU...")
            model = YOLO('/home/kai/Desktop/day1/src/lane/data/weights/best.pt')
            model.to(current_device)
            model.eval()
        except:
            print("Model loading failed completely")
            exit(1)
    else:
        exit(1)

# --------------------- 상수 설정 --------------------- #
LEFT_HIST_THRESHOLD = 1200
RIGHT_HIST_THRESHOLD = 1200
MODE_SWITCH_THRESHOLD = 10  # 모드 전환을 위한 프레임 수 (기존 NO_LANE_THRESHOLD를 대체)

# --------------------- 전역 변수 --------------------- #
# BOX_THRESHOLD는 더 이상 사용하지 않음

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
        try:
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
        except Exception as e:
            print(f"Kalman filter error: {e}")
            self.initialized = False
            return measurement

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
    try:
        if len(points_y) < 5: 
            return points_x, points_y
        X = np.array(points_y).reshape(-1, 1)
        y = np.array(points_x)
        model = make_pipeline(PolynomialFeatures(degree), RANSACRegressor(residual_threshold=25, max_trials=500))
        model.fit(X, y)
        inlier_mask = model.named_steps['ransacregressor'].inlier_mask_
        return y[inlier_mask].tolist(), X[inlier_mask].flatten().tolist()
    except Exception as e:
        print(f"RANSAC filter error: {e}")
        return points_x, points_y

def filter_lane_points_dbscan(points_x, points_y, eps=15, base_min_samples=4):
    try:
        if len(points_x) == 0: 
            return points_x, points_y
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
        print(f"DBSCAN filter error: {e}")
        return points_x, points_y

def apply_birds_eye_view(binary_mask, src_points, dst_points):
    try:
        h, w = binary_mask.shape[:2]
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(binary_mask, M, (w, h), flags=cv2.INTER_LINEAR)
        return warped
    except Exception as e:
        print(f"BEV transform error: {e}")
        return binary_mask

# =================================================================================
# ===== 칼만 필터가 적용된 Sliding Window 함수 =====
# =================================================================================
def sliding_window_with_kalman(mask, left_kalman_manager, right_kalman_manager, 
                              right_nwindows=20, left_nwindows=20,
                              right_margin=90, left_margin=90, minpix=20):
    """
    칼만 필터가 적용된 히스토그램 기반 Sliding Window 차선 탐지
    """
    try:
        # 마스크가 비어있는지 확인
        if mask is None or mask.size == 0:
            return np.zeros_like(mask), [], [], [], []
        
        # 히스토그램을 이용한 차선 기초 탐색
        base_limit_height = mask.shape[0] * 4 // 5
        histogram = np.sum(mask[base_limit_height:, :], axis=0)
        
        # 히스토그램이 모두 0인 경우 처리
        if np.sum(histogram) == 0:
            return np.dstack((mask, mask, mask)), [], [], [], []
        
        midpoint = histogram.shape[0] // 2
        
        # 왼쪽과 오른쪽 히스토그램 분석
        left_hist_sum = np.sum(histogram[:midpoint])
        right_hist_sum = np.sum(histogram[midpoint:])
        
        # 차선 존재 여부 판단
        left_lane_detected = left_hist_sum >= LEFT_HIST_THRESHOLD
        right_lane_detected = right_hist_sum >= RIGHT_HIST_THRESHOLD
        
        # 초기 위치 설정 (검출된 차선만)
        leftx_base = None
        rightx_base = None
        
        if left_lane_detected:
            leftx_base = np.argmax(histogram[:midpoint])
        
        if right_lane_detected:
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # 슬라이딩 윈도우를 위한 설정
        left_window_height = int(mask.shape[0] // left_nwindows)
        right_window_height = int(mask.shape[0] // right_nwindows)
        
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

        # 현재 x 위치 초기화
        leftx_current = leftx_base
        rightx_current = rightx_base

        # 슬라이딩 윈도우 - 칼만 필터 적용
        for window in range(max(right_nwindows, left_nwindows)):
            # 왼쪽 윈도우 (왼쪽 차선이 검출된 경우만)
            if left_lane_detected and window < left_nwindows and leftx_current is not None:
                win_y_low_left = mask.shape[0] - (window + 1) * left_window_height
                win_y_high_left = mask.shape[0] - window * left_window_height
                win_xleft_low = max(0, leftx_current - left_margin)
                win_xleft_high = min(mask.shape[1], leftx_current + left_margin)
                left_window_rects.append((int(win_xleft_low), int(win_y_low_left), int(win_xleft_high), int(win_y_high_left)))
                
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

            # 오른쪽 윈도우 (오른쪽 차선이 검출된 경우만)
            if right_lane_detected and window < right_nwindows and rightx_current is not None:
                win_y_low_right = mask.shape[0] - (window + 1) * right_window_height
                win_y_high_right = mask.shape[0] - window * right_window_height
                win_xright_low = max(0, rightx_current - right_margin)
                win_xright_high = min(mask.shape[1], rightx_current + right_margin)
                right_window_rects.append((int(win_xright_low), int(win_y_low_right), int(win_xright_high), int(win_y_high_right)))
                
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

        # RANSAC 필터링
        left_lane_points_x, left_lane_points_y = ransac_filter_points(left_lane_points_x, left_lane_points_y)
        right_lane_points_x, right_lane_points_y = ransac_filter_points(right_lane_points_x, right_lane_points_y)

        # 다항식 보정 (곡선에 맞게 재배열)
        try:
            if len(left_lane_points_y) >= 4:
                poly_left = np.polyfit(left_lane_points_y, left_lane_points_x, 3)
                p_left = np.poly1d(poly_left)
                y_left_new = np.linspace(min(left_lane_points_y), max(left_lane_points_y), num=len(left_lane_points_y))
                left_lane_points_x = p_left(y_left_new).tolist()
                left_lane_points_y = y_left_new.tolist()
        except Exception as e:
            print(f"Left lane polyfit error: {e}")
            
        try:
            if len(right_lane_points_y) >= 4:
                poly_right = np.polyfit(right_lane_points_y, right_lane_points_x, 3)
                p_right = np.poly1d(poly_right)
                y_right_new = np.linspace(min(right_lane_points_y), max(right_lane_points_y), num=len(right_lane_points_y))
                right_lane_points_x = p_right(y_right_new).tolist()
                right_lane_points_y = y_right_new.tolist()
        except Exception as e:
            print(f"Right lane polyfit error: {e}")
        
        # 기울기 기반 차선 무시 로직
        def calculate_slope(x_points, y_points):
            try:
                if len(x_points) >= 2:
                    fit = np.polyfit(y_points, x_points, 1)
                    return fit[0]
            except:
                pass
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

        # 슬라이딩 윈도우 박스 시각화 (차선 픽셀이 검출된 경우만)
        # 왼쪽 차선: 실제로 검출된 포인트가 있는 윈도우만 표시
        if len(left_lane_points_x) > 0:
            for (x_low, y_low, x_high, y_high) in left_window_rects:
                # 해당 윈도우 영역에 실제 차선 포인트가 있는지 확인
                points_in_window = False
                for px, py in zip(left_lane_points_x, left_lane_points_y):
                    if y_low <= py < y_high:
                        points_in_window = True
                        break
                if points_in_window:
                    cv2.rectangle(vis_img, (x_low, y_low), (x_high, y_high), (255, 0, 0), 2)  # 왼쪽: 파란색
        
        # 오른쪽 차선: 실제로 검출된 포인트가 있는 윈도우만 표시
        if len(right_lane_points_x) > 0:
            for (x_low, y_low, x_high, y_high) in right_window_rects:
                # 해당 윈도우 영역에 실제 차선 포인트가 있는지 확인
                points_in_window = False
                for px, py in zip(right_lane_points_x, right_lane_points_y):
                    if y_low <= py < y_high:
                        points_in_window = True
                        break
                if points_in_window:
                    cv2.rectangle(vis_img, (x_low, y_low), (x_high, y_high), (0, 0, 255), 2)  # 오른쪽: 빨간색

        # 검출된 차선 포인트 시각화
        for x, y in zip(left_lane_points_x, left_lane_points_y):
            cv2.circle(vis_img, (int(x), int(y)), 5, (255, 0, 0), -1)  # 왼쪽: 파란색
        for x, y in zip(right_lane_points_x, right_lane_points_y):
            cv2.circle(vis_img, (int(x), int(y)), 5, (0, 0, 255), -1)  # 오른쪽: 빨간색

        return vis_img, left_lane_points_x, left_lane_points_y, right_lane_points_x, right_lane_points_y
    
    except Exception as e:
        print(f"Sliding window error: {e}")
        return np.dstack((mask, mask, mask)), [], [], [], []

# --------------------- 인자 파싱 및 메인 함수 --------------------- #
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='/dev/video2', 
                        help='비디오 소스 (파일 경로 또는 웹캠 장치)')
    return parser.parse_args()

def main():
    try:
        rclpy.init()
        node = Node("CUSTOM_LANE_DETECTION_NODE")
        
        publisher   = node.create_publisher(LanePoints, "lane_points", 10)
        detect_pub  = node.create_publisher(UInt8,      "/lane_detected", 10)

        args = parse_args()
        cap  = cv2.VideoCapture(args.source)
        
        # 카메라 연결 확인
        if not cap.isOpened():
            print(f"Failed to open video source: {args.source}")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        left_kalman_manager  = LaneKalmanManager(20)
        right_kalman_manager = LaneKalmanManager(20)
        
        # 모드 전환을 위한 카운터 변수
        lane_detected_count = 0
        lane_not_detected_count = 0
        current_mode = 0  # 0: GPS 모드, 1: 차선 모드
        
        # 프레임 레이트 관리
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # 기본값
        frame_delay = 1.0 / fps

        last_time = time.time()
        frame_count = 0
        cuda_cleanup_interval = 100  # 100프레임마다 CUDA 메모리 정리
        cuda_retry_interval = 500   # 500프레임마다 CUDA 복귀 시도
        cuda_fail_count = 0         # 연속 CUDA 실패 횟수

        while rclpy.ok():
            try:
                # 프레임 레이트 제어
                current_time = time.time()
                if current_time - last_time < frame_delay:
                    time.sleep(frame_delay - (current_time - last_time))
                last_time = time.time()
                
                frame_count += 1
                
                # 주기적인 CUDA 메모리 정리
                if current_device == 'cuda' and frame_count % cuda_cleanup_interval == 0:
                    try:
                        torch.cuda.empty_cache()
                        # 메모리 사용량 체크
                        if torch.cuda.is_available():
                            memory_allocated = torch.cuda.memory_allocated() / 1024**3
                            memory_reserved = torch.cuda.memory_reserved() / 1024**3
                            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                            if memory_allocated > 0.9 * total_memory:
                                print(f"Warning: High CUDA memory usage: {memory_allocated:.2f}GB allocated")
                    except Exception as e:
                        print(f"CUDA memory cleanup error: {e}")
                
                # CPU 모드에서 주기적으로 CUDA 복귀 시도
                if current_device == 'cpu' and device == 'cuda' and frame_count % cuda_retry_interval == 0:
                    try:
                        print("Attempting to return to CUDA...")
                        torch.cuda.empty_cache()
                        # 메모리 상태 확인
                        if torch.cuda.is_available():
                            memory_allocated = torch.cuda.memory_allocated() / 1024**3
                            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                            if memory_allocated < 0.5 * total_memory:  # 메모리가 50% 이하일 때만
                                model.to('cuda')
                                current_device = 'cuda'
                                cuda_fail_count = 0
                                print("Successfully returned to CUDA!")
                            else:
                                print(f"CUDA memory still high: {memory_allocated:.2f}GB/{total_memory:.2f}GB")
                    except Exception as e:
                        print(f"Failed to return to CUDA: {e}")
                
                ret, frame = cap.read()
                if not ret:
                    if "dev" not in args.source:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("Camera disconnected or end of video")
                        break

                # 프레임 크기 검증
                if frame is None or frame.size == 0:
                    print("Empty frame received")
                    continue

                # 1) YOLO 추론
                try:
                    # CUDA OOM 에러 처리를 위한 추가 try-except
                    if current_device == 'cuda':
                        try:
                            results = model(frame, verbose=False, device=current_device)
                        except torch.cuda.OutOfMemoryError:
                            print("CUDA out of memory! Switching to CPU mode...")
                            torch.cuda.empty_cache()
                            model.to('cpu')
                            current_device = 'cpu'
                            cuda_fail_count += 1
                            results = model(frame, verbose=False, device=current_device)
                    else:
                        results = model(frame, verbose=False, device=current_device)
                    
                    if not results or len(results) == 0:
                        raise Exception("No results from YOLO")
                    results = results[0]
                    
                    # CUDA 실패가 3번 이상이면 더 이상 복귀 시도하지 않음
                    if cuda_fail_count >= 3 and device == 'cuda':
                        device = 'cpu'  # 영구적으로 CPU 모드로
                        print("Too many CUDA failures. Staying in CPU mode.")
                        
                except torch.cuda.OutOfMemoryError as e:
                    print(f"CUDA OOM error: {e}")
                    # CUDA 메모리 부족 시 영구적으로 CPU로 전환
                    if current_device == 'cuda':
                        torch.cuda.empty_cache()
                        model.to('cpu')
                        current_device = 'cpu'
                        cuda_fail_count += 1
                        try:
                            results = model(frame, verbose=False, device='cpu')
                            if not results or len(results) == 0:
                                raise Exception("No results from YOLO on CPU")
                            results = results[0]
                        except Exception as cpu_e:
                            print(f"CPU fallback also failed: {cpu_e}")
                            raise
                except Exception as e:
                    print(f"YOLO inference error: {e}")
                    # YOLO 실패 시 GPS 모드로
                    lane_not_detected_count += 1
                    lane_detected_count = 0
                    
                    if lane_not_detected_count >= MODE_SWITCH_THRESHOLD and current_mode == 1:
                        current_mode = 0
                        print(f"[Mode Switch] Lane -> GPS Mode (YOLO error)")
                    
                    detect_pub.publish(UInt8(data=current_mode))
                    
                    # 빈 차선 포인트 퍼블리시
                    lane_msg = LanePoints(
                        left_x  = [],
                        left_y  = [],
                        right_x = [],
                        right_y = [],
                    )
                    publisher.publish(lane_msg)
                    continue

                # 2) masks가 None인지 체크
                if results.masks is None or results.masks.data is None:
                    # 마스크가 없으면 차선이 감지되지 않은 것으로 처리
                    lane_not_detected_count += 1
                    lane_detected_count = 0
                    
                    # 10프레임 이상 연속으로 차선이 감지되지 않으면 GPS 모드로 전환
                    if lane_not_detected_count >= MODE_SWITCH_THRESHOLD and current_mode == 1:
                        current_mode = 0
                        print(f"[Mode Switch] Lane -> GPS Mode (not detected for {lane_not_detected_count} frames)")
                    
                    detect_pub.publish(UInt8(data=current_mode))
                    
                    # 빈 차선 포인트 퍼블리시
                    lane_msg = LanePoints(
                        left_x  = [],
                        left_y  = [],
                        right_x = [],
                        right_y = [],
                    )
                    publisher.publish(lane_msg)
                    
                    # 시각화 (빈 화면에 모드 표시)
                    empty_img = np.zeros_like(frame)
                    cv2.putText(empty_img, f"Mode: {'LANE' if current_mode == 1 else 'GPS'}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    vis = cv2.hconcat([
                        cv2.resize(frame, (640,360)),
                        cv2.resize(empty_img, (640,360))
                    ])
                    cv2.imshow("Original & Kalman Result", vis)
                    
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue

                # 3) 세그멘테이션 마스크 생성 & ROI 제거 & BEV 변환
                try:
                    masks_np = results.masks.data.cpu().numpy()
                    # 메모리 효율적인 마스크 처리
                    if current_device == 'cuda':
                        # GPU 텐서를 즉시 CPU로 이동하여 GPU 메모리 확보
                        results.masks.data = results.masks.data.cpu()
                        torch.cuda.empty_cache()
                    
                    combined = np.any(masks_np, axis=0).astype(np.uint8) * 255
                    combined[:120, :] = 0
                    src = np.float32([[130,360],[510,360],[205,240],[435,240]])
                    dst = np.float32([[130,360],[510,360],[130,0],  [510,0]])
                    final_mask = apply_birds_eye_view(combined, src, dst)
                except Exception as e:
                    print(f"Mask processing error: {e}")
                    continue

                # 4) 하단 10px 검사
                h, w = final_mask.shape
                bottom_strip = final_mask[h-10:h, :]
                bottom_has_mask = cv2.countNonZero(bottom_strip) > 0
                
                if bottom_has_mask:
                    # → 유효, 슬라이딩윈도우+칼만 수행
                    sw_img, lx, ly, rx, ry = sliding_window_with_kalman(
                        final_mask, left_kalman_manager, right_kalman_manager
                    )
                    
                    # 차선 감지 여부 판단 (차선 포인트 개수만 기준)
                    total_lane_points = len(lx) + len(rx)
                    lane_detected = total_lane_points >= 10
                    
                    # 모드 전환 로직
                    if lane_detected:
                        lane_detected_count += 1
                        lane_not_detected_count = 0
                        
                        # 10프레임 이상 연속으로 차선이 감지되면 차선 모드로 전환
                        if lane_detected_count >= MODE_SWITCH_THRESHOLD and current_mode == 0:
                            current_mode = 1
                            print(f"[Mode Switch] GPS -> Lane Mode (detected for {lane_detected_count} frames)")
                    else:
                        lane_not_detected_count += 1
                        lane_detected_count = 0
                        
                        # 10프레임 이상 연속으로 차선이 감지되지 않으면 GPS 모드로 전환
                        if lane_not_detected_count >= MODE_SWITCH_THRESHOLD and current_mode == 1:
                            current_mode = 0
                            print(f"[Mode Switch] Lane -> GPS Mode (not detected for {lane_not_detected_count} frames)")
                    
                    detect_pub.publish(UInt8(data=current_mode))

                    # → 차선 포인트 퍼블리시
                    lane_msg = LanePoints(
                        left_x  =[float(x) for x in lx],
                        left_y  =[float(y) for y in ly],
                        right_x =[float(x) for x in rx],
                        right_y =[float(y) for y in ry],
                    )
                    publisher.publish(lane_msg)

                    # 모드 정보를 시각화에 추가
                    cv2.putText(sw_img, f"Mode: {'LANE' if current_mode == 1 else 'GPS'}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if current_mode == 1 else (0, 0, 255), 2)
                    cv2.putText(sw_img, f"Lane Points: {total_lane_points} (L:{len(lx)}, R:{len(rx)})", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    sw_show = sw_img

                else:
                    # → 하단에 마스크 없으면 차선 미감지로 처리
                    lane_not_detected_count += 1
                    lane_detected_count = 0
                    
                    # 10프레임 이상 연속으로 차선이 감지되지 않으면 GPS 모드로 전환
                    if lane_not_detected_count >= MODE_SWITCH_THRESHOLD and current_mode == 1:
                        current_mode = 0
                        print(f"[Mode Switch] Lane -> GPS Mode (not detected for {lane_not_detected_count} frames)")
                    
                    detect_pub.publish(UInt8(data=current_mode))
                    # 빈 차선 포인트 퍼블리시
                    lane_msg = LanePoints(
                        left_x  = [],
                        left_y  = [],
                        right_x = [],
                        right_y = [],
                    )
                    publisher.publish(lane_msg)
                    
                    sw_show = np.zeros_like(final_mask)
                    if len(sw_show.shape) == 2:
                        sw_show = cv2.cvtColor(sw_show, cv2.COLOR_GRAY2BGR)
                    cv2.putText(sw_show, f"Mode: {'LANE' if current_mode == 1 else 'GPS'}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if current_mode == 1 else (0, 0, 255), 2)

                # 5) 시각화
                vis = cv2.hconcat([
                    cv2.resize(frame,    (640,360)),
                    cv2.resize(sw_show,  (640,360))
                ])
                cv2.imshow("Original & Kalman Result", vis)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
                    
            except Exception as e:
                print(f"Main loop error: {e}")
                # 에러 발생 시에도 빈 데이터 퍼블리시
                try:
                    detect_pub.publish(UInt8(data=current_mode))
                    lane_msg = LanePoints(
                        left_x  = [],
                        left_y  = [],
                        right_x = [],
                        right_y = [],
                    )
                    publisher.publish(lane_msg)
                except:
                    pass
                continue

    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        try:
            cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()
