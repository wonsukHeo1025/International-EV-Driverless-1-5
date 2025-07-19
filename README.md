# 🏎️ 2025 국제 대학생 EV 자율주행 대회 1/5부문

<img width="1186" height="618" alt="image" src="https://github.com/user-attachments/assets/60ba2ad2-92d8-4afd-bfaf-c74f87046e13" />

본 프로젝트는 **ROS 2 환경**에서 동작하는 자율주행 시스템으로, **계층적 제어 구조**를 통해 복잡한 주행 환경에 대응하는 것을 목표로 한다.

핵심 아키텍처는 [미션 계획] -> [경로 생성 및 회피] -> [최종 제어 선택]의 3단계로 구성된다. 이를 기반으로 커스텀 YOLOv8 모델을 이용한 차선 인식, LiDAR 포인트 클라우드를 활용한 장애물 탐지, 그리고 MPC를 이용한 GPS 웨이포인트 추종 및 장애물 회피 기능을 통합적으로 구현하였다.

---

## 프로젝트 주요 기능

- **Lane Detection**: 커스텀 YOLOv8 모델과 컴퓨터 비전(Bird's Eye View, DBSCAN, RANSAC, Sliding Window)을 결합하여 차선을 탐지하고, 차선 유지 주행(lane_steer_angle)을 위한 제어 값 생성
- **Object Detection**: Velodyne LiDAR의 3D 포인트 클라우드를 실시간으로 클러스터링, 주행 경로 상의 장애물을 탐지
- **High-Precision Localization**: RTK-GPS와 NTRIP 클라이언트를 통해 차량의 현재 위치와 자세(YAW) 파악
- **GPS Path Planning & Obstacle Avoidance**: GPS 웨이포인트로 생성된 전역 경로 /gps_path를 기반으로 MPC 컨트롤러가 장애물 유무를 반영하여 최적의 회피 경로와 조향각 /mpc_steer_angle을 계산
- **Hierarchical Control Selection**: selector_mpc.py가 정의된 우선순위에 따라 최종적으로 실행할 단 하나의 명령을 선택하여 실제 차량(Arduino)으로 전달

---

## 프로젝트 디렉터리 구조

```
 jeju_ws/src/
 ├── control/
 │   ├── arduino_control.py
 │   ├── selector_mpc.py
 │   └── gps/
 │       ├── gps_course_publisher.py
 │       ├── mission_control.py
 │       ├── mpc.py
 │       ├── 역.csv
 │       └── 정.csv
 ├── lane/
 │   ├── CMakeLists.txt
 │   ├── package.xml
 │   ├── setup.py
 │   ├── data/
 │   │   └── weights/
 │   │       └── best.pt
 │   └── lane/
 │       ├── highcontrol.py
 │       ├── lane_custom.py
 │       ├── lane_traditional.py
 │       └── path.py
 ├── my_lane_msgs/
 │   ├── CMakeLists.txt
 │   ├── package.xml
 │   └── msg/
 │       └── LanePoints.msg
 ├── point_cloud_processor/
 │   ├── CMakeLists.txt
 │   ├── package.xml
 │   ├── include/
 │   ├── launch/
 │   └── src/
 │       ├── collision_detector_node.cpp
 │       ├── point_cloud_cluster_node.cpp
 │       └── point_cloud_processor.cpp
 ├── RTK_GPS_NTRIP/
 │   ├── fix2nmea/
 │   ├── ntrip_client/
 │   ├── rtcm_msgs/
 │   ├── ublox/
 │   ├── ublox_gps/
 │   ├── ublox_msgs/
 │   └── ublox_serialization/
 └── velodyne/
     ├── velodyne/
     ├── velodyne_driver/
     ├── velodyne_laserscan/
     ├── velodyne_msgs/
     └── velodyne_pointcloud/


```

---

## 핵심 기능 상세 설명

### 1️**차선 탐지 (Lane Detection)**

- 위치: lane/
- 커스텀 YOLOv8 모델을 통해 차선 세그멘테이션 후 Bird's Eye View 변환
- DBSCAN 클러스터링 및 RANSAC 알고리즘으로 이상치 제거
- Sliding Window 알고리즘을 통한 차선 픽셀 추출
- highcontrol.py에서 조향각 /lane_steer_angle 생성

### 2️**GPS 기반 경로 생성 (GPS Course Generation)**

- 위치: control/gps/gps_course_publisher.py, control/gps/mission_control.py
- csv 파일에 있는 모든 GPS 웨이포인트 목록을 /gps_path로 퍼블리시
- 차량의 현재 위치를 계속해서 확인하며 목표 속도를 결정

### 3️**MPC(Model Predictive Control)**

- 위치: control/gps/mpc.py
- 전체 경로 /gps_path와 현재 내 위치 /current_pose를 비교하여 경로를 따라가기 위한 조향각 계산
- 만약 장애물 /obstacles이 나타나면, MPC 알고리즘을 통해 회피하며 /gps_path를 따라 주행
- 최종적으로 조향각 /mpc_steer_angle 생성

### 4️**미션 및 제어기 선택 (Mission & Controller Selector)**

- 위치: control/selector_mpc.py
- 우선순위에 따라 어떤 제어 신호를 사용할지 선택
- 차선 주행 미션일 경우 /lane_steer_angle 선택
- GPS 경로 주행 미션일 경우 /mpc_steer_angle 선택
- 선택된 조향각과 mission_control이 보낸 목표 속도를 조합하여 최종 제어 명령 생성

---

## 결과 시각화
![2](https://github.com/user-attachments/assets/082ee031-ace0-4c5d-bb3e-634fe317030e)
