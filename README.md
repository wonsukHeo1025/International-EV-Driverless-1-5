# 🏎️ 2025 국제 대학생 EV 자율주행 대회 1/5부문

## 역할: Lane Detection + Vehicle Control
![1](https://github.com/user-attachments/assets/28ef12f2-57ae-4d83-8e9d-b28233c6ba53)

본 프로젝트는 **ROS 2 환경**에서 자율주행 차량의 **차선 인식 및 경로 계획, 차량 제어**를 목적으로 개발되었다. 

딥러닝 기반의 **YOLOPv2 모델**과 이미지 처리 기법을 결합한 방식이다.

---

## 프로젝트 주요 기능

- **Lane Detection**: YOLOPv2와 다양한 이미지 처리 기법을 이용한 정밀한 차선 검출
- **Path Planning**: 실시간 차선 데이터를 기반으로 부드럽고 안정적인 주행 경로 생성
- **Vehicle Control**: Pure Pursuit 알고리즘 기반 차량 제어 (스티어링 각도 및 속도)

---

## 프로젝트 디렉터리 구조

```
jeju_ws
└── src
    ├── jeju
    │   ├── CMakeLists.txt
    │   ├── data
    │   │   └── weights
    │   │       ├── model.txt
    │   │       └── yolopv2.pt
    │   ├── jeju
    │   │   ├── control.py
    │   │   ├── highcontrol.py
    │   │   ├── __init__.py
    │   │   ├── lane_node.py
    │   │   ├── lane.py
    │   │   ├── path.py
    │   │   └── utils
    │   │       ├── __init__.py
    │   │       └── utils.py
    │   ├── LICENSE
    │   ├── package.xml
    │   ├── README.md
    │   ├── requirements.txt
    │   ├── resource
    │   │   └── jeju
    │   ├── setup.cfg
    │   ├── setup.py
    │   └── utils
    │       ├── __init__.py
    │       └── utils.py
    └── my_lane_msgs
        ├── CMakeLists.txt
        ├── msg
        │   └── LanePoints.msg
        └── package.xml

```

---

## ⚙모델 설치

YOLOPv2 모델을 다운로드하고 아래 경로에 저장:
```
your_ws/src/jeju/data/weights/yolopv2.pt
```

- [YOLOPv2 모델 다운로드](https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt)

---

## 핵심 기능 상세 설명

### 1️**차선 탐지 (lane.py)**

- YOLOPv2로 lane mask 생성
- Bird’s Eye View 변환, Sobel 필터, Morphology 필터 적용
- Sliding Window 알고리즘을 통한 차선 픽셀 추출
- DBSCAN 클러스터링 및 RANSAC 알고리즘을 이용한 이상치 제거
- 결과를 ROS 토픽으로 발행 (`LanePoints`)

### 2️**경로 계획 (path.py)**

- 수신한 차선 포인트를 기반으로 최적 경로 계산
- 좌우 차선 존재 여부에 따라 동적 경로 생성
- Cubic Spline 보간을 통한 부드러운 경로 생성
- 최종 경로를 ROS 토픽으로 발행 (`Float32MultiArray`)

### 3️**차량 제어 (highcontrol.py)**

- Pure Pursuit 알고리즘 기반 스티어링 각도 및 속도 계산
- 차량 상태 및 목표점에 따라 실시간 제어 명령 생성
- 스티어링 각도와 속도를 ROS 토픽으로 발행 (`Float32`)

### 4️**제어 명령 처리 (control.py)**

- ROS 토픽에서 차량 제어 명령을 수신 및 로그 출력
- Arduino 기반 차량 하드웨어와의 통신을 위한 기반 코드 제공

---

## 결과 시각화
![2](https://github.com/user-attachments/assets/082ee031-ace0-4c5d-bb3e-634fe317030e)

OpenCV를 통해 실시간으로 결과 확인:
- 차선 탐지 및 경로 생성
- Pure Pursuit 기반 제어 상황

---
