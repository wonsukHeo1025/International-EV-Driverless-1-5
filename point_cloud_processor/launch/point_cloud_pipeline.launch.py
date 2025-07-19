#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
두 노드 + 후처리 노드들을 한꺼번에 실행 & 파라미터 튜닝용 런치
  1) point_cloud_processor        : ROI·업샘플링·지면제거 등
  2) point_cloud_cluster_node     : 유클리디안 클러스터링
  4) collision_detector_node      : 클러스터 결과로 충돌 감지

예시)  
  ros2 launch point_cloud_processor point_cloud_pipeline.launch.py \
       min_range:=0.5 max_range:=12.0 use_upsampling:=true cluster_tolerance:=0.06
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node


def declare(name, default, desc):
    """ DeclareLaunchArgument 헬퍼 """
    return DeclareLaunchArgument(
        name,
        default_value=TextSubstitution(text=str(default)),
        description=desc,
    )


def generate_launch_description() -> LaunchDescription:
    # ─────────────― 공통 인자 선언 ─────────────―
    ld_args = [
        # point_cloud_processor
        declare("min_angle", -100.0, "ROI 최소 각도 [deg]"),
        declare("max_angle", 100.0, "ROI 최대 각도 [deg]"),
        declare("min_range", 0.2, "ROI 최소 거리 [m]"),
        declare("max_range", 10.0, "ROI 최대 거리 [m]"),
        declare("min_y", -2.0, "ROI 왼쪽 한계 [m]"),
        declare("max_y", 2.0, "ROI 오른쪽 한계 [m]"),
        declare("use_width_ratio", False, "폭 비율 사용 여부"),
        declare("width_ratio", 0.5, "폭 비율 값"),
        declare("use_upsampling", False, "업샘플링 사용 여부"),
        declare("upsample_factor", 2, "업샘플링 반복 수"),
        declare("knn_neighbors", 5, "KNN 개수"),
        declare("use_ground_removal", True, "지면 제거 사용 여부"),
        # point_cloud_cluster_node - Euclidean Clustering 파라미터
        declare("cluster_tolerance", 0.05, "유클리디안 클러스터링 거리 임계값 [m]"),
        declare("min_cluster_size", 5, "최소 클러스터 크기"),
        declare("max_cluster_size", 25000, "최대 클러스터 크기"),
        declare("merge_threshold", 0.3, "클러스터 병합 거리 [m]"),
        declare("marker_scale", 0.3, "Marker 점 크기 [m]"),
    ]

    # ─────────────― 노드 정의 ─────────────―
    processor = Node(
        package="point_cloud_processor",
        executable="point_cloud_processor",
        name="point_cloud_processor",
        parameters=[{
            "min_angle": LaunchConfiguration("min_angle"),
            "max_angle": LaunchConfiguration("max_angle"),
            "min_range": LaunchConfiguration("min_range"),
            "max_range": LaunchConfiguration("max_range"),
            "min_y": LaunchConfiguration("min_y"),
            "max_y": LaunchConfiguration("max_y"),
            "use_width_ratio": LaunchConfiguration("use_width_ratio"),
            "width_ratio": LaunchConfiguration("width_ratio"),
            "use_upsampling": LaunchConfiguration("use_upsampling"),
            "upsample_factor": LaunchConfiguration("upsample_factor"),
            "knn_neighbors": LaunchConfiguration("knn_neighbors"),
            "use_ground_removal": LaunchConfiguration("use_ground_removal"),
        }],
        output="screen",
    )

    cluster = Node(
        package="point_cloud_processor",
        executable="point_cloud_cluster_node",
        name="point_cloud_cluster_node",
        parameters=[{
            "cluster_tolerance": LaunchConfiguration("cluster_tolerance"),
            "min_cluster_size": LaunchConfiguration("min_cluster_size"),
            "max_cluster_size": LaunchConfiguration("max_cluster_size"),
            "merge_threshold": LaunchConfiguration("merge_threshold"),
            "marker_scale": LaunchConfiguration("marker_scale"),
        }],
        output="screen",
    )

    collision = Node(
        package="point_cloud_processor",
        executable="collision_detector_node",
        name="collision_detector_node",
        output="screen",
    )

    return LaunchDescription(ld_args + [
        processor,
        cluster,
        collision,
    ])
