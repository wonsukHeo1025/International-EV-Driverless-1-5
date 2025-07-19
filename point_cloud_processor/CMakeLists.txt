cmake_minimum_required(VERSION 3.8)
project(point_cloud_processor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)

find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(point_cloud_processor src/point_cloud_processor.cpp)
ament_target_dependencies(point_cloud_processor
  rclcpp sensor_msgs pcl_conversions pcl_ros
)
install(TARGETS point_cloud_processor DESTINATION lib/${PROJECT_NAME})

add_executable(point_cloud_cluster_node src/point_cloud_cluster_node.cpp)
ament_target_dependencies(point_cloud_cluster_node
  rclcpp sensor_msgs visualization_msgs geometry_msgs std_msgs
  pcl_conversions pcl_ros
)

 
# ① 실행 파일 추가
add_executable(collision_detector_node src/collision_detector_node.cpp)

# ② 의존성 연결
ament_target_dependencies(
  collision_detector_node
  rclcpp                 # 노드/타이머/로깅
  sensor_msgs            # TwistWithCovarianceStamped
  geometry_msgs          # Point 등
  std_msgs               # UInt8
  visualization_msgs     # Marker·MarkerArray
)

# ③ 설치 경로 지정
install(
  TARGETS collision_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS point_cloud_cluster_node
  DESTINATION lib/${PROJECT_NAME})
  
 # launch 디렉터리 설치
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
