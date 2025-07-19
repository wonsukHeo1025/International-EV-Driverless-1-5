/*  collision_detector_node.cpp
 *  클러스터 마커(/cluster_markers) + 속도(/fix_velocity)로
 *  충돌 플래그(/collision_flag)·디버그 마커(/collision_debug_markers)를 퍼블리시
 *
 *  수정: Green zone (flag=3) 제거, 0/1/2만 사용
 *  작성 : ChatGPT-o3  (2025-07-10)
 *  빌드 : colcon build  (C++17 필요)
 */

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <geometry_msgs/msg/point.hpp>

#include <array>
#include <vector>
#include <cmath>
#include <utility>

using namespace std::chrono_literals;

/* ───────────────── Zone 구조 ───────────────── */
struct ZoneParam
{
  float front_ext, side_ext, rear_ext, front_radius;
  float color[3];
};

/* ───────────────── 노드 ───────────────── */
class CollisionDetectorNode : public rclcpp::Node
{
public:
  CollisionDetectorNode()
  : Node("collision_detector_node")
  {
    /* Sub/Pub ------------------------------------------------------- */
    sub_markers_ = create_subscription<visualization_msgs::msg::Marker>(
      "/cluster_markers", 10,
      std::bind(&CollisionDetectorNode::cbMarkers, this, std::placeholders::_1));

    sub_vel_ = create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
      "/ublox_gps_node/fix_velocity", 10,
      std::bind(&CollisionDetectorNode::cbVelocity, this, std::placeholders::_1));

    pub_flag_ = create_publisher<std_msgs::msg::UInt8>("/collision_flag", 10);
    pub_dbg_  = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/collision_debug_markers", 10);

    /* 초기화 --------------------------------------------------------- */
    clock_ = get_clock();
    initStaticZones();
    updateDangerZone();                       // speed=0
    prev_dbg_time_ = clock_->now();

    RCLCPP_INFO(get_logger(), "Collision-only node started (no green zone)");
    RCLCPP_INFO(get_logger(), "\033[33m[ZONES]\033[0m Caution: %.1fm front, %.1fm side | Danger: dynamic by speed", 
                caution_.front_ext, caution_.side_ext);
  }

private:
  /* ───────── 속도 콜백 ───────── */
  void cbVelocity(
    const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr msg)
  {
    const auto &v = msg->twist.twist.linear;
    float prev_speed = current_speed_;
    current_speed_ = std::hypot(v.x, v.y);
    
    // 속도 구간이 변경되었을 때만 로그 출력
    size_t prev_idx = SPEED_RANGE_.size() - 1;
    size_t curr_idx = SPEED_RANGE_.size() - 1;
    
    for (size_t i=0;i<SPEED_RANGE_.size();++i) {
      if (SPEED_RANGE_[i].first <= prev_speed && prev_speed < SPEED_RANGE_[i].second) 
        prev_idx = i;
      if (SPEED_RANGE_[i].first <= current_speed_ && current_speed_ < SPEED_RANGE_[i].second) 
        curr_idx = i;
    }
    
    if (prev_idx != curr_idx) {
      RCLCPP_INFO(get_logger(), "\033[36m[SPEED] %.2f m/s - Danger zone updated: front=%.1fm, radius=%.1fm\033[0m", 
                  current_speed_, DANGER_FRONT_[curr_idx], DANGER_RADIUS_[curr_idx]);
    }
    
    updateDangerZone();
  }

  /* ───────── 마커 콜백 ───────── */
  void cbMarkers(const visualization_msgs::msg::Marker::SharedPtr msg)
  {
    uint8_t flag = evaluate(msg->points);

    /* 플래그 퍼블리시 */
    std_msgs::msg::UInt8 f;  f.data = flag;
    pub_flag_->publish(f);

    /* 터미널 색상 로그 출력 */
    static uint8_t last_flag = 255;  // 초기값
    if (flag != last_flag) {
      // ANSI 색상 코드 정의
      const char* RESET = "\033[0m";
      const char* RED = "\033[31m";
      const char* YELLOW = "\033[33m";
      const char* GREEN = "\033[32m";
      const char* BOLD = "\033[1m";
      
      switch(flag) {
        case 0:
          RCLCPP_INFO(get_logger(), "%s%s[COLLISION] Clear - No obstacles detected%s", 
                      GREEN, BOLD, RESET);
          break;
        case 1:
          RCLCPP_WARN(get_logger(), "%s%s[COLLISION] CAUTION - Obstacle in yellow zone! (flag=1)%s", 
                      YELLOW, BOLD, RESET);
          break;
        case 2:
          RCLCPP_ERROR(get_logger(), "%s%s[COLLISION] DANGER - EMERGENCY STOP! (flag=2)%s", 
                       RED, BOLD, RESET);
          break;
      }
      last_flag = flag;
    }

    /* 디버그 (1 ms 스킵) */
    auto now = clock_->now();
    if ((now - prev_dbg_time_) < rclcpp::Duration(0, 1'000'000)) return;
    prev_dbg_time_ = now;
    pub_dbg_->publish(buildDebugArray(msg->header, flag));
  }

  /* ───────── 충돌 판정 ───────── */
  uint8_t evaluate(const std::vector<geometry_msgs::msg::Point>& pts)
  {
    bool danger=false, caution=false;
    for (const auto &p : pts)
    {
      if (inZone(p.x, p.y, danger_)) { danger=true; break; }
      else if (inZone(p.x, p.y, caution_)) caution=true;
    }
    if (danger)  return 2;  // 위험 (긴급정지)
    if (caution) return 1;  // 주의 (노란색)
    return 0;               // 감지 안됨
  }

  /* ───────── Zone 포함 검사 ───────── */
  bool inZone(float x, float y, const ZoneParam& z) const
  {
    float rear  = LIDAR_REAR_OFFSET_ - z.rear_ext;
    float front = LIDAR_REAR_OFFSET_ + VEH_L_ + z.front_ext;
    float half  = VEH_W_/2 + z.side_ext;

    if (rear <= x && x <= front && -half <= y && y <= half) return true;
    if (x >= front)
    {
      float dx = x - front;
      return dx*dx + y*y <= z.front_radius*z.front_radius;
    }
    return false;
  }

  /* ───────── Danger 동적 갱신 ───────── */
  void updateDangerZone()
  {
    size_t idx = SPEED_RANGE_.size() - 1;
    for (size_t i=0;i<SPEED_RANGE_.size();++i)
      if (SPEED_RANGE_[i].first <= current_speed_
          && current_speed_ < SPEED_RANGE_[i].second) { idx=i; break; }
    danger_ = { DANGER_FRONT_[idx], 0.0f, 0.0f,
                DANGER_RADIUS_[idx], {1,0,0}};
  }

  /* ───────── 디버그 마커 ───────── */
  visualization_msgs::msg::MarkerArray
  buildDebugArray(const std_msgs::msg::Header& hdr, uint8_t flag)
  {
    visualization_msgs::msg::MarkerArray arr;
    float half = VEH_W_/2;

    /* 차량 사각형 */
    const std::array<float,4> car_rgba{1,1,1,1};
    arr.markers.push_back(makeRect(hdr,"car",0,
      LIDAR_REAR_OFFSET_, LIDAR_REAR_OFFSET_+VEH_L_, -half, half, car_rgba));

    arr.markers.push_back(makeZone(hdr,"caution",1,caution_,flag==1));
    arr.markers.push_back(makeZone(hdr,"danger",2,danger_, flag==2));

    /* 속도 텍스트 */
    visualization_msgs::msg::Marker txt;
    txt.header=hdr; txt.ns="speed"; txt.id=3;
    txt.type=visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    txt.action=visualization_msgs::msg::Marker::ADD;
    txt.pose.position.x=LIDAR_REAR_OFFSET_+VEH_L_/2;
    txt.pose.position.y=0; txt.pose.position.z=1;
    txt.pose.orientation.w=1;
    txt.scale.z=0.3;
    txt.color.r=txt.color.g=txt.color.b=txt.color.a=1.0;
    txt.text = "v=" + std::to_string(static_cast<int>(current_speed_*10)/10.0) + " m/s";
    arr.markers.push_back(std::move(txt));
    return arr;
  }

  /* ───────── 헬퍼(사각형) ───────── */
  visualization_msgs::msg::Marker makeRect(
      const std_msgs::msg::Header& hdr,const std::string& ns,int id,
      float x0,float x1,float y0,float y1,const std::array<float,4>& rgba)
  {
    visualization_msgs::msg::Marker mk;
    mk.header=hdr; mk.ns=ns; mk.id=id;
    mk.type=visualization_msgs::msg::Marker::LINE_STRIP;
    mk.action=visualization_msgs::msg::Marker::ADD;
    mk.scale.x=0.06; mk.pose.orientation.w=1;
    mk.color.r=rgba[0]; mk.color.g=rgba[1];
    mk.color.b=rgba[2]; mk.color.a=rgba[3];

    std::vector<geometry_msgs::msg::Point> p(5);
    p[0].x=x0; p[0].y=y0;
    p[1].x=x1; p[1].y=y0;
    p[2].x=x1; p[2].y=y1;
    p[3].x=x0; p[3].y=y1;
    p[4]=p[0];
    mk.points=p;
    return mk;
  }

  /* ───────── 헬퍼(Zone) ───────── */
  visualization_msgs::msg::Marker makeZone(
      const std_msgs::msg::Header& hdr,const std::string& ns,int id,
      const ZoneParam& z,bool active)
  {
    visualization_msgs::msg::Marker mk;
    mk.header=hdr; mk.ns=ns; mk.id=id;
    mk.type=visualization_msgs::msg::Marker::LINE_STRIP;
    mk.action=visualization_msgs::msg::Marker::ADD;
    mk.scale.x=0.08; mk.pose.orientation.w=1;
    mk.color.r=z.color[0]; mk.color.g=z.color[1];
    mk.color.b=z.color[2]; mk.color.a=active?0.8f:0.25f;

    float rear  = LIDAR_REAR_OFFSET_ - z.rear_ext;
    float front = LIDAR_REAR_OFFSET_ + VEH_L_ + z.front_ext;
    float half  = VEH_W_/2 + z.side_ext;

    std::vector<geometry_msgs::msg::Point> pts;
    pts.reserve(22);
    geometry_msgs::msg::Point p; p.z=0;
    p.x=rear;  p.y=-half; pts.push_back(p);
    p.x=front; p.y=-half; pts.push_back(p);

    const int n=16;
    for(int i=0;i<=n;++i){
      double th=M_PI/2 - i*M_PI/n;
      p.x = front + z.front_radius*std::cos(th);
      p.y = z.front_radius*std::sin(th);
      pts.push_back(p);
    }
    p.x=front; p.y=half; pts.push_back(p);
    p.x=rear;  p.y=half; pts.push_back(p);
    pts.push_back(pts.front());

    mk.points=pts;
    return mk;
  }

  /* ───────── 고정 파라미터 ───────── */
  void initStaticZones()
  {
    // 차량 크기 (m)
    VEH_L_=1.1f;              // 차량 길이
    VEH_W_=0.7f;              // 차량 폭
    LIDAR_REAR_OFFSET_=-0.1f; // LiDAR 기준 차량 후방 오프셋

    // Caution Zone (주의 구역 - 노란색)
    caution_ = {
      2.0f,    // front_ext: 차량 전방 1.5m 연장
      0.5f,    // side_ext: 차량 측면 0.5m 연장
      -1.0f,   // rear_ext: 차량 후방 1.0m 연장 (음수로 후방 영역 제거)
      0.8f,    // front_radius: 전방 반원 반지름 0.8m
      {1,0.65f,0}  // 색상: 노란색 (Orange)
    };

    // Danger Zone (위험 구역 - 빨간색) - 속도에 따라 동적 변경
    SPEED_RANGE_={{0,0.3f},{0.3f,1},{1,2},{2,1e9f}};
    DANGER_FRONT_ ={0.1f,0.6f,1.2f,1.5f};   // -속도별 전방 연장 거리
    DANGER_RADIUS_={0.5f,0.6f,0.7f,0.8f};   // 속도별 전방 반원 반지름
  }

  /* ───────── 멤버 변수 ───────── */
  rclcpp::Subscription<visualization_msgs::msg::Marker>::SharedPtr sub_markers_;
  rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr sub_vel_;
  rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr pub_flag_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_dbg_;

  /* 차량/Zone 파라미터 */
  float VEH_L_, VEH_W_, LIDAR_REAR_OFFSET_;
  ZoneParam caution_, danger_;  // green_ 제거
  std::vector<std::pair<float,float>> SPEED_RANGE_;
  std::vector<float> DANGER_FRONT_, DANGER_RADIUS_;

  /* 상태 */
  float current_speed_{0.0f};
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Time prev_dbg_time_;
};

/* ───────── main ───────── */
int main(int argc,char** argv)
{
  rclcpp::init(argc,argv);
  rclcpp::spin(std::make_shared<CollisionDetectorNode>());
  rclcpp::shutdown();
  return 0;
}
