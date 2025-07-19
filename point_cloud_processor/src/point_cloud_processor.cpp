/*  point_cloud_processor.cpp
 *  ROS 2  Humble 용 PointCloud2 전처리 노드
 *
 *  - 입력 : /velodyne_points   (sensor_msgs/msg/PointCloud2)
 *  - 출력 : /processed_velodyne_points_tmp  (sensor_msgs/msg/PointCloud2)
 *
 *  기능
 *    1. ROI(각도·거리·좌우 제한) 필터링
 *    2. 지면 제거(그리드 기반 최소 z 비교)
 *    3. 선택적 업샘플링(K-d Tree + KNN 평균)
 *    4. 2D 투영(z = 0) 후 퍼블리시
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <unordered_map>
#include <random>
#include <cmath>

using std::placeholders::_1;
using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

class PointCloudProcessor : public rclcpp::Node
{
public:
  PointCloudProcessor()
  : Node("point_cloud_processor")
  {
    /* ───────── 파라미터 초기값 ───────── */
    min_angle_        = this->declare_parameter<double>("min_angle",   -100.0);
    max_angle_        = this->declare_parameter<double>("max_angle",    100.0);
    min_range_        = this->declare_parameter<double>("min_range",      0.3);
    max_range_        = this->declare_parameter<double>("max_range",     10.0);
    min_y_            = this->declare_parameter<double>("min_y",        -3.0);
    max_y_            = this->declare_parameter<double>("max_y",         3.0);
    use_width_ratio_  = this->declare_parameter<bool  >("use_width_ratio", false);
    width_ratio_      = this->declare_parameter<double>("width_ratio",   0.5);

    use_upsampling_   = this->declare_parameter<bool  >("use_upsampling", false);
    upsample_factor_  = this->declare_parameter<int   >("upsample_factor", 2);
    knn_neighbors_    = this->declare_parameter<int   >("knn_neighbors",   5);

    use_ground_removal_ = this->declare_parameter<bool>("use_ground_removal", true);

    /* ───────── 통신 설정 ───────── */
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/velodyne_points", 10,
      std::bind(&PointCloudProcessor::callback, this, _1));

    pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/processed_velodyne_points", 10);

    RCLCPP_INFO(get_logger(), "ROI Angle  : %.1f° ~ %.1f°", min_angle_, max_angle_);
    RCLCPP_INFO(get_logger(), "ROI Range  : %.2fm ~ %.2fm", min_range_, max_range_);
    RCLCPP_INFO(get_logger(), "ROI Y-axis : %.2fm ~ %.2fm", min_y_,     max_y_);
    RCLCPP_INFO(get_logger(), "Ground removal : %s", use_ground_removal_ ? "ON" : "OFF");
    RCLCPP_INFO(get_logger(), "Upsampling     : %s", use_upsampling_    ? "ON" : "OFF");
  }

private:
  /* ----------- 콜백 ----------- */
  void callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    /* ① ROS → PCL */
    CloudT::Ptr cloud(new CloudT);
    pcl::fromROSMsg(*msg, *cloud);
    const size_t original_cnt = cloud->size();
    if (original_cnt == 0) {
      RCLCPP_WARN(get_logger(), "Empty cloud received.");
      return;
    }

    /* ② ROI 필터 */
    CloudT::Ptr roi_cloud = apply_roi_filter(cloud);
    const size_t roi_cnt = roi_cloud->size();
    if (roi_cnt == 0) {
      RCLCPP_WARN(get_logger(), "No points in ROI.");
      return;
    }

    /* ③ 지면 제거 */
    CloudT::Ptr nonground_cloud = use_ground_removal_
                                  ? remove_ground_plane(roi_cloud)
                                  : roi_cloud;
    const size_t ground_removed_cnt = nonground_cloud->size();

    /* ④ 업샘플링 */
    CloudT::Ptr processed_cloud = (use_upsampling_ && ground_removed_cnt > knn_neighbors_)
                                  ? increase_density(nonground_cloud)
                                  : nonground_cloud;
    const size_t final_cnt = processed_cloud->size();

    /* ⑤ 2D 투영 (z = 0) */
    for (auto & pt : *processed_cloud)  pt.z = 0.0f;

    /* ⑥ PCL → ROS */
    sensor_msgs::msg::PointCloud2 out_msg;
    pcl::toROSMsg(*processed_cloud, out_msg);
    out_msg.header = msg->header;
    pub_->publish(out_msg);

    /* 로그 */
    RCLCPP_INFO(get_logger(),
      "Published %zu points (orig:%zu → ROI:%zu → ground:%zu → final:%zu)",
      final_cnt, original_cnt, roi_cnt, ground_removed_cnt, final_cnt);
  }


  /* ----------- ROI 필터 ----------- */
  CloudT::Ptr apply_roi_filter(const CloudT::Ptr& cloud)
  {
    CloudT::Ptr out(new CloudT);
    out->reserve(cloud->size());

    for (const auto & p : *cloud)
    {
      const double dist   = std::hypot(p.x, p.y);
      const double angle  = std::atan2(p.y, p.x) * 180.0 / M_PI;

      if (dist < min_range_ || dist > max_range_) continue;
      if (angle < min_angle_ || angle > max_angle_) continue;

      bool y_ok = false;
      if (use_width_ratio_) {
        const double y_limit = std::abs(p.x) * width_ratio_;
        y_ok = std::abs(p.y) <= y_limit;
      } else {
        y_ok = (p.y >= min_y_) && (p.y <= max_y_);
      }
      if (!y_ok) continue;

      out->push_back(p);
    }
    return out;
  }

  /* ----------- 지면 제거 ----------- */
  CloudT::Ptr remove_ground_plane(const CloudT::Ptr& cloud)
  {
    constexpr double GRID = 0.5;   // [m]
    constexpr double Z_TH = 0.2;   // [m]

    struct PairHash {
      std::size_t operator()(const std::pair<int,int>& p) const noexcept
      { return (static_cast<std::size_t>(p.first) << 32) ^ p.second; }
    };

    std::unordered_map<std::pair<int,int>, float, PairHash> cell_min_z;

    /* 각 셀 최저 z 계산 */
    for (const auto & pt : *cloud) {
      int gx = static_cast<int>(std::floor(pt.x / GRID));
      int gy = static_cast<int>(std::floor(pt.y / GRID));
      auto key = std::make_pair(gx, gy);
      auto it  = cell_min_z.find(key);
      if (it == cell_min_z.end() || pt.z < it->second)
        cell_min_z[key] = pt.z;
    }

    /* ground 여부 판정 */
    CloudT::Ptr out(new CloudT);
    out->reserve(cloud->size());
    size_t ground_cnt = 0;

    for (const auto & pt : *cloud) {
      int gx = static_cast<int>(std::floor(pt.x / GRID));
      int gy = static_cast<int>(std::floor(pt.y / GRID));

      float min_neighbor_z = std::numeric_limits<float>::infinity();
      for (int dx=-1; dx<=1; ++dx)
        for (int dy=-1; dy<=1; ++dy) {
          auto it = cell_min_z.find(std::make_pair(gx+dx, gy+dy));
          if (it != cell_min_z.end() && it->second < min_neighbor_z)
            min_neighbor_z = it->second;
        }

      if (pt.z - min_neighbor_z < Z_TH) {
        ++ground_cnt;          // ground
      } else {
        out->push_back(pt);    // keep
      }
    }

    RCLCPP_INFO(get_logger(), "Ground removal: %zu points removed", ground_cnt);
    return out;
  }

  /* ----------- 업샘플링 ----------- */
  CloudT::Ptr increase_density(const CloudT::Ptr& cloud)
  {
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);

    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> noise(0.0f, 0.01f);

    CloudT::Ptr out(new CloudT(*cloud));   // 원본 복사

    const size_t n_base = cloud->size();
    const size_t n_samples = std::max<size_t>(1, n_base / 2);
    std::uniform_int_distribution<size_t> uni(0, n_base - 1);

    for (int iter=0; iter<upsample_factor_; ++iter)
    {
      std::vector<int> nn_indices(knn_neighbors_);
      std::vector<float> nn_dists(knn_neighbors_);

      for (size_t i=0; i<n_samples; ++i)
      {
        const PointT & sp = cloud->at( uni(gen) );
        int found = kdtree.nearestKSearch(sp, knn_neighbors_, nn_indices, nn_dists);
        if (found == 0) continue;

        Eigen::Vector3f mean = Eigen::Vector3f::Zero();
        for (int idx : nn_indices)
          mean += cloud->at(idx).getVector3fMap();
        mean /= static_cast<float>(found);

        PointT new_pt;
        new_pt.x = mean.x() + noise(gen);
        new_pt.y = mean.y() + noise(gen);
        new_pt.z = mean.z() + noise(gen);
        out->push_back(new_pt);
      }
    }
    RCLCPP_INFO(get_logger(), "Upsampling: added %zu new points", out->size() - cloud->size());
    return out;
  }

  /* ----------- 멤버 ----------- */
  /* 파라미터 */
  double min_angle_, max_angle_, min_range_, max_range_;
  double min_y_, max_y_, width_ratio_;
  bool   use_width_ratio_;

  bool   use_upsampling_;
  int    upsample_factor_, knn_neighbors_;

  bool   use_ground_removal_;

  /* 통신 */
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    pub_;
};

/* ----------- main ----------- */
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PointCloudProcessor>());
  rclcpp::shutdown();
  return 0;
}

