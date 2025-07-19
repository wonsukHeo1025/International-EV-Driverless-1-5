/*  point_cloud_cluster_node.cpp
 *
 *  - 입력  : /processed_velodyne_points   (sensor_msgs/PointCloud2)
 *  - 출력  : /cluster_markers             (visualization_msgs/Marker  – 클러스터 중심점)
 *
 *  기능
 *    1. Euclidean Clustering으로 3D 클러스터 추출
 *    2. 중심점 간 거리가 merge_threshold 이하인 클러스터 병합
 *    3. 병합된 클러스터들의 중심점을 Marker(POINTS)로 퍼블리시
 *
 *  수정: DBSCAN → Euclidean Clustering
 *  작성자 : ChatGPT-o3  (2025-07-10)
 */

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <Eigen/Dense>

#include <vector>
#include <unordered_set>
#include <algorithm>
#include <cmath>

using std::placeholders::_1;
using PointT  = pcl::PointXYZ;
using CloudT  = pcl::PointCloud<PointT>;
using Vec3f   = Eigen::Vector3f;

/* ─────────────────────────────────────────────────────────────────────────── */

class PointCloudClusterNode : public rclcpp::Node
{
public:
  PointCloudClusterNode()
  : Node("point_cloud_cluster_node")
  {
    /* ── 파라미터 선언 (launch/CLI 로 조정 가능) ─────────────────────── */
    cluster_tolerance_ = declare_parameter<double>("cluster_tolerance", 0.05);   // 유클리디안 클러스터링 거리 임계값 [m]
    min_cluster_size_  = declare_parameter<int>("min_cluster_size", 10);        // 최소 클러스터 크기
    max_cluster_size_  = declare_parameter<int>("max_cluster_size", 25000);     // 최대 클러스터 크기
    merge_thresh_      = declare_parameter<double>("merge_threshold", 0.01);    // 클러스터 병합 경계 [m]
    marker_scale_      = declare_parameter<double>("marker_scale", 0.30);       // Marker 점 크기 [m]

    /* ── 통신 설정 ──────────────────────────────────────────────────── */
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "/processed_velodyne_points", 10, std::bind(&PointCloudClusterNode::cbCloud, this, _1));

    pub_ = create_publisher<visualization_msgs::msg::Marker>("/cluster_markers", 10);

    RCLCPP_INFO(get_logger(), "Euclidean Clustering tolerance=%.3f  min_size=%d  max_size=%d  merge_threshold=%.3f",
                cluster_tolerance_, min_cluster_size_, max_cluster_size_, merge_thresh_);
  }

private:
  /* ========= Euclidean Clustering 구현 ========= */
  std::vector<pcl::PointIndices> euclideanClustering(const CloudT::Ptr & cloud)
  {
    // KdTree 생성
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);

    // 유클리디안 클러스터 추출 객체 설정
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(cluster_tolerance_);  // 거리 임계값
    ec.setMinClusterSize(min_cluster_size_);     // 최소 클러스터 크기
    ec.setMaxClusterSize(max_cluster_size_);     // 최대 클러스터 크기
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);

    // 클러스터 추출
    std::vector<pcl::PointIndices> cluster_indices;
    ec.extract(cluster_indices);

    return cluster_indices;
  }

  /* ========= 중심점 계산 ========= */
  static Vec3f centroid(const CloudT::Ptr & cloud, const pcl::PointIndices& indices)
  {
    Vec3f c(0,0,0);
    for (int i : indices.indices) {
      c += cloud->points[i].getVector3fMap();
    }
    c /= static_cast<float>(indices.indices.size());
    return c;
  }

  /* ========= 콜백 ========= */
  void cbCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  {
    /* ① ROS → PCL */
    CloudT::Ptr cloud(new CloudT);
    pcl::fromROSMsg(*msg, *cloud);
    if (cloud->empty()) {
      RCLCPP_WARN(get_logger(), "Empty cloud received");
      return;
    }

    /* ② Euclidean Clustering */
    auto cluster_indices = euclideanClustering(cloud);
    if (cluster_indices.empty()) {
      RCLCPP_INFO(get_logger(), "No clusters found");
      return;
    }

    /* ③ 중심점 계산 */
    std::vector<Vec3f> centers;
    centers.reserve(cluster_indices.size());
    for (const auto & indices : cluster_indices) {
      centers.emplace_back(centroid(cloud, indices));
    }

    /* ④ 클러스터 병합 (simple O(N²) – N이 적어 충분히 빠름) */
    const size_t M = centers.size();
    std::vector<int> parent(M);                         // Union-Find
    std::iota(parent.begin(), parent.end(), 0);

    auto find_root = [&](int x){
      while (parent[x] != x) x = parent[x] = parent[parent[x]];
      return x;
    };
    auto unite = [&](int a, int b){
      a = find_root(a); b = find_root(b);
      if (a != b) parent[b] = a;
    };

    for (size_t i=0; i<M; ++i)
      for (size_t j=i+1; j<M; ++j)
        if ((centers[i]-centers[j]).norm() < merge_thresh_)
          unite(i,j);

    /* root 별로 모으기 */
    std::unordered_map<int, std::vector<Vec3f>> merged;
    for (size_t i=0;i<M;++i)
      merged[find_root(i)].push_back(centers[i]);

    /* 최종 centroid 재계산 */
    std::vector<Vec3f> final_centers;
    final_centers.reserve(merged.size());
    for (auto & kv : merged)
    {
      Vec3f mean(0,0,0);
      for (auto & c : kv.second) mean += c;
      mean /= kv.second.size();
      final_centers.push_back(mean);
    }

    /* ⑤ Marker 작성 */
    visualization_msgs::msg::Marker marker;
    marker.header   = msg->header;
    marker.ns       = "cluster_centers";
    marker.id       = 0;
    marker.type     = visualization_msgs::msg::Marker::POINTS;
    marker.action   = visualization_msgs::msg::Marker::ADD;
    marker.scale.x  = marker_scale_;
    marker.scale.y  = marker_scale_;
    marker.color.a  = 1.0f;

    for (const auto & c : final_centers)
    {
      geometry_msgs::msg::Point p;
      p.x = c.x(); p.y = c.y(); p.z = c.z();
      marker.points.push_back(p);

      std_msgs::msg::ColorRGBA col;
      col.r = 1.0f; col.g = 0.0f; col.b = 0.0f; col.a = 1.0f;
      marker.colors.push_back(col);
    }

    pub_->publish(marker);
    RCLCPP_INFO(get_logger(), "Found %zu clusters, published %zu merged cluster centers", 
                cluster_indices.size(), final_centers.size());
  }

  /* ── 멤버 ───────────────────────────────────────────────────────── */
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_;

  double cluster_tolerance_, merge_thresh_, marker_scale_;
  int min_cluster_size_, max_cluster_size_;
};

/* ===== main ===== */
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PointCloudClusterNode>());
  rclcpp::shutdown();
  return 0;
}
