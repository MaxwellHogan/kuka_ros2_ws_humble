#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>

#include <geometry_msgs/msg/pose.h>
#include <geometry_msgs/msg/pose_stamped.h>

#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_srvs/srv/set_bool.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

#include <rclcpp/qos.hpp>

#include <chrono>
#include <fstream>
#include <mutex>
#include <optional>
#include <filesystem>

// Helper to get from a pose type message to a Transform type
static tf2::Transform poseToTf(const geometry_msgs::msg::Pose& p)
{
  tf2::Transform T;
  T.setOrigin(tf2::Vector3(p.position.x, p.position.y, p.position.z));
  tf2::Quaternion q(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w);
  T.setRotation(q);
  return T;
}

// Helper to get from a Transform type to a pose type message
static geometry_msgs::msg::Pose tfToPose(const tf2::Transform& T)
{
  geometry_msgs::msg::Pose p;
  p.position.x = T.getOrigin().x();
  p.position.y = T.getOrigin().y();
  p.position.z = T.getOrigin().z();
  p.orientation = tf2::toMsg(T.getRotation());
  return p;
}

// Helper function to print tf2::Transform as translation and SO(3) (rotation matrix and RPY)
void printTransform(const tf2::Transform& tf, const std::string& name)
{
  const tf2::Vector3& t = tf.getOrigin();
  const tf2::Quaternion& q = tf.getRotation();
  double roll, pitch, yaw;
  tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

  std::cout << name << " translation: [" << t.x() << ", " << t.y() << ", " << t.z() << "]" << std::endl;
  std::cout << name << " quaternion: [" << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w() << "]" << std::endl;
  std::cout << name << " RPY: [" << roll << ", " << pitch << ", " << yaw << "]" << std::endl;

  tf2::Matrix3x3 m(q);
  std::cout << name << " rotation matrix:" << std::endl;
  for (int i = 0; i < 3; ++i)
    std::cout << "[" << m[i][0] << " " << m[i][1] << " " << m[i][2] << "]" << std::endl;
}

static tf2::Matrix3x3 opti_to_frd_R()
{
  // Rows correspond to FRD axes expressed in Opti coordinates
  // x_frd = z_opti
  // y_frd = -y_opti
  // z_frd = x_opti
  return tf2::Matrix3x3(
    0,  0,  1,
    0, -1,  0,
    1,  0,  0
  );
}

static geometry_msgs::msg::PoseStamped convert_opti_pose_to_frd(const geometry_msgs::msg::PoseStamped& in)
{
  tf2::Transform T_opti = poseToTf(in.pose);

  tf2::Matrix3x3 R = opti_to_frd_R();
  tf2::Matrix3x3 Rt = R.transpose();

  // Convert translation
  tf2::Vector3 t = T_opti.getOrigin();
  tf2::Vector3 t_frd(
    R[0].dot(t),
    R[1].dot(t),
    R[2].dot(t)
  );

  // Convert rotation: R' = R * Ropti * R^T
  tf2::Quaternion q = T_opti.getRotation();
  tf2::Matrix3x3 Ropti(q);
  tf2::Matrix3x3 Rfrd = R * Ropti * Rt;

  tf2::Quaternion q_frd;
  Rfrd.getRotation(q_frd);

  geometry_msgs::msg::PoseStamped out = in;
  out.pose.position.x = t_frd.x();
  out.pose.position.y = t_frd.y();
  out.pose.position.z = t_frd.z();
  out.pose.orientation = tf2::toMsg(q_frd);
  return out;
}


class RelativePoseCommander : public rclcpp::Node
{
public:
  RelativePoseCommander() : Node("kr70_relative_pose_commander")
  {
    // --- params ---
    this->declare_parameter<std::string>("benchmark_csv_path", "benchmark_log.csv");
    this->declare_parameter<bool>("benchmark_enabled", false);

    benchmark_enabled_ = this->get_parameter("benchmark_enabled").as_bool();
    csv_path_ = this->get_parameter("benchmark_csv_path").as_string();

    // --- command subs ---
    sub_rel_pose_ = this->create_subscription<geometry_msgs::msg::Pose>(
      "relative_pose_cmd", rclcpp::QoS(1).best_effort(),
      std::bind(&RelativePoseCommander::relative_pose_callback, this, std::placeholders::_1));

    // default tool transform - 2 to 1 scale model
    flange_to_tool_.setOrigin(tf2::Vector3(0.2134, 0.5, 0.0));
    flange_to_tool_.setRotation(tf2::Quaternion(0.0, 0.0, 0.0, 1.0));

    set_tool_end_sub_ = this->create_subscription<std_msgs::msg::String>(
      "set_tool_end",
      rclcpp::QoS(1).reliable().keep_last(1),
      std::bind(&RelativePoseCommander::set_tool_end_callback, this, std::placeholders::_1)
    );

    send_to_sub_ = this->create_subscription<std_msgs::msg::String>(
      "send_to",
      10,
      std::bind(&RelativePoseCommander::send_to_callback, this, std::placeholders::_1));

    // --- move_done pub ---
    rclcpp::QoS qos_profile(1);
    qos_profile.transient_local();
    qos_profile.reliable();
    qos_profile.keep_last(1);

    move_done_pub_ = this->create_publisher<std_msgs::msg::String>("move_done", qos_profile);

    // --- OptiTrack pose sub (Best effort) ---
    // Topic: /vrpn_mocap/Grasp_1_1_rig/pose
    rclcpp::QoS vrpn_qos(1);
    vrpn_qos.best_effort();
    vrpn_qos.keep_last(1);
    vrpn_qos.durability_volatile();

    optitrack_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/vrpn_mocap/Grasp_1_1_rig/pose",
      vrpn_qos,
      std::bind(&RelativePoseCommander::optitrack_callback, this, std::placeholders::_1)
    );

    // --- benchmark enable/disable service ---
    benchmark_srv_ = this->create_service<std_srvs::srv::SetBool>(
      "benchmark_enable",
      std::bind(&RelativePoseCommander::benchmark_enable_callback, this,
                std::placeholders::_1, std::placeholders::_2)
    );

    // If benchmark enabled at startup, open file now.
    if (benchmark_enabled_) {
      open_csv_if_needed();
    }
  }

  void initialize_move_group()
  {
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), "manipulator");

    move_group_->setPlanningPipelineId("pilz");
    move_group_->setPlannerId("LIN");
    move_group_->setMaxVelocityScalingFactor(0.75);
    move_group_->setMaxAccelerationScalingFactor(0.05);
    move_group_->setPlanningTime(10.0);
    move_group_->setNumPlanningAttempts(20);
    move_group_->setStartStateToCurrentState();
    move_group_->setGoalPositionTolerance(0.001);
    move_group_->setGoalOrientationTolerance(0.00174533);
  }

private:
  // ---------- OptiTrack ----------
  void optitrack_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    geometry_msgs::msg::PoseStamped frd = convert_opti_pose_to_frd(*msg);
    
    std::lock_guard<std::mutex> lk(optitrack_mutex_);
    latest_optitrack_pose_ = frd;
  }

  std::optional<geometry_msgs::msg::PoseStamped> get_latest_optitrack_pose_copy() const
  {
    std::lock_guard<std::mutex> lk(optitrack_mutex_);
    return latest_optitrack_pose_;
  }

  // ---------- Benchmark control ----------
  void benchmark_enable_callback(const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
                                 std::shared_ptr<std_srvs::srv::SetBool::Response> res)
  {
    benchmark_enabled_ = req->data;

    if (benchmark_enabled_) {
        open_csv_if_needed();

        // Capture session baseline at the moment logging starts
        auto robot_tool_start = get_robot_tool_pose_now();                  // PoseStamped
        auto opti_start_opt   = get_latest_optitrack_pose_copy();           // optional PoseStamped (already FRD converted by callback)

        std::lock_guard<std::mutex> lk(session_mutex_);
        session_robot_T0_ = poseToTf(robot_tool_start.pose);

        if (opti_start_opt.has_value()) {
            session_opti_T0_ = poseToTf(opti_start_opt->pose);
            session_T0_valid_ = true;
        } else {
            session_T0_valid_ = false; // no opti at enable time
        }

        res->success = true;
        res->message = "Benchmark logging enabled";
        RCLCPP_INFO(this->get_logger(), "Benchmark logging ENABLED. CSV: %s", csv_path_.c_str());
    } else {
      close_csv();
      res->success = true;
      res->message = "Benchmark logging disabled";
      RCLCPP_INFO(this->get_logger(), "Benchmark logging DISABLED.");
    }
  }

  void open_csv_if_needed()
  {
    std::lock_guard<std::mutex> lk(csv_mutex_);

    if (csv_stream_.is_open())
      return;

    // ensure parent directory exists (if user provided folders)
    try {
      std::filesystem::path p(csv_path_);
      if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
      }
    } catch (...) {
      // ignore - we'll try to open anyway
    }

    bool need_header = true;
    {
      std::ifstream check(csv_path_, std::ios::binary);
      need_header = !check.good() || (check.peek() == std::ifstream::traits_type::eof());
    }

    csv_stream_.open(csv_path_, std::ios::out | std::ios::app);
    if (!csv_stream_.is_open()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open CSV file: %s", csv_path_.c_str());
      return;
    }

    if (need_header) {
      csv_stream_
        << "t_cmd_ros_ns,"
        << "command_type,"
        << "plan_time_ms,"
        << "exec_time_ms,"
        << "robot_tool_start_px,robot_tool_start_py,robot_tool_start_pz,robot_tool_start_qx,robot_tool_start_qy,robot_tool_start_qz,robot_tool_start_qw,"
        << "robot_tool_end_px,robot_tool_end_py,robot_tool_end_pz,robot_tool_end_qx,robot_tool_end_qy,robot_tool_end_qz,robot_tool_end_qw,"
        << "optitrack_start_ros_ns,optitrack_start_px,optitrack_start_py,optitrack_start_pz,optitrack_start_qx,optitrack_start_qy,optitrack_start_qz,optitrack_start_qw,"
        << "optitrack_end_ros_ns,optitrack_end_px,optitrack_end_py,optitrack_end_pz,optitrack_end_qx,optitrack_end_qy,optitrack_end_qz,optitrack_end_qw"
        << "\n";
      csv_stream_.flush();
    }
  }

  void close_csv()
  {
    std::lock_guard<std::mutex> lk(csv_mutex_);
    if (csv_stream_.is_open()) {
      csv_stream_.flush();
      csv_stream_.close();
    }
  }

  // Compute robot tool pose from current flange pose and current TCP (F_T).
  // Returns PoseStamped in the planning frame (whatever getCurrentPose returns).
  geometry_msgs::msg::PoseStamped get_robot_tool_pose_now()
  {
    geometry_msgs::msg::PoseStamped flange_pose = move_group_->getCurrentPose("flange");

    tf2::Transform curr_flange_tf;
    tf2::fromMsg(flange_pose.pose, curr_flange_tf);

    tf2::Transform F_T;
    {
      std::lock_guard<std::mutex> lk(tool_mutex_);
      F_T = flange_to_tool_;
    }

    tf2::Transform curr_tool_tf = curr_flange_tf * F_T;

    geometry_msgs::msg::PoseStamped tool_pose = flange_pose;
    geometry_msgs::msg::Pose pose_msg;
    pose_msg.position.x = curr_tool_tf.getOrigin().x();
    pose_msg.position.y = curr_tool_tf.getOrigin().y();
    pose_msg.position.z = curr_tool_tf.getOrigin().z();
    pose_msg.orientation = tf2::toMsg(curr_tool_tf.getRotation());
    tool_pose.pose = pose_msg;
    return tool_pose;
  }

  static void pose_to_xyzquat(const geometry_msgs::msg::Pose& p,
                             double& x, double& y, double& z,
                             double& qx, double& qy, double& qz, double& qw)
  {
    x = p.position.x; y = p.position.y; z = p.position.z;
    qx = p.orientation.x; qy = p.orientation.y; qz = p.orientation.z; qw = p.orientation.w;
  }

void write_benchmark_row(const std::string& command_type,
                         int64_t t_cmd_ros_ns,
                         double plan_ms,
                         double exec_ms,
                         const geometry_msgs::msg::PoseStamped& robot_tool_start,
                         const geometry_msgs::msg::PoseStamped& robot_tool_end,
                         const std::optional<geometry_msgs::msg::PoseStamped>& opti_start,
                         const std::optional<geometry_msgs::msg::PoseStamped>& opti_end)
{
  if (!benchmark_enabled_)
    return;

  std::lock_guard<std::mutex> lk(csv_mutex_);
  if (!csv_stream_.is_open())
    return;

  // --- Compute relative poses (T_rel = T0^-1 * T) ---
  tf2::Transform Trs = poseToTf(robot_tool_start.pose);
  tf2::Transform Tre = poseToTf(robot_tool_end.pose);

  tf2::Transform Trs_rel, Tre_rel;
  bool opti_rel_ok = false;
  tf2::Transform Tos_rel, Toe_rel;

  {
    std::lock_guard<std::mutex> lk_sess(session_mutex_);

    // Robot relative-to-session-start (always)
    Trs_rel = session_robot_T0_.inverse() * Trs;
    Tre_rel = session_robot_T0_.inverse() * Tre;

    // Opti relative-to-session-start (only if baseline valid)
    if (session_T0_valid_) {
      opti_rel_ok = true;

      if (opti_start.has_value()) {
        tf2::Transform Tos = poseToTf(opti_start->pose);
        Tos_rel = session_opti_T0_.inverse() * Tos;
      } else {
        opti_rel_ok = false;
      }

      if (opti_end.has_value()) {
        tf2::Transform Toe = poseToTf(opti_end->pose);
        Toe_rel = session_opti_T0_.inverse() * Toe;
      } else {
        opti_rel_ok = false;
      }
    }
  }

  // Convert rel transforms -> pose messages for logging convenience
  geometry_msgs::msg::Pose robot_start_rel_pose = tfToPose(Trs_rel);
  geometry_msgs::msg::Pose robot_end_rel_pose   = tfToPose(Tre_rel);

  double rsx,rsy,rsz,rsqx,rsqy,rsqz,rsqw;
  double rex,rey,rez,reqx,reqy,reqz,reqw;
  pose_to_xyzquat(robot_start_rel_pose, rsx,rsy,rsz,rsqx,rsqy,rsqz,rsqw);
  pose_to_xyzquat(robot_end_rel_pose,   rex,rey,rez,reqx,reqy,reqz,reqw);

  auto write_opti_rel = [&](const std::optional<geometry_msgs::msg::PoseStamped>& o,
                            const tf2::Transform& Trel)
  {
    if (!o.has_value()) {
      csv_stream_ << "0,0,0,0,0,0,0,0";
      return;
    }

    // Keep the original timestamp from OptiTrack sample (useful for debugging latency)
    int64_t t_ns = rclcpp::Time(o->header.stamp).nanoseconds();

    geometry_msgs::msg::Pose p = tfToPose(Trel);
    double x,y,z,qx,qy,qz,qw;
    pose_to_xyzquat(p, x,y,z,qx,qy,qz,qw);

    csv_stream_ << t_ns << "," << x << "," << y << "," << z << ","
                << qx << "," << qy << "," << qz << "," << qw;
  };

  // --- Write CSV row (now RELATIVE values) ---
  csv_stream_
    << t_cmd_ros_ns << ","
    << command_type << ","
    << plan_ms << ","
    << exec_ms << ","
    // robot start/end RELATIVE
    << rsx << "," << rsy << "," << rsz << "," << rsqx << "," << rsqy << "," << rsqz << "," << rsqw << ","
    << rex << "," << rey << "," << rez << "," << reqx << "," << reqy << "," << reqz << "," << reqw << ",";

  if (session_T0_valid_) {
    // OptiTrack RELATIVE (per-sample timestamp retained)
    write_opti_rel(opti_start, Tos_rel);
    csv_stream_ << ",";
    write_opti_rel(opti_end, Toe_rel);
  } else {
    // No session baseline for opti
    csv_stream_ << "0,0,0,0,0,0,0,0,";
    csv_stream_ << "0,0,0,0,0,0,0,0";
  }

  csv_stream_ << "\n";
  csv_stream_.flush();
}


  // ---------- Motion callbacks ----------
  void relative_pose_callback(const geometry_msgs::msg::Pose::SharedPtr rel_pose_msg)
  {
    // Command start timestamp
    const int64_t t_cmd_ros_ns = this->now().nanoseconds();

    // Snapshot start poses
    geometry_msgs::msg::PoseStamped robot_tool_start = get_robot_tool_pose_now();
    std::optional<geometry_msgs::msg::PoseStamped> opti_start = get_latest_optitrack_pose_copy();

    // clear trajectory info
    move_group_->clearPoseTargets();
    move_group_->clearPathConstraints();
    move_group_->setStartStateToCurrentState();
    move_group_->setMaxVelocityScalingFactor(0.75);
    move_group_->setMaxAccelerationScalingFactor(0.05);

    // current flange pose
    geometry_msgs::msg::PoseStamped curr_flange_pose = move_group_->getCurrentPose("flange");

    tf2::Transform curr_tf, rel_tf;
    tf2::fromMsg(curr_flange_pose.pose, curr_tf);
    tf2::fromMsg(*rel_pose_msg, rel_tf);

    // TCP correction: interpret relative motion at tool, convert to flange motion
    tf2::Transform F_T, T_F;
    {
      std::lock_guard<std::mutex> lk(tool_mutex_);
      F_T = flange_to_tool_;
    }
    T_F = F_T.inverse();
    rel_tf = F_T * rel_tf * T_F;

    tf2::Transform goal_tf = curr_tf * rel_tf;
    geometry_msgs::msg::Pose goal_pose;
    goal_pose.position.x = goal_tf.getOrigin().x();
    goal_pose.position.y = goal_tf.getOrigin().y();
    goal_pose.position.z = goal_tf.getOrigin().z();
    goal_pose.orientation = tf2::toMsg(goal_tf.getRotation());

    move_group_->setPoseTarget(goal_pose, "flange");

    // --- Plan timing ---
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    auto t0 = std::chrono::steady_clock::now();
    bool success = (move_group_->plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    auto t1 = std::chrono::steady_clock::now();
    double plan_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double exec_ms = 0.0;

    if (success)
    {
      RCLCPP_INFO(this->get_logger(), "Relative move planned. Executing...");

      // --- Execute timing ---
      auto e0 = std::chrono::steady_clock::now();
      move_group_->execute(plan);
      auto e1 = std::chrono::steady_clock::now();
      exec_ms = std::chrono::duration<double, std::milli>(e1 - e0).count();

      auto msg = std_msgs::msg::String();
      msg.data = "COMPLETE";
      move_done_pub_->publish(msg);
    }
    else
    {
      auto msg = std_msgs::msg::String();
      msg.data = "FAILURE";
      move_done_pub_->publish(msg);
      RCLCPP_WARN(this->get_logger(), "Planning failed for relative move!");
    }

    // Snapshot end poses (even if failure; useful for debugging)
    geometry_msgs::msg::PoseStamped robot_tool_end = get_robot_tool_pose_now();
    std::optional<geometry_msgs::msg::PoseStamped> opti_end = get_latest_optitrack_pose_copy();

    write_benchmark_row("RELATIVE_POSE", t_cmd_ros_ns, plan_ms, exec_ms,
                        robot_tool_start, robot_tool_end, opti_start, opti_end);
  }

  void set_tool_end_callback(const std_msgs::msg::String::SharedPtr msg)
  {
    const std::string& key = msg->data;
    tf2::Transform new_tcp;

    if (key == "GRASP_1_1") {
      new_tcp.setOrigin(tf2::Vector3(0.346, 0.0, 0.0));
      new_tcp.setRotation(tf2::Quaternion(0.0, 0.0, 0.0, 1.0));
    } else if (key == "GRASP_2_1") {
      new_tcp.setOrigin(tf2::Vector3(0.2134, 0.5, 0.0));
      new_tcp.setRotation(tf2::Quaternion(0.0, 0.0, 0.0, 1.0));
    } else {
      RCLCPP_WARN(this->get_logger(),
        "set_tool_end: unknown key '%s' (expected 'GRASP_1_1' or 'GRASP_2_1')", key.c_str());
      return;
    }

    {
      std::lock_guard<std::mutex> lk(tool_mutex_);
      flange_to_tool_ = new_tcp;
    }

    RCLCPP_INFO(this->get_logger(),
      "Tool set to %s: t=[%.4f %.4f %.4f], q=[%.3f %.3f %.3f %.3f]",
      key.c_str(),
      new_tcp.getOrigin().x(), new_tcp.getOrigin().y(), new_tcp.getOrigin().z(),
      new_tcp.getRotation().x(), new_tcp.getRotation().y(),
      new_tcp.getRotation().z(), new_tcp.getRotation().w());
  }

  void send_to_callback(const std_msgs::msg::String::SharedPtr msg)
  {
    if (msg->data == "START")
    {
      RCLCPP_INFO(this->get_logger(), "Received START on /send_to. Moving to start position.");
      move_to_start_position();
    }
    else
    {
      RCLCPP_WARN(this->get_logger(), "Received unknown command on /send_to: '%s'", msg->data.c_str());
    }
  }

  void move_to_start_position()
  {
    const int64_t t_cmd_ros_ns = this->now().nanoseconds();

    geometry_msgs::msg::PoseStamped robot_tool_start = get_robot_tool_pose_now();
    std::optional<geometry_msgs::msg::PoseStamped> opti_start = get_latest_optitrack_pose_copy();

    move_group_->clearPoseTargets();
    move_group_->clearPathConstraints();
    move_group_->setStartStateToCurrentState();
    move_group_->setMaxVelocityScalingFactor(0.25);
    move_group_->setMaxAccelerationScalingFactor(0.005);

    std::vector<std::string> joint_names = {"A1","A2","A3","A4","A5","A6","E1"};
    std::vector<double> joint_positions = {
      0.7335758472472327,
      -1.0502588587045958,
      1.8730315027042506,
      -0.9959110511267444,
      2.036048745499025,
      -0.61374328613456,
      -3.644,
    };

    std::map<std::string, double> target_joint_values;
    for (size_t i = 0; i < joint_names.size(); ++i)
      target_joint_values[joint_names[i]] = joint_positions[i];

    move_group_->setJointValueTarget(target_joint_values);

    moveit::planning_interface::MoveGroupInterface::Plan my_plan;

    auto t0 = std::chrono::steady_clock::now();
    bool success = (move_group_->plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    auto t1 = std::chrono::steady_clock::now();
    double plan_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double exec_ms = 0.0;

    if (success)
    {
      auto e0 = std::chrono::steady_clock::now();
      move_group_->execute(my_plan);
      auto e1 = std::chrono::steady_clock::now();
      exec_ms = std::chrono::duration<double, std::milli>(e1 - e0).count();

      auto msg = std_msgs::msg::String();
      msg.data = "COMPLETE";
      move_done_pub_->publish(msg);
    }
    else
    {
      auto msg = std_msgs::msg::String();
      msg.data = "FAILURE";
      move_done_pub_->publish(msg);
      RCLCPP_ERROR(this->get_logger(), "Failed to plan move to start position!");
    }

    geometry_msgs::msg::PoseStamped robot_tool_end = get_robot_tool_pose_now();
    std::optional<geometry_msgs::msg::PoseStamped> opti_end = get_latest_optitrack_pose_copy();

    write_benchmark_row("SEND_TO_START", t_cmd_ros_ns, plan_ms, exec_ms,
                        robot_tool_start, robot_tool_end, opti_start, opti_end);
  }

  // ---------- members ----------
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr sub_rel_pose_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr send_to_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr move_done_pub_;

  // Tool transform
  tf2::Transform flange_to_tool_;
  std::mutex tool_mutex_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr set_tool_end_sub_;

  // OptiTrack cache
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr optitrack_sub_;
  mutable std::mutex optitrack_mutex_;
  std::optional<geometry_msgs::msg::PoseStamped> latest_optitrack_pose_;

  // Benchmark logging
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr benchmark_srv_;
  bool benchmark_enabled_{false};
  std::string csv_path_;
  std::mutex csv_mutex_;
  std::ofstream csv_stream_;

  // Session baseline (captured when logging is enabled)
  std::mutex session_mutex_;
  bool session_T0_valid_{false};
  tf2::Transform session_robot_T0_;
  tf2::Transform session_opti_T0_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RelativePoseCommander>();
  node->initialize_move_group();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
