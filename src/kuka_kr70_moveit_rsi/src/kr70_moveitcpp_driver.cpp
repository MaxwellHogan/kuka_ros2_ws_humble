#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.h>
#include <std_msgs/msg/string.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <rclcpp/qos.hpp>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
#include <moveit_msgs/msg/move_it_error_codes.hpp>


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

    // Print the rotation matrix
    tf2::Matrix3x3 m(q);
    std::cout << name << " rotation matrix:" << std::endl;
    for (int i = 0; i < 3; ++i)
        std::cout << "[" << m[i][0] << " " << m[i][1] << " " << m[i][2] << "]" << std::endl;
}


class RelativePoseCommander : public rclcpp::Node
{
public:
  RelativePoseCommander() : Node("kr70_relative_pose_commander")
  {
    sub_rel_pose_ = this->create_subscription<geometry_msgs::msg::Pose>(
      "relative_pose_cmd", rclcpp::QoS(1).best_effort(),
      std::bind(&RelativePoseCommander::relative_pose_callback, this, std::placeholders::_1));


    // this topic is used move the robot to pre-defined locations 
    send_to_sub_ = this->create_subscription<std_msgs::msg::String>(
      "send_to",
      10,
      std::bind(&RelativePoseCommander::send_to_callback, this, std::placeholders::_1));

    rclcpp::QoS qos_profile(1);
    qos_profile.transient_local();   // keep last message for late joiners
    qos_profile.reliable();          // ensure reliable delivery
    qos_profile.keep_last(1);

    // this topic will pulish a movement complete message when the movment is complete
    move_done_pub_ = this->create_publisher<std_msgs::msg::String>(
        "move_done", qos_profile);
  }

  void initialize_move_group()
  {
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), "manipulator");
    
    move_group_->setPlanningPipelineId("pilz");
    move_group_->setPlannerId("LIN");
    move_group_->setMaxVelocityScalingFactor(0.1);
    move_group_->setMaxAccelerationScalingFactor(0.01);
    move_group_->setPlanningTime(5.0);
    move_group_->setNumPlanningAttempts(10);
    move_group_->setStartStateToCurrentState();
    move_group_->setEndEffectorLink("flange");

    std::vector<std::string> link_names = move_group_->getLinkNames();
    for (const auto& ln : link_names){
        RCLCPP_INFO(this->get_logger(), "Link: %s", ln.c_str());
    }

    // auto param_names = this->list_parameters({}, 10).names;
    // for (const auto& n : param_names)
    //     RCLCPP_INFO(this->get_logger(), "Parameter: %s", n.c_str());

    // 1. Get current flange pose
    geometry_msgs::msg::PoseStamped curr_flange_pose = move_group_->getCurrentPose("flange");

    // 2. Convert current and relative pose to tf2::Transform
    tf2::Transform curr_tf, rel_tf;
    tf2::fromMsg(curr_flange_pose.pose, curr_tf);

  }
private:
  void relative_pose_callback(const geometry_msgs::msg::Pose::SharedPtr rel_pose_msg)
  {
   
    // --- clear trajectory info ---
    move_group_->clearPoseTargets();
    move_group_->clearPathConstraints();
    move_group_->setStartStateToCurrentState();

    // --- first setup goal pose ---
      // 1. Get current flange pose
      geometry_msgs::msg::PoseStamped curr_flange_pose = move_group_->getCurrentPose("flange");

      // 2. Convert current and relative pose to tf2::Transform
      tf2::Transform curr_tf, rel_tf;
      tf2::fromMsg(curr_flange_pose.pose, curr_tf);
      tf2::fromMsg(*rel_pose_msg, rel_tf);

      printTransform(curr_tf, "Current TF");
      printTransform(rel_tf, "Relative TF");

      // 3. Compose: next_goal = curr_flange * rel_tf
      tf2::Transform goal_tf = curr_tf * rel_tf;
      geometry_msgs::msg::Pose goal_pose;
      goal_pose.position.x = goal_tf.getOrigin().x();
      goal_pose.position.y = goal_tf.getOrigin().y();
      goal_pose.position.z = goal_tf.getOrigin().z();
      goal_pose.orientation = tf2::toMsg(goal_tf.getRotation());

      printTransform(goal_tf, "Goal TF");

    // now compute the path using cartesian path + time parameterization (this is good for small movements) 
      // 1) Build waypoints from current -> goal
      std::vector<geometry_msgs::msg::Pose> wps;
      wps.push_back(curr_flange_pose.pose); // reminder : current pose of the flange 
      wps.push_back(goal_pose);

      // 2) Compute straight-line EEF path
      moveit_msgs::msg::RobotTrajectory traj_msg;
      moveit::core::RobotState start_state(*move_group_->getCurrentState());
      move_group_->setStartState(start_state);

      // eef_step ~ 5 mm, jump_threshold 0 = disabled
      const double eef_step = 0.010;     // 10 mm
      const double jump_thr = 0.0;       // disabled
      const bool   avoid_collisions = true;

      double fraction = move_group_->computeCartesianPath(
          wps, eef_step, jump_thr, traj_msg, avoid_collisions);

      // --- Basic trajectory sanity check ---
      if (fraction < 0.99 || traj_msg.joint_trajectory.points.size() < 2) {
        auto msg = std_msgs::msg::String();
        msg.data = "FAILURE";
        move_done_pub_->publish(msg);
        RCLCPP_WARN(this->get_logger(),
                    "Cartesian path incomplete: fraction=%.3f, points=%zu. Aborting execution.",
                    fraction, traj_msg.joint_trajectory.points.size());
        return;
      }

      // 3) Time-parameterize for smoothness (TOTG)
      robot_trajectory::RobotTrajectory traj(
          move_group_->getCurrentState()->getRobotModel(), move_group_->getName());
      traj.setRobotTrajectoryMsg(*move_group_->getCurrentState(), traj_msg);

      trajectory_processing::TimeOptimalTrajectoryGeneration totg;
      bool totg_ok = totg.computeTimeStamps(
        traj,
        0.1 /*max_vel_scaling*/,
        0.1 /*max_acc_scaling*/
      );

      // --- Check for TOTG failure ---
      if (!totg_ok) {
        auto msg = std_msgs::msg::String();
        msg.data = "FAILURE";
        move_done_pub_->publish(msg);
        RCLCPP_ERROR(this->get_logger(), "Time parameterization (TOTG) failed. Aborting execution.");
        return;
      }

      traj.getRobotTrajectoryMsg(traj_msg);

    // Execute
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    plan.trajectory_ = traj_msg;
  
    RCLCPP_INFO(this->get_logger(), "Relative move planned. Executing...");
    // blocking while pla executes - key an eye on the bloody robot while here
    auto exec_code = move_group_->execute(plan);   

    // Check either enum or the numeric code:
    bool exec_success =
        (exec_code == moveit::core::MoveItErrorCode::SUCCESS) ||
        (exec_code.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS);

    if (exec_success) {
      auto msg = std_msgs::msg::String();
      msg.data = "COMPLETE";
      move_done_pub_->publish(msg);
    } else {
      auto msg = std_msgs::msg::String();
      msg.data = "FAILURE";
      move_done_pub_->publish(msg);
      RCLCPP_ERROR(this->get_logger(), "Execution failed (code=%d).", exec_code.val);
    }
    
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
    // clear trajectory info 
    move_group_->clearPoseTargets();
    move_group_->clearPathConstraints();
    move_group_->setStartStateToCurrentState();
    
    // Set joint names in MoveIt order (as in your SRDF/URDF group)
    std::vector<std::string> joint_names = {
        "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
    };

    // Set positions in the same order as above - this is somehow buggered in the joint state topics 
    std::vector<double> joint_positions = {
        0.6335370651814217,    // joint_1
        -0.7664526143670498,   // joint_2
        1.3098277157196965,    // joint_3 (from data: index 4)
        -1.157045083658617,    // joint_4 (index 2)
        2.165536468033736,     // joint_5 (index 3)
        3.8152792728840925     // joint_6
    };

    // Assign values to the correct joint names
    std::map<std::string, double> target_joint_values;
    for (size_t i = 0; i < joint_names.size(); ++i)
        target_joint_values[joint_names[i]] = joint_positions[i];

    move_group_->setJointValueTarget(target_joint_values);

    moveit::planning_interface::MoveGroupInterface::Plan my_plan;
    bool success = (move_group_->plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success)
    {
      move_group_->execute(my_plan);
      auto msg = std_msgs::msg::String();
      msg.data = "COMPLETE";
      move_done_pub_->publish(msg);
    }
    else
    {
      auto msg = std_msgs::msg::String();
      msg.data = "FAILURE";
      move_done_pub_->publish(msg);RCLCPP_ERROR(this->get_logger(), "Failed to plan move to start position!");
    }
  }

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr sub_rel_pose_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr send_to_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr move_done_pub_;

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
