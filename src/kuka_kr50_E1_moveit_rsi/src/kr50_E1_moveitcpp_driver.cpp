#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.h>
#include <std_msgs/msg/string.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <rclcpp/qos.hpp>

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
    
  

    // default tool transform - 2 to 1 scale model
    flange_to_tool_.setOrigin(tf2::Vector3(0.2134, 0.5, 0.0));
    flange_to_tool_.setRotation(tf2::Quaternion(0.0, 0.0, 0.0, 1.0));

    // subscriber to set/update tool transform at runtime
    set_tool_end_sub_ = this->create_subscription<std_msgs::msg::String>(
      "set_tool_end",
      rclcpp::QoS(1).reliable().keep_last(1),
      std::bind(&RelativePoseCommander::set_tool_end_callback, this, std::placeholders::_1)
    );

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
    move_group_->setMaxVelocityScalingFactor(0.8);
    move_group_->setMaxAccelerationScalingFactor(0.03);
    move_group_->setPlanningTime(10.0);
    move_group_->setNumPlanningAttempts(20);
    move_group_->setStartStateToCurrentState();
    move_group_->setGoalPositionTolerance(0.001);
    move_group_->setGoalOrientationTolerance(0.00174533);


    // std::vector<std::string> link_names = move_group_->getLinkNames();
    // for (const auto& ln : link_names){
    //     RCLCPP_INFO(this->get_logger(), "Link: %s", ln.c_str());

    auto robot_model_ptr = move_group_->getRobotModel();
    auto joint_models = robot_model_ptr->getActiveJointModels();
    for (const auto& joint_model : joint_models) {
        const auto& bounds = joint_model->getVariableBounds();
        for (const auto& b : bounds) {
            RCLCPP_INFO(this->get_logger(),
                "Joint: %s  min: %f  max: %f",
                joint_model->getName().c_str(), b.min_position_, b.max_position_);
        }
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
    
      // --- clear traj{ectory info ---
    move_group_->clearPoseTargets();
    move_group_->clearPathConstraints();
    move_group_->setStartStateToCurrentState();
    move_group_->setMaxVelocityScalingFactor(0.8); // set velocity scaling factor
    move_group_->setMaxAccelerationScalingFactor(0.03);
    
    // 1. Get current flange pose
    geometry_msgs::msg::PoseStamped curr_flange_pose = move_group_->getCurrentPose("flange");

    // 2. Convert current and relative pose to tf2::Transform
    tf2::Transform curr_tf, rel_tf;
    
    tf2::fromMsg(curr_flange_pose.pose, curr_tf);
    tf2::fromMsg(*rel_pose_msg, rel_tf);

    printTransform(curr_tf, "Current TF");
    printTransform(rel_tf, "Flange Relative TF");

    // calculate how much we need to move the flange by for the tool frame (use current flange->tool)
    tf2::Transform F_T, T_F;
    {
      std::lock_guard<std::mutex> lk(tool_mutex_);
      F_T = flange_to_tool_;
    }
    T_F = F_T.inverse();
    rel_tf = F_T * rel_tf * T_F;

    // 3. Compose: next_goal = curr_flange * rel_tf
    tf2::Transform goal_tf = curr_tf * rel_tf;
    geometry_msgs::msg::Pose goal_pose;
    goal_pose.position.x = goal_tf.getOrigin().x();
    goal_pose.position.y = goal_tf.getOrigin().y();
    goal_pose.position.z = goal_tf.getOrigin().z();
    goal_pose.orientation = tf2::toMsg(goal_tf.getRotation());
    
    printTransform(rel_tf, "Tool Relative TF");
    printTransform(goal_tf, "Goal TF");


    // 4. Plan and execute
    move_group_->setPoseTarget(goal_pose, "flange");

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (move_group_->plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    if (success)
    {
      RCLCPP_INFO(this->get_logger(), "Relative move planned. Executing...");
      move_group_->execute(plan);

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
  }


  // callbck to set the flange to tool transform
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
    // clear trajectory info 
    move_group_->clearPoseTargets();
    move_group_->clearPathConstraints();
    move_group_->setStartStateToCurrentState();
    move_group_->setMaxVelocityScalingFactor(0.1);
    move_group_->setMaxAccelerationScalingFactor(0.005);

    // Set joint names in MoveIt order - ros2 topic echo /joint_states
    std::vector<std::string> joint_names = {
        "A1", "A2", "A3", "A4", "A5", "A6", "E1"
    };

    // Set positions in the same order as above 
    std::vector<double> joint_positions = {
          0.7335758472472327,
          -1.0502588587045958,
          1.8730315027042506,
          -0.9959110511267444,
          2.036048745499025,
          -0.61374328613456,
          -3.591,
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
  // --- tool (TCP) transform management ---
  tf2::Transform flange_to_tool_;   // F_T (flange -> tool), defaults to identity
  std::mutex tool_mutex_;           // guard runtime updates
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr set_tool_end_sub_;

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
