#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

class StringRepublisher : public rclcpp::Node
{
public:
  StringRepublisher()
  : Node("string_republisher")
  {
    // Parameters so you can change topics without recompiling
    this->declare_parameter<std::string>("input_topic", "/KUKA_GRASP_flags");
    this->declare_parameter<std::string>("output_topic", "/GRASP_flags");

    const auto input_topic = this->get_parameter("input_topic").as_string();
    const auto output_topic = this->get_parameter("output_topic").as_string();

    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();

    publisher_ = this->create_publisher<std_msgs::msg::String>(output_topic, qos);

    RCLCPP_INFO(this->get_logger(), "Hello!");

    subscription_ = this->create_subscription<std_msgs::msg::String>(
      input_topic,
      10,
      [this](const std_msgs::msg::String::SharedPtr msg)
      {
        // Republish exactly the same message contents
        std_msgs::msg::String out;
        out.data = msg->data;
        publisher_->publish(out);

        RCLCPP_INFO(this->get_logger(), "Heard: '%s' -> republished", msg->data.c_str());
      });

    RCLCPP_INFO(this->get_logger(), "Republishing %s -> %s",
                input_topic.c_str(), output_topic.c_str());
  }

private:
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<StringRepublisher>());
  rclcpp::shutdown();
  return 0;
}
