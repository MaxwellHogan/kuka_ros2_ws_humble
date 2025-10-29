#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import String


class MoveDoneListener(Node):
    def __init__(self):
        super().__init__('move_done_listener')

        # Match QoS used in C++ publisher
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            String,
            'move_done',
            self.listener_callback,
            qos_profile
        )

        self.get_logger().info("Subscribed to /move_done")

    def listener_callback(self, msg):
        self.get_logger().info(f"Received move_done message: {msg.data}")


def main(args=None):
    rclpy.init(args=args)
    node = MoveDoneListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
