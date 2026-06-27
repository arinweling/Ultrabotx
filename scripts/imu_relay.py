#!/usr/bin/env python3
import sys
import socket
import struct

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Imu
except ImportError:
    print("\nError: ROS 2 python client (rclpy) or sensor_msgs is not sourced/installed in this shell.")
    print("Please source ROS 2 first: source /opt/ros/jazzy/setup.bash\n")
    sys.exit(1)

class ImuRelay(Node):
    def __init__(self):
        super().__init__('imu_relay')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target_addr = ('127.0.0.1', 12345)
        self.subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        self.get_logger().info("IMU Relay Node started. Forwarding to 127.0.0.1:12345 via UDP...")

    def imu_callback(self, msg):
        # Pack w, x, y, z as 4 doubles (8 bytes each -> 32 bytes)
        data = struct.pack('4d', msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z)
        try:
            self.sock.sendto(data, self.target_addr)
        except Exception:
            pass

def main():
    rclpy.init()
    node = ImuRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
