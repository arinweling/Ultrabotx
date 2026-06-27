#!/usr/bin/env python3
import sys
import argparse
import threading
import time
from collections import deque
import numpy as np

# Import ROS 2 python client
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Imu
except ImportError:
    print("\nError: ROS 2 python client (rclpy) or sensor_msgs is not sourced/installed.")
    print("Please source ROS 2 first: source /opt/ros/jazzy/setup.bash\n")
    sys.exit(1)

def quaternion_to_euler(w, x, y, z):
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

def quaternion_to_rotation_matrix(w, x, y, z):
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm > 0:
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    return R

class Imu3DVisualizer(Node):
    def __init__(self, use_gui=True):
        super().__init__('imu_3d_visualizer')
        
        self.lock = threading.Lock()
        self.use_gui = use_gui
        self.msg_count = 0
        self.latest_q = (1.0, 0.0, 0.0, 0.0) # w, x, y, z
        self.latest_rpy = (0.0, 0.0, 0.0)

        # Base Cuboid Vertices (centered, 1.6 x 2.8 x 0.6)
        self.base_vertices = np.array([
            [-0.8, -1.4, -0.3], # 0
            [ 0.8, -1.4, -0.3], # 1
            [ 0.8,  1.4, -0.3], # 2
            [-0.8,  1.4, -0.3], # 3
            [-0.8, -1.4,  0.3], # 4
            [ 0.8, -1.4,  0.3], # 5
            [ 0.8,  1.4,  0.3], # 6
            [-0.8,  1.4,  0.3]  # 7
        ])

        self.faces = [
            [0, 1, 2, 3], # Bottom (Z-)
            [4, 5, 6, 7], # Top (Z+)
            [0, 1, 5, 4], # Front (Y-)
            [2, 3, 7, 6], # Back (Y+)
            [0, 3, 7, 4], # Left (X-)
            [1, 2, 6, 5]  # Right (X+)
        ]

        # Subscription to /imu/data
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        if self.use_gui:
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                self.plt = plt
                self.Poly3DCollection = Poly3DCollection
                self.setup_gui()
            except ImportError:
                print("matplotlib not found, falling back to terminal visualization.")
                self.use_gui = False

    def setup_gui(self):
        self.plt.ion()
        self.fig = self.plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.suptitle("Real-time 3D IMU Orientation", fontsize=14, fontweight='bold')

        # Create 3D PolyCollection for the Cuboid
        self.poly = self.Poly3DCollection([], facecolors='cyan', edgecolors='black', alpha=0.7)
        self.ax.add_collection3d(self.poly)

        # Create lines for body-fixed coordinate axes (Red: X, Green: Y, Blue: Z)
        self.line_x, = self.ax.plot([], [], [], 'r-', linewidth=3.5, label='Body X')
        self.line_y, = self.ax.plot([], [], [], 'g-', linewidth=3.5, label='Body Y')
        self.line_z, = self.ax.plot([], [], [], 'b-', linewidth=3.5, label='Body Z')

        # Axis labeling & limits
        self.ax.set_xlim(-2.5, 2.5)
        self.ax.set_ylim(-2.5, 2.5)
        self.ax.set_zlim(-2.5, 2.5)
        
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.legend()
        self.ax.view_init(elev=20, azim=45)

    def imu_callback(self, msg):
        w = msg.orientation.w
        x = msg.orientation.x
        y = msg.orientation.y
        z = msg.orientation.z
        
        r, p, yaw = quaternion_to_euler(w, x, y, z)

        with self.lock:
            self.msg_count += 1
            self.latest_q = (w, x, y, z)
            self.latest_rpy = (r, p, yaw)

    def update_gui(self):
        with self.lock:
            w, x, y, z = self.latest_q
            r, p, yaw = self.latest_rpy
            msg_cnt = self.msg_count

        if msg_cnt == 0:
            return

        R = quaternion_to_rotation_matrix(w, x, y, z)

        # 1. Rotate the Cuboid Vertices
        rotated_vertices = self.base_vertices @ R.T
        face_verts = [[rotated_vertices[i] for i in face] for face in self.faces]
        self.poly.set_verts(face_verts)

        # 2. Rotate and Draw the Axis lines (length=1.8)
        axes_length = 1.8
        axis_x = R[:, 0] * axes_length
        axis_y = R[:, 1] * axes_length
        axis_z = R[:, 2] * axes_length

        self.line_x.set_data([0, axis_x[0]], [0, axis_x[1]])
        self.line_x.set_3d_properties([0, axis_x[2]])

        self.line_y.set_data([0, axis_y[0]], [0, axis_y[1]])
        self.line_y.set_3d_properties([0, axis_y[2]])

        self.line_z.set_data([0, axis_z[0]], [0, axis_z[1]])
        self.line_z.set_3d_properties([0, axis_z[2]])

        # Update title with current angles
        self.ax.set_title(f"Roll: {r:.1f}° | Pitch: {p:.1f}° | Yaw: {yaw:.1f}°", fontsize=11)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_terminal(self):
        with self.lock:
            w, x, y, z = self.latest_q
            r, p, yaw = self.latest_rpy
            msg_cnt = self.msg_count

        if msg_cnt == 0:
            sys.stdout.write("\rWaiting for IMU messages on /imu/data...")
            sys.stdout.flush()
            return

        # Print a clean, formatted block in the terminal
        sys.stdout.write("\033[H\033[J")  # Clear screen and move cursor to home
        sys.stdout.write("=============================================\n")
        sys.stdout.write("   Live 3D Orientation (Terminal Visualizer) \n")
        sys.stdout.write("=============================================\n")
        sys.stdout.write(f"IMU Messages: {msg_cnt}\n\n")
        sys.stdout.write(f"Quaternion: w={w: 5.3f}, x={x: 5.3f}, y={y: 5.3f}, z={z: 5.3f}\n\n")
        sys.stdout.write("Euler Angles:\n")
        sys.stdout.write(f"  Roll (X):  {r: 7.2f}°\n")
        sys.stdout.write(f"  Pitch (Y): {p: 7.2f}°\n")
        sys.stdout.write(f"  Yaw (Z):   {yaw: 7.2f}°\n\n")
        
        # Simple ASCII meters bargraphs
        def make_bar(val, scale=0.15):
            bars = int(abs(val) * scale)
            bars = min(bars, 20)
            sign = "+" if val >= 0 else "-"
            return sign + ("#" * bars) + (" " * (20 - bars))

        sys.stdout.write(f"Roll:  [{make_bar(r)}]\n")
        sys.stdout.write(f"Pitch: [{make_bar(p)}]\n")
        sys.stdout.write(f"Yaw:   [{make_bar(yaw)}]\n")
        sys.stdout.write("\nPress Ctrl+C to quit.\n")
        sys.stdout.flush()

def main(args=None):
    parser = argparse.ArgumentParser(description="Visualize live 3D ROS 2 IMU orientation.")
    parser.add_argument('--terminal', action='store_true', help="Force terminal-only text mode (no GUI).")
    parsed_args, unknown = parser.parse_known_args()

    rclpy.init(args=args)
    
    use_gui = not parsed_args.terminal
    visualizer = Imu3DVisualizer(use_gui=use_gui)

    # Spin rclpy in a background thread so the GUI or CLI can run in the main thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(visualizer,), daemon=True)
    spin_thread.start()

    try:
        if visualizer.use_gui:
            import matplotlib.pyplot as plt
            # Main thread handles matplotlib GUI updates
            while rclpy.ok():
                visualizer.update_gui()
                plt.pause(0.03)  # ~30 FPS
        else:
            # Main thread handles terminal rendering updates
            while rclpy.ok():
                visualizer.update_terminal()
                time.sleep(0.1)  # 10 Hz refresh
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
