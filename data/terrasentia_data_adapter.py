import numpy as np
from pathlib import Path
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

def get_u(vx, vy, omega_z, dt):
    dx = vx * dt
    dy = vy * dt
    dtheta = omega_z * dt
    return np.array([dx, dy, dtheta])

class TerrasentiaDataAdapter:
    def __init__(self, bag_path: Path):
        self.bag_path = Path(bag_path)
        self.typestore = get_typestore(Stores.ROS1_NOETIC)
        self.last_vx = None
        self.last_vy = None
        self.last_ts = None

    def __iter__(self):
        with Reader(self.bag_path) as reader:
            # Топики как в вашем баге
            odom_topic = "/zed2_front/zed_node/odom"
            imu_topic = "/terrasentia/imu"
            path_map_topic = "/zed2_front/zed_node/path_map"

            wanted_topics = {odom_topic, imu_topic, path_map_topic}

            for connection, timestamp, data in reader.messages():
                if connection.topic not in wanted_topics:
                    continue

                msg = self.typestore.deserialize_ros1(data, connection.msgtype)
                
                # ===== 1. Одометрия (как источник скорости для predict) =====
                if connection.topic == odom_topic:
                    # Сохраняем скорость из одометрии
                    self.last_vx = msg.twist.twist.linear.x
                    self.last_vy = msg.twist.twist.linear.y
                    
                    # ИСПОЛЬЗУЕМ ОДОМЕТРИЮ КАК UPDATE (как в Ford использовали pose_raw)
                    # Но с выравниванием по первой точке ground truth
                    x = msg.pose.pose.position.x
                    y = msg.pose.pose.position.y
                    z = np.array([x, y, 0.0])
                    yield 'update', z, timestamp
                
                # ===== 2. IMU (для predict) =====
                elif connection.topic == imu_topic:
                    omega_z = msg.angular_velocity.z
                    
                    if self.last_ts is not None and self.last_vx is not None:
                        dt = (timestamp - self.last_ts) / 1e9
                        u = get_u(self.last_vx, self.last_vy, omega_z, dt)
                        yield 'predict', u, timestamp
                    
                    self.last_ts = timestamp
                
                # ===== 3. Path Map (как ground truth) =====
                elif connection.topic == path_map_topic:
                    if len(msg.poses) > 0:
                        last_pose = msg.poses[-1]
                        x = last_pose.pose.position.x
                        y = last_pose.pose.position.y
                        z = np.array([x, y, 0.0])
                        yield 'ground_truth', z, timestamp