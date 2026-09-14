import numpy as np
from pathlib import Path
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
import math

def get_u(vx, vy, omega_z, dt):
    dx = vx * dt
    dy = vy * dt
    dtheta = omega_z * dt
    return np.array([dx, dy, dtheta])

def quaternion_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

class CitrusFarmDataAdapter:
    def __init__(self, bag_path: Path):
        self.bag_path = Path(bag_path)
        self.typestore = get_typestore(Stores.ROS1_NOETIC)

        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_imu_ts = None

        self.first_odom = None
        self.first_gt = None

    def __iter__(self):
        with Reader(self.bag_path) as reader:
            odom_topic = "/jackal_velocity_controller/odom"
            imu_topic = "/microstrain/imu/data"
            gt_topic = "/gps/fix/odometry"

            wanted_topics = {odom_topic, imu_topic, gt_topic}

            for connection, timestamp, data in reader.messages():
                if connection.topic not in wanted_topics:
                    continue

                try:
                    msg = self.typestore.deserialize_ros1(data, connection.msgtype)
                except KeyError:
                    continue

                if connection.topic == odom_topic:
                    try:
                        x = msg.pose.pose.position.x
                        y = msg.pose.pose.position.y
                        
                        qx = msg.pose.pose.orientation.x
                        qy = msg.pose.pose.orientation.y
                        qz = msg.pose.pose.orientation.z
                        qw = msg.pose.pose.orientation.w
                        theta = quaternion_to_yaw(qx, qy, qz, qw)

                        if self.first_odom is None:
                            self.first_odom = np.array([x, y])
                        
                        z = np.array([x, y, theta])
                        yield 'update', z, timestamp
                        
                    except Exception:
                        continue

                elif connection.topic == imu_topic:
                    try:
                        omega_z = msg.angular_velocity.z
                        ax = msg.linear_acceleration.x
                        ay = msg.linear_acceleration.y
                        
                        if self.last_imu_ts is not None:
                            dt = (timestamp - self.last_imu_ts) / 1e9
                            
                            if 0 < dt < 0.1:
                                self.last_vx += ax * dt
                                self.last_vy += ay * dt

                                MAX_SPEED = 5.0
                                self.last_vx = np.clip(self.last_vx, -MAX_SPEED, MAX_SPEED)
                                self.last_vy = np.clip(self.last_vy, -MAX_SPEED, MAX_SPEED)
                                
                                u = get_u(self.last_vx, self.last_vy, omega_z, dt)
                                yield 'predict', u, timestamp
                        
                        self.last_imu_ts = timestamp
                        
                    except Exception:
                        continue

                elif connection.topic == gt_topic:
                    try:
                        x = msg.pose.pose.position.x
                        y = msg.pose.pose.position.y
                        
                        qx = msg.pose.pose.orientation.x
                        qy = msg.pose.pose.orientation.y
                        qz = msg.pose.pose.orientation.z
                        qw = msg.pose.pose.orientation.w
                        theta = quaternion_to_yaw(qx, qy, qz, qw)

                        if self.first_gt is None:
                            self.first_gt = np.array([x, y])
                        
                        if self.first_odom is not None and self.first_gt is not None:
                            offset_x = self.first_odom[0] - self.first_gt[0]
                            offset_y = self.first_odom[1] - self.first_gt[1]
                            
                            x_aligned = x + offset_x
                            y_aligned = y + offset_y
                        else:
                            x_aligned = x
                            y_aligned = y
                        
                        z = np.array([x_aligned, y_aligned, theta])
                        yield 'ground_truth', z, timestamp
                        
                    except Exception:
                        continue