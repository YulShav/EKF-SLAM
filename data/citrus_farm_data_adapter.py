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
    def __init__(self, bag_path: Path, use_every_n_rtk: int = 5):
        self.bag_path = Path(bag_path)
        self.typestore = get_typestore(Stores.ROS1_NOETIC)
        self.use_every_n_rtk = use_every_n_rtk

        self.last_odom_ts = None
        self.last_omega_z = 0.0
        self.rtk_counter = 0

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
                        vx = msg.twist.twist.linear.x
                        vy = msg.twist.twist.linear.y

                        if self.first_odom is None:
                            x = msg.pose.pose.position.x
                            y = msg.pose.pose.position.y
                            self.first_odom = np.array([x, y])
                        
                        if self.last_odom_ts is not None:
                            dt = (timestamp - self.last_odom_ts) / 1e9
                            if 0 < dt < 0.1:
                                u = get_u(vx, vy, self.last_omega_z, dt)
                                yield 'predict', u, timestamp
                        
                        self.last_odom_ts = timestamp
                        
                    except Exception as e:
                        continue

                elif connection.topic == imu_topic:
                    try:
                        self.last_omega_z = msg.angular_velocity.z
                        
                    except Exception as e:
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

                        self.rtk_counter += 1
                        
                        if self.rtk_counter % self.use_every_n_rtk == 0:
                            yield 'update', z, timestamp
                        else:
                            yield 'ground_truth', z, timestamp
                        
                    except Exception as e:
                        continue