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
    """
    Конвертирует кватернион в угол рысканья (yaw)
    """
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

class CitrusFarmDataAdapter:
    def __init__(self, bag_path: Path):
        self.bag_path = Path(bag_path)
        self.typestore = get_typestore(Stores.ROS1_NOETIC)
        
        # Для predict от IMU
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_imu_ts = None
        
        # Для хранения последнего угла
        self.last_theta = 0.0

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

                # ============================================
                # 1. ODOMETRY — ТОЛЬКО UPDATE (позиция + угол)
                # ============================================
                if connection.topic == odom_topic:
                    try:
                        # Позиция
                        x = msg.pose.pose.position.x
                        y = msg.pose.pose.position.y
                        
                        # Угол из кватерниона
                        qx = msg.pose.pose.orientation.x
                        qy = msg.pose.pose.orientation.y
                        qz = msg.pose.pose.orientation.z
                        qw = msg.pose.pose.orientation.w
                        theta = quaternion_to_yaw(qx, qy, qz, qw)
                        
                        # Полное измерение [x, y, theta]
                        z = np.array([x, y, theta])
                        yield 'update', z, timestamp
                        
                    except Exception as e:
                        print(f"Ошибка парсинга odom: {e}")
                        continue

                # ============================================
                # 2. IMU — ТОЛЬКО PREDICT (угловая скорость + ускорение)
                # ============================================
                elif connection.topic == imu_topic:
                    try:
                        # Угловая скорость
                        omega_z = msg.angular_velocity.z
                        
                        # Линейное ускорение
                        ax = msg.linear_acceleration.x
                        ay = msg.linear_acceleration.y
                        
                        # Вычисляем DT
                        if self.last_imu_ts is not None:
                            dt = (timestamp - self.last_imu_ts) / 1e9
                            
                            if 0 < dt < 0.1:
                                # Интегрируем ускорение в скорость
                                self.last_vx += ax * dt
                                self.last_vy += ay * dt
                                
                                # Ограничиваем скорость (чтобы не улетела)
                                MAX_SPEED = 5.0
                                self.last_vx = np.clip(self.last_vx, -MAX_SPEED, MAX_SPEED)
                                self.last_vy = np.clip(self.last_vy, -MAX_SPEED, MAX_SPEED)
                                
                                # Вычисляем приращения
                                u = get_u(self.last_vx, self.last_vy, omega_z, dt)
                                yield 'predict', u, timestamp
                        
                        self.last_imu_ts = timestamp
                        
                    except Exception as e:
                        print(f"Ошибка парсинга imu: {e}")
                        continue

                # ============================================
                # 3. GROUND TRUTH (для сравнения)
                # ============================================
                elif connection.topic == gt_topic:
                    try:
                        x = msg.pose.pose.position.x
                        y = msg.pose.pose.position.y
                        
                        # Угол из кватерниона
                        qx = msg.pose.pose.orientation.x
                        qy = msg.pose.pose.orientation.y
                        qz = msg.pose.pose.orientation.z
                        qw = msg.pose.pose.orientation.w
                        theta = quaternion_to_yaw(qx, qy, qz, qw)
                        
                        z = np.array([x, y, theta])
                        yield 'ground_truth', z, timestamp
                        
                    except Exception as e:
                        print(f"Ошибка парсинга ground_truth: {e}")
                        continue