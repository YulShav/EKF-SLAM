import numpy as np
from pathlib import Path
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

def get_u(vx, vy, omega_z, dt):
    dx = vx * dt
    dy = vy * dt
    dtheta = omega_z * dt

    return np.array([dx, dy, dtheta])


class FordDataAdapter:
    def __init__(self, bag_path: Path):
        self.bag_path = Path(bag_path)
        self.typestore = get_typestore(Stores.LATEST)
        self.last_vx = None
        self.last_vy = None
        self.last_ts = None

    def __iter__(self):
        with Reader(self.bag_path) as reader:
            for conn in reader.connections:
                self.typestore.register(conn.msgtype)
            
            for connection, timestamp, data in reader.messages():
                msg = self.typestore.deserialize_ros1(data, connection.msgtype)
                
                if connection.topic == '/velocity_raw':
                    self.last_vx = msg.vector.x
                    self.last_vy = msg.vector.y
                    self.last_ts = timestamp
                
                elif connection.topic == '/imu':
                    omega_z = msg.angular_velocity.z
                    
                    if self.last_ts is not None and self.last_vx is not None:
                        dt = (timestamp - self.last_ts) / 1e9
                        u = get_u(self.last_vx, self.last_vy, omega_z, dt)
                        yield 'predict', u, timestamp
                    
                    self.last_ts = timestamp
                
                elif connection.topic == '/pose_ground_truth':
                    x = msg.pose.position.x
                    y = msg.pose.position.y
                    z = np.array([x, y, 0.0])
                    yield 'update', z, timestamp