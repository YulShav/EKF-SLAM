import numpy as np
from pathlib import Path
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

def load_data(bag_path):
    bag_path = Path(bag_path)
    typestore = get_typestore(Stores.LATEST)

    timestamps = []
    us = []
    zs = []

    last_time = None
    last_vx = None

    with Reader(bag_path) as reader:
        velocity = None
        ground_truth = None
        
        for connection in reader.connections:
            if connection.topic == '/velocity_raw':
                velocity = connection
            elif connection.topic == '/pose_ground_truth':
                ground_truth = connection

        for connection, timestamp, data in reader.messages():
            msg = typestore.deserialize(data, connection.msgtype)

            if connection.topic == '/velocity_raw':
                vx = msg.vector.x

                if last_time is not None:
                    dt = timestamp - last_time
                    dx = vx * dt
                    dy = 0.0
                    dtheta = 0.0
                    
                    u = np.array([dx, dy, dtheta])
                    us.append(u)
                    timestamps.append(timestamp)
                
                last_time = timestamp
                last_vx = vx

            elif connection.topic == '/pose_ground_truth':
                x = msg.pose.position.x
                y = msg.pose.position.y
                z = np.array([x, y, 0.0])
                zs.append(z)

    return timestamps, us, zs