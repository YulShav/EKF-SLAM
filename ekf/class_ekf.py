import numpy as np

class EKF:
    def __init__(self):
        self.x = np.zeros(3)  # [x, y, theta]
        self.P = np.eye(3) * 0.1
        self.Q = np.eye(3) * 0.05
        self.Q[2, 2] = 0.01 
        self.R = np.eye(3) * 0.3

    def predict(self, u):
        """
        u = [dx, dy, dtheta] - приращения
        """
        dx, dy, dtheta = u
        theta = self.x[2]

        self.x[0] += dx * np.cos(theta) - dy * np.sin(theta)
        self.x[1] += dx * np.sin(theta) + dy * np.cos(theta)
        self.x[2] += dtheta

        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        A = np.array([
            [1, 0, -dx * sin_t - dy * cos_t],
            [0, 1,  dx * cos_t - dy * sin_t],
            [0, 0, 1]
        ])
        
        self.P = A @ self.P @ A.T + self.Q

    def update(self, z):
        z = np.array(z).flatten()
        if len(z) == 2:
            z = np.array([z[0], z[1], self.x[2]])
        
        H = np.eye(3)
        y = z - self.x
        y[2] = np.arctan2(np.sin(y[2]), np.cos(y[2]))
        
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))
        self.P = (np.eye(3) - K @ H) @ self.P
    
    def set_pose(self, x):
        x = np.array(x).flatten()
        if len(x) == 2:
            self.x = np.array([x[0], x[1], 0.0])
        else:
            self.x = x
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))