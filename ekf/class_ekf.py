import numpy as np

class EKF:
    def __init__(self):
        self.x = np.zeros(3)
        self.P = np.eye(3) * 0.1
        self.Q = np.eye(3) * 0.05
        self.R = np.eye(3) * 0.3

    def predict(self, u):
        self.x[0] += u[0] * np.cos(self.x[2])
        self.x[1] += u[1] * np.sin(self.x[2])
        self.x[2] += u[2]

        A = np.array([
            [1, 0, -u[0] * np.sin(self.x[2])],
            [0, 1,  u[1] * np.cos(self.x[2])],
            [0, 0, 1]
        ])

        self.P = A @ self.P @ A.T + self.Q

    def update(self, z):
        H = np.eye(3)

        y = z - self.x
        S = H @ self.P @ H.T + self.R

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ H) @ self.P