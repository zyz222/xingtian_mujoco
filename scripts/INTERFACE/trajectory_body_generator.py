import numpy as np

class TrajectoryGenerator:
    def __init__(self, dt=0.01, horizon=0.1):
        self.dt = dt
        self.horizon = horizon

    def generate(self, current_pose, desired_velocity):
        steps = int(self.horizon / self.dt)
        traj = np.zeros((steps, 3))
        x, y, yaw = current_pose

        for i in range(steps):
            vx = desired_velocity["vx"]
            vy = desired_velocity["vy"]
            wz = desired_velocity["wz"]
            x += (vx * np.cos(yaw) - vy * np.sin(yaw)) * self.dt
            y += (vx * np.sin(yaw) + vy * np.cos(yaw)) * self.dt
            yaw += wz * self.dt
            traj[i] = [x, y, yaw]

        return traj
