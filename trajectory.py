from visual_kinematics.RobotSerial import *
from visual_kinematics.RobotTrajectory import *
import numpy as np
from math import pi


def main():
    np.set_printoptions(precision=3, suppress=True)

    dh_params = np.array([[0.275, 0, 0, 0.5 * pi],
                          [0, 0.190, 0, 0.5 * pi],
                          [0.700, 0, 0.5 * pi, -0.5 * pi],
                          [0.190, 0,  -0.5 * pi, pi],
                          [0.500, 0, 0.5 * pi, 0 ],
                          [0.162, 0., 0.5 * pi, 0.5 * pi]])
    robot = RobotSerial(dh_params)

    # =====================================
    # trajectory
    # =====================================

    frames = [Frame.from_euler_3(np.array([0.5 * pi, 0., pi]), np.array([[0.28127], [0.], [1.03182]])),
              Frame.from_euler_3(np.array([0.25 * pi, 0., 0.75 * pi]), np.array([[0.28127], [0.], [0.63182]])),
              Frame.from_euler_3(np.array([0.5 * pi, 0., pi]), np.array([[0.28127], [0.], [0.43182]]))]
    time_points = np.array([0., 8., 8.])
    trajectory = RobotTrajectory(robot, frames, time_points)
    trajectory.show(motion="p2p")


if __name__ == "__main__":
    main()