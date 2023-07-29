from visual_kinematics.RobotSerial import *
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
    # forward
    # =====================================

    theta = np.array([0., 0., 0, 0., 0., 0.])
    f = robot.forward(theta)

    print("-------forward-------")
    print("end frame t_4_4:")
    print(f.t_4_4)
    print("end frame xyz:")
    print(f.t_3_1.reshape([3, ]))
    print("end frame abc:")
    print(f.euler_3)
    print("end frame rotational matrix:")
    print(f.r_3_3)
    print("end frame quaternion:")
    print(f.q_4)
    print("end frame angle-axis:")
    print(f.r_3)

    robot.show()


if __name__ == "__main__":
    main()