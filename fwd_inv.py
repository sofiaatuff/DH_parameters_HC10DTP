import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def forward_kinematics(theta):
    dh_params = np.array([[0.275, 0., 0., 0.],
                          [0, 0.301, 0.5 * pi, 0.5 * pi],
                          [0., 0.700, 0., 0.5 * pi],
                          [0.190, 0., 0.5 * pi, pi],
                          [0.500, 0., -0.5 * pi, 0.],
                          [0.162, 0., 0.5 * pi, 0.]])
    
    num_joints = dh_params.shape[0]
    T = np.eye(4)

    for i in range(num_joints):
        a, d, alpha, theta_offset = dh_params[i]
        theta_i = theta[i] + theta_offset
        A = np.array([[cos(theta_i), -sin(theta_i) * cos(alpha), sin(theta_i) * sin(alpha), a * cos(theta_i)],
                      [sin(theta_i), cos(theta_i) * cos(alpha), -cos(theta_i) * sin(alpha), a * sin(theta_i)],
                      [0, sin(alpha), cos(alpha), d],
                      [0, 0, 0, 1]])
        T = np.matmul(T, A)

    return T[:3, 3]


def inverse_kinematics(target_pos):
    dh_params = np.array([[0.275, 0., 0., 0.],
                          [0, 0.301, 0.5 * pi, 0.5 * pi],
                          [0., 0.700, 0., 0.5 * pi],
                          [0.190, 0., 0.5 * pi, pi],
                          [0.500, 0., -0.5 * pi, 0.],
                          [0.162, 0., 0.5 * pi, 0.]])
    
    num_joints = dh_params.shape[0]
    theta = np.zeros(num_joints)

    for _ in range(100):
        current_pos = forward_kinematics(theta)
        error = target_pos - current_pos
        if np.linalg.norm(error) < 1e-6:
            break

        J = np.zeros((3, num_joints))

        for i in range(num_joints):
            a, d, alpha, theta_offset = dh_params[i]
            theta_i = theta[i] + theta_offset
            A = np.array([[cos(theta_i), -sin(theta_i) * cos(alpha), sin(theta_i) * sin(alpha), a * cos(theta_i)],
                          [sin(theta_i), cos(theta_i) * cos(alpha), -cos(theta_i) * sin(alpha), a * sin(theta_i)],
                          [0, sin(alpha), cos(alpha), d],
                          [0, 0, 0, 1]])

            J[:, i] = np.cross(A[:3, 2], current_pos - A[:3, 3])

        theta += np.linalg.pinv(J) @ error

    return theta


def visualize_robot(theta):
    dh_params = np.array([[0.275, 0., 0., 0.],
                          [0, 0.301, 0.5 * pi, 0.5 * pi],
                          [0., 0.700, 0., 0.5 * pi],
                          [0.190, 0., 0.5 * pi, pi],
                          [0.500, 0., -0.5 * pi, 0.],
                          [0.162, 0., 0.5 * pi, 0.]])
    
    num_joints = dh_params.shape[0]
    T = np.eye(4)
    positions = []

    for i in range(num_joints):
        a, d, alpha, theta_offset = dh_params[i]
        theta_i = theta[i] + theta_offset
        A = np.array([[cos(theta_i), -sin(theta_i) * cos(alpha), sin(theta_i) * sin(alpha), a * cos(theta_i)],
                      [sin(theta_i), cos(theta_i) * cos(alpha), -cos(theta_i) * sin(alpha), a * sin(theta_i)],
                      [0, sin(alpha), cos(alpha), d],
                      [0, 0, 0, 1]])
        T = np.matmul(T, A)
        positions.append(T[:3, 3])

    positions = np.array(positions)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'bo-')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Robot Arm')
    plt.show()


# Example usage:
target_position = np.array([0.9, 0.9, 0.9])
joint_angles = inverse_kinematics(target_position)
print("Joint angles:", joint_angles)
end_effector_pos = forward_kinematics(joint_angles)
print("End-effector position:", end_effector_pos)
visualize_robot(joint_angles)
