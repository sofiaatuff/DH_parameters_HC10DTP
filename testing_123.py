import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def forward_kinematics(dh_params):
    num_joints = dh_params.shape[0]
    T = np.eye(4)

    positions = [T[:3, 3]]

    for i in range(num_joints):
        a, d, alpha, theta = dh_params[i]
        A = np.array([[cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
                      [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
                      [0, sin(alpha), cos(alpha), d],
                      [0, 0, 0, 1]])
        T = np.matmul(T, A)
        positions.append(T[:3, 3])

    return positions

# DH parameters
dh_params = np.array([[0.275, 0., 0., 0.],
                      [0, 0.301, 0.5 * pi, 0.5 * pi],
                      [0., 0.700, 0., 0.5 * pi],
                      [0.190, 0., 0.5 * pi, pi],
                      [0.500, 0., -0.5 * pi, 0.],
                      [0.162, 0., 0.5 * pi, 0.]])

# Calculate forward kinematics
end_effector_positions = forward_kinematics(dh_params)

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting the robot arm
for i in range(len(end_effector_positions) - 1):
    ax.plot([end_effector_positions[i][0], end_effector_positions[i+1][0]],
            [end_effector_positions[i][1], end_effector_positions[i+1][1]],
            [end_effector_positions[i][2], end_effector_positions[i+1][2]], 'bo-')

# Set plot labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Robot Arm')

# Show the plot
plt.show()
