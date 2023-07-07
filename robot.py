import numpy as np
import pandas as pd

def calculate_homogeneous_matrix(a, alpha, d, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    # Construct the homogeneous transformation matrix
    homogeneous_matrix = np.array([
        [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
        [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
        [0, sin_alpha, cos_alpha, d],
        [0, 0, 0, 1]
    ])

    return homogeneous_matrix

def calculate_homogeneous_matrices(dh_parameters):
    num_links = len(dh_parameters)
    homogeneous_matrices = []

    # Calculate the homogeneous transformation matrix for each link
    for i in range(num_links):
        a = dh_parameters[i]['a']
        alpha = dh_parameters[i]['alpha']
        d = dh_parameters[i]['d']
        theta = dh_parameters[i]['theta']

        homogeneous_matrix = calculate_homogeneous_matrix(a, alpha, d, theta)
        homogeneous_matrices.append(homogeneous_matrix)

    return homogeneous_matrices


def calculate_robot_homogeneous_matrix(homogeneous_matrices):
    # Multiply the individual homogeneous matrices to get the overall homogeneous matrix
    robot_homogeneous_matrix = np.eye(4)

    for matrix in homogeneous_matrices:
        robot_homogeneous_matrix = np.dot(robot_homogeneous_matrix, matrix)

    return robot_homogeneous_matrix



# DH parameters for Motoman HC10DPT Robot
dh_parameters = [
    {'a': 0, 'alpha': 0, 'd': 0.275, 'theta': 0.1},
    {'a': 0.301, 'alpha': np.pi/2, 'd': 0, 'theta': 0.2 + np.pi/2 },
    {'a': 0.7, 'alpha': 0, 'd': 0, 'theta': 0.3+ np.pi/2},
    {'a': 0, 'alpha': np.pi/2, 'd': 0.19, 'theta': 0.4 + np.pi},
    {'a': 0.0, 'alpha': -np.pi/2, 'd': 0.5, 'theta': 0.5},
    {'a': 0, 'alpha': np.pi/2, 'd': 0.162, 'theta': 0.6}
]


homogeneous_matrices = calculate_homogeneous_matrices(dh_parameters)
robot_homogeneous_matrix = calculate_robot_homogeneous_matrix(homogeneous_matrices)

# Print the DH parameters for the robot
robot = pd.DataFrame(dh_parameters)
print(robot, end='\n')


# Print the homogeneous matrices for each link
for i, matrix in enumerate(homogeneous_matrices):
    print(f"Homogeneous matrix for Link {i+1}:")
    print(matrix)
    print()


# Print the overall homogeneous matrix for the robot
print("Overall Homogeneous Matrix for the Robot:")
print(robot_homogeneous_matrix)