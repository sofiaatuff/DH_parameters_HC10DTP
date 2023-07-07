import numpy as np

# Denavit-Hartenberg parameters for the robot
d = [0.275, 0, 0, 0.19, 0.5, 0.162]
a = [0, 0.301, 0.7, 0, 0, 0]
alpha = [0, np.pi/2, 0, np.pi/2, -np.pi/2, np.pi/2]

def forward_kinematics(theta):
    """
    Perform forward kinematics to compute the end-effector position and orientation.
    :param theta: Joint angles [theta1, theta2, theta3, theta4, theta5, theta6]
    :return: Homogeneous transformation matrix representing the end-effector pose
    """
    T = np.eye(4)
    for i in range(6):
        A = np.array([[np.cos(theta[i]), -np.sin(theta[i])*np.cos(alpha[i]), np.sin(theta[i])*np.sin(alpha[i]), a[i]*np.cos(theta[i])],
                      [np.sin(theta[i]), np.cos(theta[i])*np.cos(alpha[i]), -np.cos(theta[i])*np.sin(alpha[i]), a[i]*np.sin(theta[i])],
                      [0, np.sin(alpha[i]), np.cos(alpha[i]), d[i]],
                      [0, 0, 0, 1]])
        T = np.dot(T, A)
    return T

def inverse_kinematics(T):
    """
    Perform inverse kinematics to compute the joint angles given the desired end-effector pose.
    :param T: Homogeneous transformation matrix representing the end-effector pose
    :return: Joint angles [theta1, theta2, theta3, theta4, theta5, theta6]
    """
    theta = np.zeros(6)
    P = T[:3, 3]
    R = T[:3, :3]
    
    # Wrist center position
    P_wc = P - 0.1 * R[:, 2]
    
    # Joint 1 angle (theta1)
    theta[0] = np.arctan2(P_wc[1], P_wc[0])
    
    # Joint 3 angle (theta3)
    l = np.linalg.norm(P_wc[:2])
    h = P_wc[2] - d[0]
    D = (l**2 + h**2 - a[1]**2 - a[2]**2) / (2 * a[1] * a[2])
    theta[2] = np.arctan2(-np.sqrt(1 - D**2), D)
    
    # Joint 2 angle (theta2)
    alpha = np.arctan2(h, l)
    beta = np.arctan2(a[2] * np.sin(theta[2]), a[1] + a[2] * np.cos(theta[2]))
    theta[1] = alpha - beta
    
    # Joint 4 angle (theta4)
    R03 = np.dot(np.linalg.inv(forward_kinematics(theta[:3]))[:3, :3], R)
    theta[3] = np.arctan2(R03[1, 2], R03[0, 2])
    
    # Joint 5 angle (theta5)
    theta[4] = np.arctan2(np.sqrt(1 - R03[2, 2]**2), R03[2, 2])
    
    # Joint 6 angle (theta6)
    theta[5] = np.arctan
