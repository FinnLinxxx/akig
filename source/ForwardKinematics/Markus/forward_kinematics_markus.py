import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def tf_matrix_dh(theta, d, a, alpha):
    """
    Returns the homogeneous TF Matrix based on the four Denavit-Hartenberg parameters.
    :param theta: Angle between the x-axes around the old z-axis. Given in radians
    :param d: Distance between the origins of the two consecutive frames along the previous z-axis
    :param a: Distance between the origins along the x-axis of the next frame
    :param alpha: Angle between the z-axes around the previous x-axis
    :return: homogeneous TF matrix as 2D numpy array
    """
    return np.array([[cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
                     [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
                     [0, sin(alpha), cos(alpha), d],
                     [0, 0, 0, 1]])


def denavit_hartberg_tf_matrix(joint_pos, dh_params):
    """
    Iterates over the joint positions of one pose and calls tf_matrix_dh with them. Then it multiplies the returned TF
    matrices to obtain the TF from the last frame to the base frame.
    :param joint_pos: list of joint positions of one pos given in radians. Order: Base to end frame
    :param dh_params: 2D Array with the Denavit-Hartenberg calibration parameters (a, d, alpha). One row per
    joint in the order of base to end frame. Columns in the order of a, d, alpha. Alpha given in radians.
    :return: TF matrix for homogeneous coordinates from the end frame to the base frame.
    """
    tf = np.eye(4)
    for i in range(joint_pos.size):
        tf = tf @ tf_matrix_dh(joint_pos[i], dh_params[i, 1], dh_params[i, 0],
                               dh_params[i, 2])
    return tf


def forward_kinematic(joint_pos, timestamps=None, params=None):
    """
    Calculates the origin of the end frame in the base frame as well as the the rotation matrix between the two frames.
    If no params are given the default parameters for the UR5 as specified by the manufacturer are used.
    :param joint_pos: list of joint positions of one pos given in radians. Order: Base to end frame
    :param timestamps: optional. List od timestamps corresponding to the individual poses
    :param params: optional. 2D Array with the Denavit-Hartenberg calibration parameters (a, d, alpha). One row per
    joint in the order of base to end frame. Columns in the order of a, d, alpha. Alpha given in radians.
    :return: 2D list containing: Timestamps if given, origin of the end frame in the base frame, flattened rotation matrix
    """

    ur5_calibration_parameters = np.array([[0, 0.089159, np.pi / 2],
                                           [-0.425, 0, 0],
                                           [-0.39225, 0, 0],
                                           [0, 0.10915, np.pi / 2],
                                           [0, 0.09465, -np.pi / 2],
                                           [0, 0.0823, 0]])

    if params is None:
        params = ur5_calibration_parameters

    # arrange poses array
    poses = np.zeros((joint_pos.shape[0], 12))  # 3 translation, 9 rotation matrix

    for pose_idx, joint_pos_set in enumerate(joint_pos):
        tf_matrix = denavit_hartberg_tf_matrix(joint_pos_set, params)
        endpoint = tf_matrix[:3, 3]
        rotation_matrix = tf_matrix[0:3, 0:3]

        poses[pose_idx, :3] = endpoint[:3].ravel()
        poses[pose_idx, 3:] = rotation_matrix.flatten()

    if timestamps is not None:
        poses = np.hstack((np.array(timestamps)[:, None], poses))

    return poses


if __name__ == '__main__':
    print('Reading joint positions from text file...')
    ur5_joint_pos_list = np.empty((0, 6))
    file_path = 'UR5_joint_pos_05Pts125hzLAN.txt'
    with open(file_path) as fp:
        line = fp.readline()
        while line:
            values = line.split()
            try:
                joint_pos = [float(val) for val in values]
                if len(joint_pos) == 6:
                    ur5_joint_pos_list = np.append(ur5_joint_pos_list, np.array([joint_pos]), axis=0)
            except ValueError:
                pass

            line = fp.readline()

    print('Calculating forward kinematics...')
    UR5_forward_kinematic = forward_kinematic(ur5_joint_pos_list)

    print('Reading laser tracker points from text file...')
    point_list_LT = np.empty((0, 3))
    file_path = 'coordinates_tracker_output_05Pts125hzLAN.txt'
    with open(file_path) as fp:
        line = fp.readline()
        while line:
            try:
                values = line.split()
                lt_coor = np.array([[float(val)] for val in values]).T
                if len(values) == 3:
                    point_list_LT = np.append(point_list_LT, lt_coor, axis=0)
            except ValueError:
                pass

            line = fp.readline()

    print('Testing forward kinematics algorithm...')
    test = forward_kinematic(np.array([[0.062412720173597336, -0.9297316710101526, 0.5339670181274414,
                                        -1.198200527821676, 1.6025909185409546, -1.4876373449908655]]))
    print(test)

    print('Plotting points...')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(UR5_forward_kinematic[:, 0], UR5_forward_kinematic[:, 1], UR5_forward_kinematic[:, 2])
    ax.set_zlim([0.35, 0.95])
    plt.title('UR5 endframe origins')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim([0.35, 0.95])
    every_nth_point = 25
    ax.quiver(UR5_forward_kinematic[::every_nth_point, 0], UR5_forward_kinematic[::every_nth_point, 1],
              UR5_forward_kinematic[::every_nth_point, 2], UR5_forward_kinematic[::every_nth_point, 3],
              UR5_forward_kinematic[::every_nth_point, 6], UR5_forward_kinematic[::every_nth_point, 9],
              length=0.1, normalize=True, color='red')
    ax.quiver(UR5_forward_kinematic[::every_nth_point, 0], UR5_forward_kinematic[::every_nth_point, 1],
              UR5_forward_kinematic[::every_nth_point, 2], UR5_forward_kinematic[::every_nth_point, 4],
              UR5_forward_kinematic[::every_nth_point, 7], UR5_forward_kinematic[::every_nth_point, 10],
              length=0.1, normalize=True, color='green')
    ax.quiver(UR5_forward_kinematic[::every_nth_point, 0], UR5_forward_kinematic[::every_nth_point, 1],
              UR5_forward_kinematic[::every_nth_point, 2], UR5_forward_kinematic[::every_nth_point, 5],
              UR5_forward_kinematic[::every_nth_point, 8], UR5_forward_kinematic[::every_nth_point, 11],
              length=0.1, normalize=True, color='blue')
    plt.title('UR5 endframe axes')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_list_LT[:, 0], point_list_LT[:, 1], point_list_LT[:, 2])
    plt.title('LT CCR UR5 points')
    ax.set_zlim([0, 0.3])
    plt.show()
