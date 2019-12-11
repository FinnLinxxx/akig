import time
import os

from numpy import array, sin, cos, zeros, float64, hstack, int64, asarray, vstack, pi
import pandas as pd
from scipy.signal import savgol_filter

from utils.rosutils import load_rosbag_csv
import matplotlib.pyplot as plt

# theta, a, d, alpha for each joint (6x4)
UR5_PARAMS = array([[0, 0, 0.089159, 1.5707963267949],
                    [0, -0.425, 0, 0],
                    [0, -0.39225, 0, 0],
                    [0, 0, 0.10915, 1.5707963267949],
                    [0, 0, 0.09465, -1.5707963267949],
                    [0, 0, 0.0823, 0]])


def forward_kinematic(joints, timestamps=None, params=None):
    """
    Convert Joint positions to cartesian coordinates and rotation matrices
    :param joints: nx6 array of joint readings
    :param params: if None the default UR5 param set is used otherwise 6x4 matrix (theta, a, d, alpha per joint)
    :return:
    """

    # load default UR 5 params if not specified
    if params is None:
        params = UR5_PARAMS

    # arange poses array
    poses = zeros((joints.shape[0], 12), dtype=float64)

    for i, j in enumerate(joints):
        p = params.copy()
        p[:, 0] = j
        T, R, t = denavit_hartb(p)
        poses[i, :3] = t
        poses[i, 3:] = R.flatten()

    if timestamps is not None:
        poses = hstack((array(timestamps)[:, None], poses))

    return poses


def denavit_hartb(params):
    """
    Compute transformation matrix T, translation t and rotation matrix R from DH parameters
    :param params: DH (Denavit-Hartenberg) parameters
    :return: T, R, t
    """
    Ti = []
    for i in range(params.shape[0]):
        A = dh_rot(*params[i, :])
        Ti += [A]

    T = Ti[0] @ Ti[1] @ Ti[2] @ Ti[3] @ Ti[4] @ Ti[5]

    # return T, R, t
    return T, T[:3, :3], T[:3, 3],


def quat2dcm(quat):
    """
    method to compute Direction-Cosine-Matrix from a unit quaternion
    Rotation "direction" is defined by quat (local -> global or vice versa)

    :param quat: 4-vector
    :return: (3, 3) numpy ndarray
    """
    q = asarray(quat).flatten()

    a, b, c, d = q[0], q[1], q[2], q[3]
    c11 = a * a + b * b - c * c - d * d
    c12 = 2 * (b * c - a * d)
    c13 = 2 * (b * d + a * c)
    c21 = 2 * (b * c + a * d)
    c22 = a * a - b * b + c * c - d * d

    c23 = 2 * (c * d - a * b)
    c31 = 2 * (b * d - a * c)
    c32 = 2 * (c * d + a * b)
    c33 = a * a - b * b - c * c + d * d

    return array([[c11, c12, c13],
                  [c21, c22, c23],
                  [c31, c32, c33]])


def dh_rot(theta, a, d, alpha):
    """
    Denavit Hartenberg Rotation Matrix [Robotic, Vision and Control, P. Corke p. 138]

    :param theta: joint angle
    :param a: link length
    :param d: link offset
    :param alpha: link twist
    :return: transformation matrix 4x4
    """

    r = array([[cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
               [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
               [0, sin(alpha), cos(alpha), d],
               [0, 0, 0, 1]])
    return r


def load_and_save_joints(txt_path, target_path=None, plot_timestamps=False, rtde=False, smooth=False,
                         include_vel=False):

    if target_path and os.path.isfile(target_path):
        return pd.read_pickle(target_path)

    if rtde:
        joints = pd.read_csv(txt_path, delimiter=' ')
        urp = joints[['actual_TCP_pose_0', 'actual_TCP_pose_1', 'actual_TCP_pose_2', 'actual_TCP_pose_3',
                      'actual_TCP_pose_4', 'actual_TCP_pose_5']]

        joints = joints[['timestamp', 'actual_q_0', 'actual_q_1', 'actual_q_2', 'actual_q_3', 'actual_q_4',
                         'actual_q_5', 'actual_qd_0', 'actual_qd_1', 'actual_qd_2', 'actual_qd_3', 'actual_qd_4',
                         'actual_qd_5']]
    else:
        joints = load_rosbag_csv(txt_path)

    if plot_timestamps:
        # timestamp analysis [ms]
        dt = (joints['field.header.stamp'] / 1e9 - joints['time'] / 1e9) * 1e3
        plt.plot(dt, '+')
        #format_figure()
        plt.show()

    if rtde:
        stamps = joints['timestamp']
        joints = joints[['actual_q_0', 'actual_q_1', 'actual_q_2',
                         'actual_q_3', 'actual_q_4', 'actual_q_5']].values
    else:
        stamps = joints['field.header.stamp']
        vels = joints[['field.velocity0',
                       'field.velocity1', 'field.velocity2', 'field.velocity3',
                       'field.velocity4', 'field.velocity5']].values
        joints = joints[['field.position0',
                         'field.position1', 'field.position2', 'field.position3',
                         'field.position4', 'field.position5']].values

    if smooth:
        joints = savgol_filter(joints, 61, 2, axis=0)

    # convert to poses
    poses = forward_kinematic(joints, timestamps=stamps)

    if rtde:
        # fake timestamp
        poses[:, 0] += time.time()
        t = pd.to_datetime(poses[:, 0], unit='s')
        df = pd.DataFrame(data=hstack((poses, joints, urp)), index=t,
                          columns=['ut', 'x', 'y', 'z'] + ['dcm_{}'.format(i) for i in range(9)] +
                                  ['j_{}'.format(i) for i in range(6)] + ['xc', 'yc', 'zc', 'rxc', 'ryc', 'rzc'])
        df['t'] = t

    else:
        t = pd.to_datetime(poses[:, 0], unit='ns')
        if include_vel:
            df = pd.DataFrame(data=hstack((poses, joints, vels)), index=t,
                              columns=['ut', 'x', 'y', 'z'] + ['dcm_{}'.format(i) for i in range(9)] +
                                      ['j_{}'.format(i) for i in range(6)] + ['dj_{}'.format(i) for i in range(6)])
        else:
            df = pd.DataFrame(data=hstack((poses, joints)), index=t,
                              columns=['ut', 'x', 'y', 'z'] + ['dcm_{}'.format(i) for i in range(9)] +
                                      ['j_{}'.format(i) for i in range(6)])
        df['t'] = t
        df['ut'] /= 1e9

    df['k'] = [i for i in range(df.shape[0])]

    if target_path is not None:
        df.to_pickle(target_path)

    return df


def load_and_save_tf(txt_path, target_path=None):

    if target_path and os.path.isfile(target_path):
        return pd.read_pickle(target_path)

    data = []
    # load whole csv
    with open(txt_path) as fp:
        for line in fp:
            elements = line.split(',')
            if elements[4] == 'tool0_controller':
                l = {'time': int64(elements[0]), 'stamp': int64(elements[2]), 'x': float64(elements[5]), 'y': float64(elements[6]),
                     'z': float64(elements[7]),
                     'qx': float64(elements[8]), 'qy': float64(elements[9]), 'qz': float64(elements[10]),
                     'qw': float64(elements[11])}
                data.append(l)

    df = pd.DataFrame(data)

    t = pd.to_datetime(df.stamp.values, unit='ns')
    df['ut'] = df.stamp.values
    df.drop(['stamp'], axis='columns', inplace=True)

    df['t'] = t
    df['ut'] /= 1e9
    df['time'] /= 1e9
    df['k'] = [i for i in range(df.shape[0])]
    df.set_index('t', drop=False, inplace=True)

    def helper(quat):
        return quat2dcm(quat).flatten()

    # convert quat to dcm
    dcms = df[['qw', 'qx', 'qy', 'qz']].apply(helper, axis=1).values
    vcms = vstack(dcms.tolist())
    df[['dcm_{}'.format(i) for i in range(9)]] = pd.DataFrame(vcms, index=df.index)

    # convert dcm to euler
    def comp_eul(dcm):
        from utils.misc import dcm2euler

        eul = dcm2euler(asarray(dcm).reshape((3, 3))).T.flatten().squeeze()
        return eul

    eul = df[['dcm_{}'.format(i) for i in range(9)]].apply(comp_eul, axis=1).values
    eul = vstack(eul.tolist())

    df[['ro', 'pi', 'ya']] = pd.DataFrame(eul, index=df.index)

    if target_path:
        df.to_pickle(target_path)

    return df


if __name__ == '__main__':
    # joints = pd.read_csv(r'/home/tomas/Data/UR_Kinematic/Messung_060219/static.txt')
    # joints = joints[['field.position0',
    #    'field.position1', 'field.position2', 'field.position3',
    #    'field.position4', 'field.position5']].values
    #
    # poses = forward_kinematic(joints)
    #
    # poses_ref = pd.read_csv(r'/home/tomas/Data/UR_Kinematic/Messung_060219/poses/static.txt', header=None)
    #
    # print('OK')

    poses = load_and_save_tf(r'/home/tomas/Data/UR_Trafo_2/20190528/tf_tps_ur.csv')
    poses_j = load_and_save_joints(r'/home/tomas/Data/UR_Trafo_2/20190528/joints_tps_ur.csv')

    def plot_eul():
        def comp_eul(dcm):
            from utils.misc import dcm2euler

            eul = dcm2euler(asarray(dcm).reshape((3, 3))).T.flatten().squeeze()
            return eul

        eul = poses_j[['dcm_{}'.format(i) for i in range(9)]].apply(comp_eul, axis=1).values
        eul_new = poses[['dcm_{}'.format(i) for i in range(9)]].apply(comp_eul, axis=1).values

        eul = vstack(eul.tolist())
        eul_new = vstack(eul_new.tolist())

        ax1 = plt.subplot(311)
        plt.plot(eul[:, 0] / pi * 200, '+', label='Joints')
        plt.plot(eul_new[:, 0] / pi * 200, '+', label='Pose')

        plt.subplot(312, sharex=ax1)
        plt.plot(eul[:, 1] / pi * 200, '+', label='Joints')
        plt.plot(eul_new[:, 1] / pi * 200, '+', label='Pose')

        plt.subplot(313, sharex=ax1)
        plt.plot(eul[:, 2] / pi * 200, '+', label='Joints')
        plt.plot(eul_new[:, 2] / pi * 200, '+', label='Pose')
        plt.legend()

        #format_figure()
        plt.tight_layout()

        # second figure
        plt.figure()
        ax2 = plt.subplot(311)
        plt.plot(poses_j.x, '+', label='Joints')
        plt.plot(poses.x, '+', label='Pose')

        plt.subplot(312, sharex=ax2)
        plt.plot(poses_j.y, '+', label='Joints')
        plt.plot(poses.y, '+', label='Pose')

        plt.subplot(313, sharex=ax2)
        plt.plot(poses_j.z, '+', label='Joints')
        plt.plot(poses.z, '+', label='Pose')
        plt.legend()

        # format_figure()
        plt.tight_layout()

        plt.show()

    plot_eul()

    print('OK')
