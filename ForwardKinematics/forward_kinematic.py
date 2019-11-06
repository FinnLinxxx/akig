from numpy import array, sin, cos, zeros, float64, hstack
import numpy as np

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

print("Calculating forward kinematics")

# theta, a, d, alpha for each joint (6x4)
# Denavit Hartenberg Parameter
#
# theta [rad]
# a [m]
# d [m]
# alpha [rad]
def forward_kinematic(joints, timestamps=None, params=None):

    """
    Convert Joint positions to cartesian coordinates and rotation matrices

    :param joints: nx6 array of joint readings
    :param params: if None the default UR5 param set is used otherwise 6x4 matrix (theta, a, d, alpha per joint)
    :return:
    """

    # load default UR 5 params if not specified

    python_array = array([[ 1 , 0 , 0],
                        0 , 1 , 0],
                        0 , 0 , 1]])

    params = UR5_PARAMS

    # arange poses array
    poses = zeros((joints.shape[0], 12), dtype=float64)

    for i, j in enumerate(joints):
        p = params.copy()
        p[:, 0] = j
        T, R, t = denavit_hartb(p)
        poses[i, :3] = t
        poses[i, 3:] = R.flatten() #flatten() returns a copy of the array collapsed into one dimension.

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
    for i in range(params.shape[0]): # loop through every joint 1 to j (6)
        A = dh_rot(*params[i, :])
        print(A)
        Ti += [A] # chain different Matrices in python behind each other
        print(Ti)

    T = Ti[0] @ Ti[1] @ Ti[2] @ Ti[3] @ Ti[4] @ Ti[5]

    # return T, R, t
    return T, T[:3, :3], T[:3, 3]





def dh_rot(theta, a, d, alpha):
    """
    Denavit Hartenberg Rotation Matrix [Robotic, Vision and Control, P. Corke p. 138]

    :param theta: joint angle
    :param a: link length
    :param d: link offset
    :param alpha: link twist
    :return: transformation matrix 4x4
    """
   # r = another array

    return r

result = forward_kinematic(np.array([[0,0,0,0,0,0]]))
#result = forward_kinematic(np.array([[0.062412720173597336, -0.9297316710101526, 0.5339670181274414, -1.198200527821676, 1.6025909185409546, -1.4876373449908655]]))
print(result)
