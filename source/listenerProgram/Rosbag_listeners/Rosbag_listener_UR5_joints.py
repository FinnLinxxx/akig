#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
# from geometry_msgs.msg import PointStamped

def callback(data, f):
    for joint_idx in range(6):
        f.write('%10.6f\t' % (data.position[joint_idx]))
    f.write('\n')

def listener(f):
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/joint_states", JointState, callback, f)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    f = open('UR5_joint_pos_05Pts125hzLAN.txt', 'w')
    f.write('%10s\t'*6 % ('Joint0', 'Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5'))
    f.write('\n')
    listener(f)
    f.close()
