#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped

def callback(data, f):
    f.write('%10.6f\t%10.6f\t%10.6f\n' % (data.point.x, data.point.y, data.point.z))
    
def listener(f):
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/tracker_points", PointStamped, callback, f)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    f = open('coordinates_tracker_output_05Pts125hzLAN.txt', 'w')
    f.write('%10s\t%10s\t%10s\n'%('x', 'y', 'z'))
    listener(f)
    f.close()
