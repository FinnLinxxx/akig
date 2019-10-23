#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String

def callback(data):
    # listen to the x coordinate, measured by the lasertracker 
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.point.x)
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('trackerListener', anonymous=True)

    rospy.Subscriber("/tracker_points", PointStamped, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
