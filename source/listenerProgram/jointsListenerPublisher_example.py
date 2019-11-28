#!/usr/bin/env python
import rospy
import roslib
import math
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import geometry_msgs.msg
import std_msgs.msg



vellist = 42

def callback(data):
    
    # listen to the velocity at joint 0 (base)
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.velocity[0])

    vellist = data.velocity[0]
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    sub = rospy.Subscriber("/joint_states", JointState, callback)
    
    dh_pos_pub = rospy.Publisher('tool0_dhPos', geometry_msgs.msg.PointStamped,queue_size=1)
    while not rospy.is_shutdown():
        
    #create publisher
        cmd = geometry_msgs.msg.PointStamped()
        cmd.header = std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id='tool0')
        cmd.point.x = vellist
        cmd.point.y = 2
        cmd.point.z = 3
        dh_pos_pub.publish(cmd)



if __name__ == '__main__':
    listener()
