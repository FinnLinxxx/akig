#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2

def talker():
    pub = rospy.Publisher('tscan_cloud', PointCloud2, queue_size=10)
    rospy.init_node('ptcl2_talker', anonymous=True)
    rate = rospy.Rate(3) # 3hz
    while not rospy.is_shutdown():

        msg = PointCloud2()
        msg.header.stamp = rospy.get_time()
        msg.header.frame_id = 'leverarm0'
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
