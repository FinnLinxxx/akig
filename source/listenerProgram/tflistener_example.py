#!/usr/bin/env python  
import roslib
import rospy
import math
import tf
import geometry_msgs.msg
import std_msgs.msg

if __name__ == '__main__':
    rospy.init_node('robotarm_tf_listener')

    listener = tf.TransformListener()

    turtle_vel = rospy.Publisher('tool0Position', geometry_msgs.msg.PointStamped,queue_size=1)

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            (trans,rot) = listener.lookupTransform('/map', '/tool0', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

        cmd = geometry_msgs.msg.PointStamped()
        cmd.header = std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id='tool0')
        cmd.point.x = trans[0]
        cmd.point.y = trans[1]
        cmd.point.z = trans[2]
        print("Publish PointStamped on tool0Position, x: ", trans[0], " y: ", trans[1], " z: ", trans[2])
        turtle_vel.publish(cmd)

        rate.sleep()
