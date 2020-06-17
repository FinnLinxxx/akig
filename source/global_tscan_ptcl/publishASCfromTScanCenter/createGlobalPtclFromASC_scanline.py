#!/usr/bin/env python
# PointCloud2 color cube
# https://answers.ros.org/question/289576/understanding-the-bytes-in-a-pcl2-message/
import rospy
import struct

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

def tscan_talker(t,x_tscan,y_tscan,parent_frame,tscan_idx,iii):

    pub = rospy.Publisher("tscan_cloud2", PointCloud2, queue_size=2)
    rospy.init_node("create_cloud_xyz")

    while not rospy.is_shutdown():
        #pcl def

        if float(t[tscan_idx])/1000000000 < rospy.get_time():
            print(rospy.get_time())
            print(iii)
            points = []
            while float(t[tscan_idx])/1000000000 < rospy.get_time():
                if not (x_tscan[tscan_idx] == "NaN" or y_tscan[tscan_idx] == "NaN"):
                    x = float(x_tscan[tscan_idx]) /1000
                    y = -float(y_tscan[tscan_idx]) /1000
                    z = 0 
                    pt = [x, y, z]
                    points.append(pt)
                    iii = iii + 1
                tscan_idx = tscan_idx + 1 #counter

            fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
          # PointField('rgb', 12, PointField.UINT32, 1),
            ]
            
            header = Header()
            header.frame_id = parent_frame
            pc2 = point_cloud2.create_cloud(header, fields, points)

            pc2.header.stamp = rospy.Time.now()
            pub.publish(pc2)



if __name__ == '__main__':
    try:

        parent_frame = "leverarm" #tf frame of defined leverarm (T-Scan zero)
        tscan_idx = 0

        filename_scan = "tscan_17_01_unixtime.txt"
        f=open(filename_scan,"r")
        lines=f.readlines()
        result_t=[]
        result_x=[]
        result_y=[]
        for t in lines:
            result_t.append(t.split(',')[0])
        for x in lines:
            result_x.append(x.split(',')[1])
        for y in lines:
            result_y.append(y.split(',')[2].strip()) #.strip() because of \n seperation
        f.close()
        iii = 0
        print(len(result_t))
        tscan_talker(result_t,result_x,result_y,parent_frame,tscan_idx,iii)

    except rospy.ROSInterruptException:
        pass
