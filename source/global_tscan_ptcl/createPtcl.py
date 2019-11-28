#!/usr/bin/env python
# PointCloud2 color cube
# https://answers.ros.org/question/289576/understanding-the-bytes-in-a-pcl2-message/
import rospy
import struct

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


def tscan_talker(t,x_tscan,y_tscan,time_delta,parent_frame,start_meas_tscan_rostime,iii):
    pub = rospy.Publisher("tscan_cloud2", PointCloud2, queue_size=2)
    rospy.init_node("create_cloud_xyz")
    rate = rospy.Rate(1000/time_delta)
    start_meas_tscan_rostime = rospy.get_time() + 3 #remove this line if correct start_meas_tscan_rostime is provided by user
    start_meas_tscan_rostime = 1574188342.201552

    while not rospy.is_shutdown():
        #pcl def
        if start_meas_tscan_rostime < rospy.get_time(): #may use rospy.Time.now()

            print rospy.get_time()
            points = []
            lim = 8
             
            iii = iii + 13135/(1000/time_delta)

            for i in range(1+iii, 13135/(1000/time_delta) + iii):  # Approx only!!! Measurement per seconds=13135 in respekt to hz (1/10) for now
                if not (x_tscan[i] == "INVALID" or y_tscan[i] == "INVALID"):
                    x = float(x_tscan[i]) /1000
                    y = 0
                    z = float(y_tscan[i]) /1000
                    pt = [x, y, z]
                    points.append(pt)

            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    # PointField('rgb', 12, PointField.UINT32, 1),
                    ]
            
            #print points
            
            header = Header()
            header.frame_id = parent_frame
            pc2 = point_cloud2.create_cloud(header, fields, points)



            pc2.header.stamp = rospy.Time.now()
            pub.publish(pc2)
            rate.sleep()

if __name__ == '__main__':
    try:
        iii = 0
        filename_scan = "SCAN0002.asc"
        f=open(filename_scan,"r")
        lines=f.readlines()
        result_t=[]
        result_x=[]
        result_y=[]
        for t in lines:
            result_t.append(t.split('\t')[0])
        for x in lines:
            result_x.append(x.split('\t')[1])
        for y in lines:
            result_y.append(y.split('\t')[2].strip()) #.strip() because of \n seperation
        f.close()
        time_delta = 100 #in milliseconds
        parent_frame = "lever"
        start_meas_tscan_rostime = 1111# not used right now as this is provided by the time calibration later on 
        
        #print result_y
        print float(result_x[1130])
        tscan_talker(result_t,result_x,result_y,time_delta,parent_frame,start_meas_tscan_rostime,iii)
    except rospy.ROSInterruptException:
        pass
