import socket
import numpy as np
# import struct
# import chardet
import time
import datetime

from comput1 import *


HOST = '192.168.178.82'
#'192.168.136.1'; old: '128.131.178.60'
#  Server-Address

PORT = 50000         # Arbitrary non-privileged portnn
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.settimeout(30)
s.listen(1)
addr = None
cont = True
dat = np.zeros([10, 6])


while cont:

    conn, addr = s.accept()

    if addr is None or conn is None:
        s.close()
        exit()


    print('Connected by', addr)


    ind = True
    i = 0
    no = np.zeros(1)
    pose = np.zeros((100, 7))
    joints = np.zeros((100,7))
    while ind:
        try:
            no += 1


            # Pose
            data = conn.recv(90).decode()  # 'ISO-8859-1'
            # print(data)

            try:
                po = np.array(data[2:len(data)-1].split(','))  # tcpPose: p[] ... 2:len(); joint: 1:len()-1
                print(po)
                po = po.astype(float)
                data = ''
                # jo, po, no = parseStr(xmlstr)    # Kuka
            except:
                compAscii(pose, '191117_pose')
                break

            pose[i, 1:] = po
            pose[i, 0] = no
            compAscii(pose[i, :], '191117_pose')



            # Joints
            data = conn.recv(90).decode()  # 'ISO-8859-1'
            print(data)

            try:
                jo = np.array(data[1:len(data)-1].split(','))  # tcpPose: p[] ... 2:len(); joint: 1:len()-1
                jo = jo.astype(float)
                data = ''
                # jo, po, no = parseStr(xmlstr)    # Kuka
            except:
                compAscii(pose, '191117_joints')
                break

            joints[i, 1:] = jo
            joints[i, 0] = no
            compAscii(joints[i, :], '191117_joints')


            i += 1

        except socket.timeout:
            print('Timeout!')
            break

        except (ConnectionAbortedError, ConnectionResetError):
            compFile(joints, 'Joints')
            break

        if not po.all():
            print('Break!')
            break

    conn.close()


s.close()
