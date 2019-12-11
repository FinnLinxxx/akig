def compRaster(startPos):
    import numpy as np
    #startPos = np.array([100,100,100]) #Pos endeffector in the WCS (cm)!

    # 1x1x1m Raster
    res = 0.1 #mm
    rast = int(round(0.2/res)) #+1
    # Horizontal
    pos = np.zeros((rast,3))
    for j in range (0,rast):
        pos[j,0] = startPos[0]+j*res
        pos[j,1] = startPos[1]+j*res
        pos[j,2] = startPos[2]+j*res

    newPos = np.zeros((rast*rast,2))
    newP = np.tile(pos[:,0],(rast,1))
    newPos[:,0] = np.reshape(newP,np.size(newP))
    newP = np.repeat(pos[:,1],rast)
    sz = np.size(newP)
    newPos[:, 1] = np.reshape(newP,sz)

    # Vertical
    newP = np.repeat(pos[:,2],rast*rast)
    sz = np.size(newP)
    a = np.reshape(newP,sz)

    vs = np.tile(newPos, (rast, 1))
    Pose = np.zeros((sz,7))
    Pose[:, 1] = vs[:, 0]
    Pose[:, 2] = vs[:, 1] # Achsen 1 und 2 vertauscht!
    Pose[:, 3] = a

    # Umkehrspiel! verschiedene Winkelstellungen

    # Numbering!
    ind = np.arange(0,sz)
    Pose[:,0] = ind.T


    # Orientation!
    Pose[:,4] = np.ones((sz))*startPos[3]
    Pose[:,5] = np.ones((sz))*startPos[4]
    Pose[:,6] = np.ones((sz))*startPos[5]

    #Pose[1:5,4] = np.arange(45,5,-10)

    print(sz)

    return Pose


def compFile(data,variablename):
    from scipy.io import savemat

    savemat(('test_{}.mat'.format(variablename)),{variablename: data})
    print('saved')

    return

def compAscii(data, variablename):

    f = open('{}.txt'.format(variablename), 'a')
    #  FrameName,Base,Shoulder,Elbow,W1,W2,W3
    data_string = '{:d},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n'.format(int(data[0]), data[1], data[2], data[3], data[4], data[5], data[6])
    f.write(data_string)
    f.close()

    return


def parseStr(string):
    import numpy as np
    import xml.etree.ElementTree as ett

    a = ett.fromstring(string)
    # et.fromstringlist(command) #Syntax-Fehler! -> Problem fehlender Abstand zw den Tags!

    try:
        for child in a.iter('Joints'):
            child.tag, child.attrib
        joint = np.zeros((1, 6))
        joint[0, 0] = float(child.get('A1'))
        joint[0, 1] = float(child.get('A2'))
        joint[0, 2] = float(child.get('A3'))
        joint[0, 3] = float(child.get('A4'))
        joint[0, 4] = float(child.get('A5'))
        joint[0, 5] = float(child.get('A6'))
    except:
        joint = np.zeros((1, 6))
        pass

    try:
        for child in a.iter('Info'):
            child.tag, child.attrib
        no = np.zeros((1, 1))
        no[0, 0] = float(child.get('Number'))
    except:
        no = np.zeros((1,1))
        pass

    try:
        for child in a.iter('Pose'):
            child.tag, child.attrib
        pose = np.zeros((1, 8))
        pose[0, 0] = float(child.get('XPos'))
        pose[0, 1] = float(child.get('YPos'))
        pose[0, 2] = float(child.get('ZPos'))
        pose[0, 3] = float(child.get('APos'))
        pose[0, 4] = float(child.get('BPos'))
        pose[0, 5] = float(child.get('CPos'))
        pose[0, 6] = float(child.get('S'))
        pose[0, 7] = float(child.get('T'))
    except:
        pose = np.zeros((1, 8))
        pass

    return joint, pose, no


def compRaster_reversel(startPos):
    import numpy as np

    # startPos = np.array([100,100,100]) #Pos endeffector in the WCS (cm)!

    # 1x1x1m Raster
    res = 200  # mm
    rast = int(round(800 / res))  # +1
    # Horizontal
    pos = np.zeros((rast, 3))
    for j in range(0, rast):
        pos[j, 1] = startPos[0] + j * res
        pos[j, 0] = startPos[1] + j * res
        pos[j, 2] = startPos[2] + j * res

    newPos = np.zeros((rast * rast * 2, 2))
    pp = np.zeros(rast * 2)
    pp[0:rast] = pos[:, 0]
    t = pos[:, 0]
    pp[rast:rast * 2] = t[::-1]  ##pp ()
    newP = np.tile(pp, (rast, 1))
    newPos[:, 0] = np.reshape(newP, np.size(newP))
    newP = np.repeat(pos[:, 1], rast * 2)  # *2
    sz = np.size(newP)
    newPos[:, 1] = np.reshape(newP, sz)

    # Vertical
    newP = np.repeat(pos[:, 2], rast * rast * 2)
    sz = np.size(newP)
    a = np.reshape(newP, sz)

    vs = np.tile(newPos, (rast, 1))
    Pose = np.zeros((sz, 9))
    Pose[:, 2] = vs[:, 0]
    Pose[:, 1] = vs[:, 1]  # Achsen 1 und 2 vertauscht!
    Pose[:, 3] = a

    # Umkehrspiel! verschiedene Winkelstellungen

    # Numbering!
    ind = np.arange(0, sz)
    Pose[:, 0] = ind.T

    # Orientation!
    A = np.array([9.4, -142.2])
    B = np.array([-30.5, -15.7])
    C = np.array([108.8, -95.5])
    Pose[:, 4] = np.tile(np.repeat(A, rast), rast * rast)
    Pose[:, 5] = np.tile(np.repeat(B, rast), rast * rast)
    Pose[:, 6] = np.tile(np.repeat(C, rast), rast * rast)
    Pose[:, 7] = np.ones((sz)) * startPos[6]  # not used in KRL
    Pose[:, 8] = np.ones((sz)) * startPos[7]  # not used in KRL

    # Pose[1:5,4] = np.arange(45,5,-10)

    print(sz)

    return Pose


def compRaster_plane(startPos):
    import numpy as np
    # startPos = np.array([100,100,100]) #Pos endeffector in the WCS (cm)!

    # Resolution, RasterSize
    res_1 = 100  # mm
    rast_1 = int(round(1400 / res_1))
    res_2 = 100  # mm
    rast_2 = int(round(1400 / res_2))

    # Raster
    pos_1 = np.zeros((rast_1))
    pos_2 = np.zeros((rast_2))
    for j in range(0, rast_1):
        pos_1[j] = startPos[1] + j * res_1  # y
    for j in range(0, rast_1):
        pos_2[j] = startPos[2] + j * res_2  # z

    newPos = np.zeros((rast_1 * rast_2, 2))
    newP = np.tile(pos_1, (rast_2, 1))
    newPos[:, 0] = np.reshape(newP, np.size(newP))
    newP = np.repeat(pos_2, rast_1)
    sz = np.size(newP)
    newPos[:, 1] = np.reshape(newP, sz)


    Pose = np.zeros((sz,9))
    Pose[:, 1] = np.ones((sz))*startPos[0]  # x
    Pose[:, 2] = newPos[:,0]
    Pose[:, 3] = newPos[:,1]

    # Numbering!
    ind = np.arange(0, sz)
    Pose[:, 0] = ind.T

    # Orientation!
    Pose[:, 4] = np.ones((sz)) * startPos[3]
    Pose[:, 5] = np.ones((sz)) * startPos[4]
    Pose[:, 6] = np.ones((sz)) * startPos[5]
    Pose[:, 7] = np.ones((sz)) * startPos[6]  # not used in KRL
    Pose[:, 8] = np.ones((sz)) * startPos[7]  # not used in KRL

    # Pose[1:5,4] = np.arange(45,5,-10)

    print(sz)

    return Pose

def compRaster_Scale(startPos):
    import numpy as np
    #startPos = np.array([100,100,100]) #Pos endeffector in the WCS (cm)!

    # Resolution, RasterSize in mm
    res_x = 200
    res_y = 200
    res_z = 100
    dim_x = 800
    dim_y = 1200
    dim_z = 400
    rast_x = int(round(dim_x / res_x))
    rast_y = int(round(dim_y / res_y))
    rast_z = int(round(dim_z / res_z))


    # Horizontal
    pos_x = np.zeros((rast_x))
    pos_y = np.zeros((rast_y))
    pos_z = np.zeros((rast_z))
    for j in range (0,rast_x):
        pos_x[j] = startPos[0]+j*res_x
    for j in range(0, rast_y):
        pos_y[j] = startPos[1]+j*res_y
    for j in range(0, rast_z):
        pos_z[j] = startPos[2]+j*res_z

    newPos = np.zeros((rast_x*rast_y,2))
    newP = np.tile(pos_x,(rast_y,1))
    newPos[:,0] = np.reshape(newP,np.size(newP))
    newP = np.repeat(pos_y,rast_x)
    sz = np.size(newP)
    newPos[:, 1] = np.reshape(newP,sz)

    # Vertical
    newP = np.repeat(pos_z,rast_x*rast_y)
    sz = np.size(newP)
    a = np.reshape(newP,sz)

    vs = np.tile(newPos, (rast_z, 1))
    Pose = np.zeros((sz,9))
    Pose[:, 1] = vs[:, 0]
    Pose[:, 2] = vs[:, 1] # Achsen 1 und 2 vertauscht!
    Pose[:, 3] = a

    # Umkehrspiel! verschiedene Winkelstellungen

    # Numbering!
    ind = np.arange(0,sz)
    Pose[:,0] = ind.T


    # Orientation!
    Pose[:,4] = np.ones((sz))*startPos[3]
    Pose[:,5] = np.ones((sz))*startPos[4]
    Pose[:,6] = np.ones((sz))*startPos[5]
    Pose[:,7] = np.ones((sz))*startPos[6]
    Pose[:,8] = np.ones((sz))*startPos[7]

    #Pose[1:5,4] = np.arange(45,5,-10)

    print(sz)

    return Pose

def readTrafoPoses(file):
    import numpy as np

    # 'C:\\Users\\Horvath.IGGP\\Desktop\\kuka_identPoints.txt'
    f = open(file, 'r')
    a = f.read()
    a = np.fromstring(a, dtype=float, sep=',')
    b = np.reshape(a, (-1, 9))

    return b



# String-Operations:
#    iXPos = string.find("XPos")
#    string[iXPos]
#    b = string.split('"')
#    joints = np.zeros((1, 6))
#    joints[0, 0] = float(b[1])
#    joints[0, 1] = float(b[3])
#    command2 = ('<Robot><Data><Joints A1="{:.1f}" A2="{:.1f}" A3="{:.1f}" A4="{:.1f}" A5="{:.1f}" A6="{:.1f}" /></Data></Robot>'.format(Jointse[i, 0], pose[i, 1], pose[i, 2], pose[i, 0], pose[i, 1], pose[i, 2]))




#def parsePose(string: object) -> object:
#
#    import numpy as np
#    import xml.etree.cElementTree as ett
#
#    a = ett.fromstring(string)
#    # et.fromstringlist(command) #Syntax-Fehler! -> Problem fehlender Abstand zw den Tags!
#
#    for child in a.iter('Pose'):
#        print(child.tag, child.attrib)
#    pose = np.zeros((1, 7))
#    pose[0, 1] = float(child.get('XPos'))
#    pose[0, 2] = float(child.get('YPos'))
#    pose[0, 3] = float(child.get('ZPos'))
#    pose[0, 4] = float(child.get('APos'))
#    pose[0, 5] = float(child.get('BPos'))
#    pose[0, 6] = float(child.get('CPos'))
#
#    for child in a.iter('Info'):
#        print(child.tag, child.attrib)
#    pose[0, 0] = float(child.get('Number'))
#
#    return pose


# Begrenzung des Strings auf Mindestlaenge!
#def myreceive(self):
#    chunks = []
#    bytes_recd = 0
#    while bytes_recd < MSGLEN:
#        chunk = self.sock.recv(min(MSGLEN - bytes_recd, 2048))
#        if chunk == '':
#            raise RuntimeError("socket connection broken")
#        chunks.append(chunk)
#        bytes_recd = bytes_recd + len(chunk)
#    return ''.join(chunks)

