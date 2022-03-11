import rospy
import math
import transforms3d as tfs
import numpy as np
from geometry_msgs.msg import PoseStamped
from pymycobot.mycobot import MyCobot
from pymycobot.genre import Coord
from pymycobot import PI_PORT,PI_BAUD
import time
mc = MyCobot(PI_PORT, PI_BAUD)
rospy.init_node('listener', anonymous=True)
hand1 = [0.0282, 0.1295, 0.3686, 28.93, -0.57, -10.93,
        -0.1718, -0.0615, 0.2900, -19.03, 14.05, -42.62,
        -0.1183, -0.1852, 0.2471, 9.24, 1.07, -46.45,
        -0.1319, -0.1805, 0.2552, -4.25, 21.15, -89.41,
        0, -0.1396, 0.3053, 38.92, 42.63, -46.54,
        0.0414, -0.1936, 0.2518, 33.0, 64.56, -65.38,
        0.1128, -0.0883, 0.3016, 0.07, 17.37, -44.51,
        0.1839, -0.1774, 0.2218, 13.85, 1.1, -52.8,]

def trans_to_euler(x,y,z,w):
    r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    r = r / math.pi * 180
    p = math.asin(2 * (w * y - z * x))
    p = p / math.pi * 180
    y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    y = y / math.pi * 180
    return [r,p,y]

def get_matrix_eular_radu(x, y, z, rx, ry, rz):
    rmat = tfs.euler.euler2mat(math.radians(rx), math.radians(ry), math.radians(rz))
    rmat = tfs.affines.compose(np.squeeze(np.asarray((x, y, z))), rmat, [1, 1, 1])
    return rmat


def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def rot2quat_minimal(m):
    quat = tfs.quaternions.mat2quat(m[0:3, 0:3])
    return quat[1:]


def quatMinimal2rot(q):
    p = np.dot(q.T, q)
    w = np.sqrt(np.subtract(1, p[0][0]))
    return tfs.quaternions.quat2mat([w, q[0], q[1], q[2]])


def matrix_to_eular(m):
    rx, ry, rz = tfs.euler.mat2euler(m[0:3, 0:3])
    pos = np.squeeze(m[0:3, 3:4])
    return (pos, math.degrees(rx), math.degrees(ry), math.degrees(rz))

def tsai(hand,camera):
    Hgs, Hcs = [], []
    for i in range(0, len(hand), 6):
        Hgs.append(get_matrix_eular_radu(hand[i], hand[i + 1], hand[i + 2], hand[i + 3], hand[i + 4], hand[i + 5]))

        m = get_matrix_eular_radu(camera[i], camera[i + 1], camera[i + 2], camera[i + 3], camera[i + 4], camera[i + 5])
        m = np.linalg.inv(m)
        Hcs.append(m)
    Hgijs = []
    Hcijs = []
    A = []
    B = []
    size = 0
    for i in range(len(Hgs)):
        for j in range(i + 1, len(Hgs)):
            size += 1
            Hgij = np.dot(np.linalg.inv(Hgs[j]), Hgs[i])
            Hgijs.append(Hgij)
            Pgij = np.dot(2, rot2quat_minimal(Hgij))

            Hcij = np.dot(Hcs[j], np.linalg.inv(Hcs[i]))
            Hcijs.append(Hcij)
            Pcij = np.dot(2, rot2quat_minimal(Hcij))

            A.append(skew(np.add(Pgij, Pcij)))
            B.append(np.subtract(Pcij, Pgij))
    MA = np.asarray(A).reshape(size * 3, 3)
    MB = np.asarray(B).reshape(size * 3, 1)
    Pcg_ = np.dot(np.linalg.pinv(MA), MB)
    pcg_norm = np.dot(np.conjugate(Pcg_).T, Pcg_)
    Pcg = np.sqrt(np.add(1, np.dot(Pcg_.T, Pcg_)))
    Pcg = np.dot(np.dot(2, Pcg_), np.linalg.inv(Pcg))
    Rcg = quatMinimal2rot(np.divide(Pcg, 2)).reshape(3, 3)

    A = []
    B = []
    id = 0
    for i in range(len(Hgs)):
        for j in range(i + 1, len(Hgs)):
            Hgij = Hgijs[id]
            Hcij = Hcijs[id]
            A.append(np.subtract(Hgij[0:3, 0:3], np.eye(3, 3)))
            B.append(np.subtract(np.dot(Rcg, Hcij[0:3, 3:4]), Hgij[0:3, 3:4]))
            id += 1

    MA = np.asarray(A).reshape(size * 3, 3)
    MB = np.asarray(B).reshape(size * 3, 1)
    Tcg = np.dot(np.linalg.pinv(MA), MB).reshape(3, )
    # 标记物在机械臂末端的位姿
    marker_in_hand = tfs.affines.compose(Tcg, np.squeeze(Rcg), [1, 1, 1])

    # 手在眼外，标定机器人基座和相机之间的位置，这样我们就可以通过 标记物在相机的位置，相机在机械臂基坐标系的位置，求得物品在机械臂基坐标的位置
    hand_in_marker = np.linalg.inv(marker_in_hand)
    for i in range(len(Hgs)):
        marker_in_camera = np.linalg.inv(Hcs[i])
        base_in_hand = np.linalg.inv(Hgs[i])

        hand_in_camera = np.dot(marker_in_camera, hand_in_marker)
        base_in_camera = np.dot(hand_in_camera, base_in_hand)
        camera_in_base = np.linalg.inv(base_in_camera)
        print(camera_in_base)
    # 这里会输出多组数据，可以求一下均值
    # 这里求得的数据是相机在机械臂基坐标系中的位姿
i = 0
camera = []
hand = []
while not rospy.is_shutdown():
    if i == 0:
        mc.send_coords([-88, -75, 330, 0, 0, 0], 30, 0)
        time.sleep(5)
    elif i == 1:
        mc.send_coords([-88, -75, 380, 0, 0, 0], 30, 0)
        time.sleep(5)
    elif i == 2:
        mc.send_coords([88, -75, 330, 0, 0, 0], 30, 0)
        time.sleep(5)
    elif i == 3:
        mc.send_coords([88, -75, 380, 0, 0, 0], 30, 0)
        time.sleep(5)
    elif i == 4:
        mc.send_coords([88, 75, 330, 0, 0, 0], 30, 0)
        time.sleep(5)
    elif i == 5:
        mc.send_coords([88, 75, 380, 0, 0, 0], 30, 0)
        time.sleep(5)
    elif i == 6:
        mc.send_coords([-88, 65, 330, 0, 0, 0], 30, 0)
        time.sleep(5)
    elif i == 7:
        mc.send_coords([-88, 65, 380, 0, 0, 0], 30, 0)
        time.sleep(5)
    coords = mc.get_coords()
    hand = hand+coords
    #print(coords)
    mc.pause()
    ###################
    msg = rospy.wait_for_message('/aruco_single/pose', PoseStamped)
    print("get it")
    b = str(msg)
    a = b.split()
    position = []
    position.append(float(a[13]))
    position.append(float(a[15]))
    position.append(float(a[17]))
    orientation = []
    orientation.append(float(a[20]))
    orientation.append(float(a[22]))
    orientation.append(float(a[24]))
    orientation.append(float(a[26]))
    #print(position)
    #print(orientation)
    trans_orientation = trans_to_euler(orientation[0],orientation[1],orientation[2],orientation[3])
    #print("transferred orientation>>>>")
    #print(trans_orientation)
    camera=camera+position+trans_orientation

    i = i+1
    if i==8:
        break


print("final>>>>>>>")
print(camera)
print(len(camera))
tsai(hand,camera)