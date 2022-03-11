from pymycobot.mycobot import MyCobot
from pymycobot.genre import Angle
from pymycobot.genre import Coord
from pymycobot import PI_PORT, PI_BAUD  # 当使用树莓派版本的mycobot时，可以引用这两个变量进行MyCobot初始化
import time
import math
import serial

# 初始化一个MyCobot对象
mc = MyCobot("COM5", 115200)
#mc.send_angles([0, 0,0,0,0,0], 20)
#time.sleep(5)
# 获得当前位置的坐标
angle_datas = mc.get_angles()
print(angle_datas)
coords = mc.get_coords()
print(coords)
#mc.send_angles([82.61, 91.58, -30, 4.39, 165, 48.25], 20)
#time.sleep(5)
#mc.send_coords([43.5, -86.9, 350, 0, 0, 0], 30, 0)
#time.sleep(15)
mc.send_angles([0.26, 71.85, 37.5, -19.42, 0, 89.73], 20)
time.sleep(5)
#mc.send_coords([183.9, -177.4, 221.8, 13.85, 1.1, -52.8], 10, 0)
#time.sleep(15)
coords = mc.get_coords()
print(coords)
mc.pause()
#mc.send_angles([-80, -75, -95, 84, -3, 52], 10)
#mc.release_all_servos()
