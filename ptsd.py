#  -*- coding=utf-8 -*-
import math
import numpy as np

# 欧拉角转四元数
while (1):
    which_mode = input("欧拉角转四元数输入1，四元数转欧拉角输入2。\n请输入:")
    if (which_mode == "1"):
        print("\n源坐标系到目标坐标系旋转顺序为X,Y,Z,左手系.")
        r = float(input("绕x轴旋转角度:"))
        p = float(input("绕y轴旋转角度:"))
        y = float(input("绕z轴旋转角度:"))

        sinp = math.sin(math.radians(p / 2))
        siny = math.sin(math.radians(y / 2))
        sinr = math.sin(math.radians(r / 2))

        cosp = math.cos(math.radians(p / 2))
        cosy = math.cos(math.radians(y / 2))
        cosr = math.cos(math.radians(r / 2))

        w = cosr * cosp * cosy + sinr * sinp * siny
        x = sinr * cosp * cosy - cosr * sinp * siny
        y = cosr * sinp * cosy + sinr * cosp * siny
        z = cosr * cosp * siny - sinr * sinp * cosy

        print("x : {}".format(x))
        print("y : {}".format(y))
        print("z : {}".format(z))
        print("w : {}".format(w))
    elif (which_mode == "2"):
        print("请按顺序输入4元数")
        x = input("x:")
        y = input("y:")
        z = input("z:")
        w = input("w:")

        x = float(x)
        y = float(y)
        z = float(z)
        w = float(w)

        r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        r = r / math.pi * 180
        p = math.asin(2 * (w * y - z * x))
        p = p / math.pi * 180
        y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        y = y / math.pi * 180
        print("\n源坐标系到目标坐标系旋转顺序为X,Y,Z,左手系．")
        print("绕x轴旋转角度: {}".format(r))
        print("绕y轴旋转角度: {}".format(p))
        print("绕z轴旋转角度: {}\n".format(y))

