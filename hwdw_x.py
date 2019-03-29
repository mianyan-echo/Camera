# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 18:58:57 2018

@author: linxiaobaobao
本程序用于做防爆处理后的红外热像仪的双目定位
"""
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt


class DoubleH:
    def __init__(self, M1_1=np.zeros((3, 4)), R_1=np.zeros((3, 3)), t_1=np.zeros((3, 1)),
                 M1_2=np.zeros((3, 4)), R_2=np.zeros((3, 3)), t_2=np.zeros((3, 1)),
                 k_1=np.array([[-0.533110933755017], [0.437580595249307]]),
                 k_2=np.array([[-0.533110933755017], [0.437580595249307]]),
                 mid_1=[1024, 544], mid_2=[1024, 544]):
        """
        <1>函数功能：
        成像矩阵初始化，其中M_x是成像矩阵，M1_x是内参数矩阵，R_x是旋转矩阵，t_x是平移矩阵
        由R和t可求外参数矩阵M2_x,_1是左眼，_2是右眼
        <2>使用示例：
        DH=DoubleH()
        """
        self.M1_1 = M1_1
        self.M2_1 = np.zeros((4, 4))
        self.M2_1[0:3, 0:3] = R_1
        self.M2_1[0:3, 3:4] = t_1
        self.M2_1[3, 3] = 1
        self.M_1 = np.dot(M1_1, self.M2_1)

        self.M1_2 = M1_2
        self.M2_2 = np.zeros((4, 4))
        self.M2_2[0:3, 0:3] = R_2
        self.M2_2[0:3, 3:4] = t_2
        self.M2_2[3, 3] = 1
        self.M_2 = np.dot(M1_2, self.M2_2)

        self.fdx_l = self.M1_1[0, 0]
        self.fdx_r = self.M1_2[0, 0]
        self.fdy_l = self.M1_1[1, 1]
        self.fdy_r = self.M1_2[1, 1]

        self.k_1 = k_1
        self.k_2 = k_2

        self.mid_1 = mid_1
        self.mid_2 = mid_2

    def Obtain(self, M1_1, R_1, t_1, k_1, mid_1, M1_2, R_2, t_2, k_2, mid_2):
        """
        <1>函数功能：
        更新M1,R,t以及畸变矫正系数，获得新的成像变换矩阵M和畸变矫正系数
        <2>使用示例：
        DoubleH.Obtain(M1_l,R_l,t_l,k_l,M1_r,R_r,t_r,k_r)
        """
        self.M1_1 = M1_1
        self.M2_1 = np.zeros((4, 4))
        self.M2_1[0:3, 0:3] = R_1
        self.M2_1[0:3, 3:4] = t_1
        self.M2_1[3, 3] = 1
        self.M_1 = np.dot(M1_1, self.M2_1)

        self.M1_2 = M1_2
        self.M2_2 = np.zeros((4, 4))
        self.M2_2[0:3, 0:3] = R_2
        self.M2_2[0:3, 3:4] = t_2
        self.M2_2[3, 3] = 1
        self.M_2 = np.dot(M1_2, self.M2_2)

        self.fdx_l = self.M1_1[0, 0]
        self.fdx_r = self.M1_2[0, 0]
        self.fdy_l = self.M1_1[1, 1]
        self.fdy_r = self.M1_2[1, 1]

        self.k_1 = k_1
        self.k_2 = k_2

        self.mid_1 = mid_1
        self.mid_2 = mid_2

    def positioning1(self, frame, eye, lower=100, upper=255):
        """
        寻找单张图片中的目标点，确定目标点坐标
        """
        # lower = np.array(lower)
        # upper = np.array(upper)
        # name=0
        # 设置参数寄存器

        mask = cv2.inRange(frame, lower, upper)
        # 腐蚀操作,去除噪点
        # mask = cv2.erode(mask, None, iterations=1)
        # 膨胀操作，去除噪点
        # mask = cv2.dilate(mask, None, iterations=2)

        # imgzb1=self.imgzb1     #在循环中收集坐标
        imgzb1 = np.zeros((1, 2))
        # 轮廓检测
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        self.shape = frame.shape
        # 如果存在轮廓
        cents = np.zeros((1, 2))
        radius = 0
        for i in range(len(cnts)):
            # 找到面积最大的轮廓
            c = cnts[i]
            # 确定外接圆
            r = radius
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            if radius > r:
                cents = [x, y]
                imgzb1[0, 0:2] = cents

        # print(imgzb1)
        if eye == 'l':
            imgzb2 = self.jbjz(imgzb1[0], self.k_1, self.mid_1, self.fdx_l, self.fdy_l)
        elif eye == 'r':
            imgzb2 = self.jbjz(imgzb1[0], self.k_2, self.mid_2, self.fdx_r, self.fdy_r)
        else:
            print('请输入正确的左右眼代号！')
        imgzb3 = imgzb2[0]
        # print(imgzb3)
        # imgzb1[0,2]=1.0
        # x3=np.dot(imgzb1,TT)
        # x=list(x3[0,0:2])

        # x = list(imgzb1[0,0:2])
        # xx=[0,0]
        # xx[0]=int(x[0])
        # xx[1]=int(x[1])
        return imgzb3

    def positioning(self, frame, lower=200, upper=255):
        """
        #寻找单张图片中的校准点，返回校准点图像坐标
        """
        # lower = np.array(lower)
        # upper = np.array(upper)
        # name=0
        # 设置参数寄存器

        mask = cv2.inRange(frame, lower, upper)
        # 腐蚀操作,去除噪点
        # mask = cv2.erode(mask, None, iterations=1)
        # 膨胀操作，去除噪点
        # mask = cv2.dilate(mask, None, iterations=2)

        # 轮廓检测
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        self.shape = frame.shape
        # 如果存在轮廓

        radius = 0
        imgzb = np.zeros((len(cnts), 2))
        for i in range(len(cnts)):
            # 找到面积最大的轮廓
            c = cnts[i]
            # 确定外接圆
            # r=radius
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            imgzb[i, :] = np.array([x, y])

        imgzb2 = np.zeros((1, 2))
        imgzb2 = imgzb[0, 0:2]
        imgzb3 = self.jbjz(imgzb2)
        # print('畸变矫正结果：'+str(imgzb3))
        x = list(imgzb3[0])
        return x

        # imgzb1[0,2]=1.0
        # x3=np.dot(imgzb1,TT)
        # x=list(x3[0,0:2])

    def Lsm(self, imgzb, zb, zc):
        """
        #<1>函数功能：
        #最小二乘法求成像矩阵M
        #<2>使用示例：
        #DoubleH.Obtain(M1,R,t)
        #<3>参数说明
        #imgzb为（n,2）的目标点图片坐标系坐标
        #zb为（n,3）的目标点世界坐标系坐标
        #zc为摄像机坐标系深度坐标
        """
        X = np.zeros((3, 12))
        X[0, 0:3] = zb[0, 0:3]
        X[0, 3] = 1
        X[1, 4:7] = zb[0, 0:3]
        X[1, 7] = 1
        X[2, 8:11] = zb[0, 0:3]
        X[2, 11] = 1

        Y = np.zeros((3, 1))
        Y[0, 0] = imgzb[0, 0] * zc
        Y[1, 0] = imgzb[0, 1] * zc
        Y[2, 0] = zc

        for i in range(1, zb.shape[0]):
            X1 = np.zeros((3, 12))
            X1[0, 0:3] = zb[i, 0:3]
            X1[0, 3] = 1
            X1[1, 4:7] = zb[i, 0:3]
            X1[1, 7] = 1
            X1[2, 8:11] = zb[i, 0:3]
            X1[2, 11] = 1
            X = np.vstack((X, X1))

            Y1 = np.zeros((3, 1))
            Y1[0, 0] = imgzb[i, 0] * zc
            Y1[1, 0] = imgzb[i, 1] * zc
            Y1[2, 0] = zc
            Y = np.vstack((Y, Y1))
        X = np.mat(X)
        Y = np.mat(Y)
        m = (X.T * X).I * X.T * Y
        M = np.zeros((4, 3))
        M = np.mat(M)
        M[0, 0:4] = m[0:4].T
        M[1, 0:4] = m[4:8].T
        M[2, 0:4] = m[8:12].T
        return np.asarray(M)

    def Get(self, f_l, zb_l, zc_l, f_r, zb_r, zc_r):
        """
        #通过校准的方法获取成像矩阵
        """
        imgzb_l = self.positioning(f_l)
        imgzb_r = self.positioning(f_r)
        self.M_1 = self.Lsm(imgzb_l, zb_l, zc_l)
        self.M_2 = self.Lsm(imgzb_r, zb_r, zc_r)

    def Dh(self, imgzb_l, imgzb_r):
        """
        获取目标点三维坐标
        """
        u1 = imgzb_l[0]
        v1 = imgzb_l[1]
        u2 = imgzb_r[0]
        v2 = imgzb_r[1]
        # print('u1:',u1)
        # print('v1',v1)
        # print('u2',u2)
        # print('v2',v2)

        M_1 = self.M_1
        M_2 = self.M_2

        a = np.zeros((4, 3))
        b = np.zeros((4, 1))

        a[0, 0] = u1 * M_1[2, 0] - M_1[0, 0]
        a[0, 1] = u1 * M_1[2, 1] - M_1[0, 1]
        a[0, 2] = u1 * M_1[2, 2] - M_1[0, 2]

        a[1, 0] = v1 * M_1[2, 0] - M_1[1, 0]
        a[1, 1] = v1 * M_1[2, 1] - M_1[1, 1]
        a[1, 2] = v1 * M_1[2, 2] - M_1[1, 2]

        a[2, 0] = u2 * M_2[2, 0] - M_2[0, 0]
        a[2, 1] = u2 * M_2[2, 1] - M_2[0, 1]
        a[2, 2] = u2 * M_2[2, 2] - M_2[0, 2]

        a[3, 0] = v2 * M_2[2, 0] - M_2[1, 0]
        a[3, 1] = v2 * M_2[2, 1] - M_2[1, 1]
        a[3, 2] = v2 * M_2[2, 2] - M_2[1, 2]

        # sad=M_1[0,3]-u1*M_1[2,3]
        # print('sad',sad)
        # print('M_1 0 3',M_1[0,3])
        # print('M_1 0 3',M_1[2,3])

        b[0, 0] = M_1[0, 3] - u1 * M_1[2, 3]
        b[1, 0] = M_1[1, 3] - v1 * M_1[2, 3]

        b[2, 0] = M_2[0, 3] - u2 * M_2[2, 3]
        b[3, 0] = M_2[1, 3] - v2 * M_2[2, 3]
        X = np.mat(a)
        Y = np.mat(b)
        # print('M_1')
        # print(M_1)
        # print('M_2')
        # print(M_2)
        # print('X')
        # print(X)
        # print('Y')
        # print(Y)
        m = (X.T * X).I * X.T * Y
        # x = solve(a, b)
        return np.asarray(m)

    def jbjz(self, imgzb=np.zeros((1, 2)), k=np.array([[-0.5331], [0.4376]]),
             mid=[1024, 544], fdx=1600, fdy=1600):
        """
        功能：畸变矫正
        """
        k = np.mat(k)

        u = imgzb[0]
        v = imgzb[1]

        D = np.zeros((2, 2))

        u0 = mid[0]
        v0 = mid[1]

        x = (u - u0) / fdx
        y = (v - v0) / fdy
        r2 = x ** 2 + y ** 2
        r4 = r2 ** 2

        D[0, 0] = (u - u0) * r2
        D[0, 1] = (u - u0) * r4
        D[1, 0] = (v - v0) * r2
        D[1, 1] = (v - v0) * r4
        D = np.mat(D)

        imgzb = np.mat(imgzb)
        d = np.zeros((1, 2))
        d = np.mat(d)
        d = D * k + imgzb.T

        return np.asarray(d.T)

    # 请正确输入左右眼rtsp地址


if __name__ == "__main__":
    cap_l = cv2.VideoCapture(r'rtsp://admin:Zxcvbnm123@192.168.1.102:554/ONVIFMedia')
    cap_r = cv2.VideoCapture(r'rtsp://admin:Zxcvbnm123@192.168.1.103:554/ONVIFMedia')

    A = DoubleH()

    # 左眼内参
    M1_1 = np.array([[1598.1, 0, 1020.9, 0], [0, 1592.5, 556.2, 0], [0, 0, 1, 0]])

    # 右眼内参
    M1_2 = np.array([[1586.8, 0, 1025.9, 0], [0, 1577.7, 537.1, 0], [0, 0, 1, 0]])

    # 旋转矩阵初始化
    R_1 = np.identity(3)
    R_2 = np.identity(3)

    # 平移矩阵初始化
    t_1 = np.zeros((3, 1))
    t_1[0, 0] = 175  # 300

    t_2 = np.zeros((3, 1))
    t_2[0, 0] = -175  # -300
    # 畸变矫正参数
    k_1 = np.array([[0.4053], [-0.1616]])
    k_2 = np.array([[0.4041], [-0.1597]])
    # 中心坐标
    mid_1 = [M1_1[0, 2], M1_1[1, 2]]
    mid_2 = [M1_2[0, 2], M1_2[1, 2]]

    A.Obtain(M1_1, R_1, t_1, k_1, mid_1, M1_2, R_2, t_2, k_2, mid_2)

    # create new frames for the camera
    k = 1
    plt.ion()
    fig, ax = plt.subplots()
    x, y = [], []
    colors = ['b', 'g', 'r', 'orange']
    sc = ax.scatter(x, y, c=colors[1], cmap='brg', s=500)
    plt.xlim((-800, 800))
    plt.ylim((-800, 800))
    plt.grid(True)

    nn = 1000
    zbb = np.zeros((nn, 3))

    while True:

        print(k)
        k = k + 1
        a, moreUsefulImgData_l = cap_l.read()
        a, moreUsefulImgData_r = cap_l.read()

        time.sleep(0.05)

        # try:
        imgzb_l = A.positioning1(moreUsefulImgData_l, 'l')
        imgzb_r = A.positioning1(moreUsefulImgData_r, 'r')
        # print(imgzb_l)
        # print(imgzb_r)
        try:
            X = A.Dh(imgzb_l, imgzb_r)
        except:
            X = np.zeros((3, 1))
        x = X[0, 0]
        y = -X[1, 0]
        z = X[2, 0]
        zbb[k - 1, 0] = x
        zbb[k - 1, 1] = y
        zbb[k - 1, 2] = z
        # x,y,z=doublec.binocularpositioning(moreUsefulImgData_l,moreUsefulImgData_r)
        # except:
        #    x=0
        #    y=0
        #    z=0
        # zblist=zb2.get(coordinate=[x,y,z])
        print("相机坐标：", x, y, z)
        sc.set_offsets(np.c_[x, y])
        fig.canvas.draw_idle()
        plt.pause(0.01)
        if k == nn:
            break

    cap_l.release()
    cap_r.release()
