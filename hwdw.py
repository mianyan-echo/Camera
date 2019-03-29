import os
import cv2
import gc
import numpy as np
import matplotlib.pyplot as plt
from hwdw_x import DoubleH
from multiprocessing import Process, Manager

# TODO：加入进程锁
# 向共享缓冲栈中写入数据:
def write(stack, cam, top: int) -> None:
    """
    :param cam: 摄像头参数
    :param stack: Manager.list对象
    :param top: 缓冲栈容量
    :return: None
    """
    print('Process to write: %s' % os.getpid())
    cap = cv2.VideoCapture(cam)
    while True:
        _, img = cap.read()
        if _:
            stack.append(img)
            # 每到一定容量清空一次缓冲栈
            # 利用gc库，手动清理内存垃圾，防止内存溢出
            if len(stack) >= top:
                del stack[:]
                gc.collect()


# 在缓冲栈中读取数据:
def read(stack1, stack2) -> None:
    print('Process to read: %s' % os.getpid())
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

    nn = 100
    zbb = np.zeros((nn, 3))
    while True:
        if len(stack1) != 0 and len(stack2) != 0:
            moreUsefulImgData_l = stack1.pop()
            moreUsefulImgData_r = stack2.pop()

            cv2.imshow("img1", moreUsefulImgData_l)
            cv2.imshow("img2", moreUsefulImgData_r)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # time.sleep(0.05)

            # try:
            imgzb_l = A.positioning1(moreUsefulImgData_l, 'l')
            imgzb_r = A.positioning1(moreUsefulImgData_r, 'r')
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


if __name__ == '__main__':
    # 父进程创建缓冲栈，并传给各个子进程：
    manger = Manager()
    q1 = manger.list()
    q2 = manger.list()
    pw1 = Process(target=write, args=(q1, "rtsp://admin:Zxcvbnm123@192.168.1.102:554/ONVIFMedia", 100))
    pw2 = Process(target=write, args=(q2, "rtsp://admin:Zxcvbnm123@192.168.1.103:554/ONVIFMedia", 100))
    pr = Process(target=read, args=(q1, q2))
    # 启动子进程pw，写入:
    pw1.start()
    pw2.start()
    # 启动子进程pr，读取:
    pr.start()

    # 等待pr结束:
    pr.join()

    # pw进程里是死循环，无法等待其结束，只能强行终止:
    pw1.terminate()
    pw2.terminate()
