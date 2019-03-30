import os
import cv2
import gc
import numpy as np
from multiprocessing import Process, Manager, Lock


# 向共享缓冲栈中写入数据:
def write(stack, cam, top: int, locks) -> None:
    """
    :param locks: 进程锁
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
                with locks:
                    print("清栈")
                    del stack[:]
                    gc.collect()
                    stack.append(img)


# 在缓冲栈中读取数据:
def read(stack1) -> None:
    print('Process to read: %s' % os.getpid())
    # 裁剪量
    cut = 15
    mat = []

    while True:
        if len(stack1) != 0:

            more_useful_img_data_l = stack1.pop()
            more_useful_img_data_l[:cut, :, :] = 0
            more_useful_img_data_l[more_useful_img_data_l.shape[0] - cut:, :, :] = 0
            more_useful_img_data_l[:, :cut, :] = 0
            more_useful_img_data_l[:, more_useful_img_data_l.shape[1] - cut:, :] = 0

            pri = np.zeros(more_useful_img_data_l.shape, np.uint8)

            # 制作透过红色的蒙版
            lower = np.array([235, 235, 235])
            upper = np.array([255, 255, 255])
            mask_r = cv2.inRange(more_useful_img_data_l, lower, upper)

            # 将蒙版与图像结合，得到只透红色的图像
            res = cv2.bitwise_and(more_useful_img_data_l, more_useful_img_data_l, mask=mask_r)

            # 画出图像中红色图形的外轮廓
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)  # 将处理后的图像转换为灰度图，便于二值化
            # 中值滤波
            gray = cv2.medianBlur(gray, 3)
            re, thresh = cv2.threshold(gray, 1, 1, cv2.THRESH_TOZERO)  # 将图像二值化，此函数只接受灰度图
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # 在二值化的基础上找出轮廓
            # image为thresh图像，contours为轮廓点的坐标，hierarchy为轮廓索引
            # 目前为止，hierarchy属性显得非常鸡肋，因为是动态视频，每次屏幕刷新索引号都不一样，所以无法应用索引号去噪

            if not contours:
                """
                这里一定要先判断轮廓是否为空，cv2.moments()函数是不能接收空轮廓的。
                后面这里可以加上输出接口，向右转、向后退、板子收起。
                """
                continue
            else:
                """
                同样，这里也可以加入输出接口
                """
                mat = cv2.moments(contours[-1])
                # 返回一个字典
                if int(mat['m00']) == 0:  # 这个判断会屏蔽中点在边缘的情况
                    continue
                else:  # 找出中点坐标
                    cx = mat['m10'] / mat['m00']
                    cy = mat['m01'] / mat['m00']

                    print(cx, cy)

            # 在画布上画出轮廓，做出中心点
            img = cv2.drawContours(pri, contours, -1, (255, 255, 255), 1)
            img = cv2.circle(img, (int(cx), int(cy)), 3, (0, 0, 255), -1)
            cv2.imshow("video", img)

            cv2.imshow("img1", more_useful_img_data_l)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    print(mat, "\n", contours)


if __name__ == '__main__':
    lock = Lock()
    # 父进程创建缓冲栈，并传给各个子进程：
    manger = Manager()
    q1 = manger.list()
    q2 = manger.list()
    # w1 = Process(target=write, args=(q1, "rtsp://admin:Zxcvbnm123@192.168.1.102:554/ONVIFMedia", 100, lock))
    pw2 = Process(target=write, args=(q2, "rtsp://admin:Zxcvbnm123@192.168.1.102:554/ONVIFMedia", 100, lock))
    pr = Process(target=read, args=(q2,))
    # 启动子进程pw，写入:
    # pw1.start()
    pw2.start()
    # 启动子进程pr，读取:
    pr.start()

    # 等待pr结束:
    pr.join()

    # pw进程里是死循环，无法等待其结束，只能强行终止:
    # pw1.terminate()
    pw2.terminate()
