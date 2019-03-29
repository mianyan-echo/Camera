import os
import cv2
import gc
from multiprocessing import Process, Manager


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
    while True:
        if len(stack1) != 0 and len(stack2) != 0:
            value1 = stack1.pop()
            value2 = stack2.pop()
            cv2.imshow("img1", value1)
            cv2.imshow("img2", value2)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
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
