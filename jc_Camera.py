import os
import cv2
from multiprocessing import Process, Manager


# 向共享缓冲栈中写入数据:
def write(stack, top: int) -> None:
    print('Process to write: %s' % os.getpid())
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        if _:
            stack.append(img)
            # 每到一定容量清空一次缓冲栈
            if len(stack) >= top:
                del stack[:]


# 在缓冲栈中读取数据:
def read(stack) -> None:
    print('Process to read: %s' % os.getpid())
    while True:
        if len(stack) != 0:
            value = stack.pop()
            cv2.imshow("img", value)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break


if __name__ == '__main__':
    # 父进程创建缓冲栈，并传给各个子进程：
    q = Manager().list()
    pw = Process(target=write, args=(q, 100))
    pr = Process(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    pr.start()

    # 等待pr结束:
    pr.join()

    # pw进程里是死循环，无法等待其结束，只能强行终止:
    pw.terminate()
