# Camera
# Python多进程opencv

***

前几天遇到了一个问题，利用opencv程序调取rtsp视频流，因为处理程序要消耗的CPU时间过于长，VideoCapture的read是按帧读取，所以经常导致内存溢出，延时还高得出奇。

所以想到是不是可以利用多进程把读取视频和处理视频分开，这样就可以消除因处理图片所导致的延迟。

## 所用库

```python
multiprocessing
gc
opencv-python
os
```

## 实现方法

* 一开始是想用多线程，但是因为GIL的存在，像实时处理视频这样的CPU密集型任务多线程等于没用。然后就选择了多进程。
* 然后要考虑怎样在两个进程中传参的问题:
  * multiprocessing中有Quaue、SimpleQuaue等进程间传参类，还有Manager这个大管家。
    * Quaue这一类都是严格的数据结构队列类型
    * Manager比较特殊，它提供了可以在进程间传递的列表、字典等python原生类型
* 还要考虑怎样才能达到处理进程可以在读取进程中得到最新的一帧:
  * 其实VideoCapture是一个天生的队列，先进先出。如果要达到实时获得最新帧的目的，就需要栈来存储视频帧，而不是队列。
    * 这样的话，Quaue这一大类就都没有可能了，肯定不能用它来传参。
    * 提到栈突然想到了python的列表，它的append和pop操作完全可以当”不严格“的栈来用。所以顺理成章地multiprocessing.Manager.list就是最好的进程间传参类型。
* 再就是传参栈自动清理的问题，压栈频率肯定是要比出栈频率高的，时间一长就会在栈中积累大量无法出栈的视频帧，会导致程序崩溃，这就需要有一个自动清理机制:
  * 设置一个传参栈容量，每当达到这个容量就直接把栈清空，再利用gc库手动发起一次python垃圾回收。这样就不会导致严重的内存溢出和程序崩溃。

***

## 实现代码

```python
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
    pw = Process(target=write, args=(q, "rtsp://xxx:xxx@192.168.1.102:554", 100))
    pr = Process(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    pr.start()

    # 等待pr结束:
    pr.join()

    # pw进程里是死循环，无法等待其结束，只能强行终止:
    pw.terminate()
```

实际上这个程序就是把VideoCapture的队列读取改成了栈读取。这个程序可以写成一个类，来作为一个新形式的VideoCapture。