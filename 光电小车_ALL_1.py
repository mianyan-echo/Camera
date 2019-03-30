'''
画出红色图形的轮廓，其中很多参数待调，计划通过TK设立滑块来实时调节参数
找出轮廓中心点
加入控制模块结构
'''

import cv2
import numpy as np

# 定义接口模块
# 后续改写为返回True/False表明是否成功执行
def Turn(key = 0):
    """
    左转右转控制模块，控制左右转电机。
    :param key: 输入接口
    :return: control command
    """
    if key == 1:
        return "左转"
    elif key == 0:
        return "直行"
    elif key == (-1):
        return "右转"

def Go_on(key):
    """
    前行后退停机控制
    :param key: 输入接口
    :return: control command
    """
    if key == 1:
        return "前行"
    elif key == 0:
        return "停机"
    elif key == (-1):
        return "后退"

cap = cv2.VideoCapture("rtsp://admin:Zxcvbnm123@192.168.1.103:554/ONVIFMedia")  # 读取摄像头

while True:
    # 读取下一帧解码本帧
    ret, im = cap.read()

    # 用numpy建立一个和视频一样长宽的全黑画布
    # 便于后边将图像画于画布上
    pri = np.zeros(im.shape[:2], np.uint8)

    # 将图像转化到HSV色彩空间格式，便于提取颜色
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    cv2.imshow("hsv", hsv)

    # 制作透过红色的蒙版
    lower_red = np.array([0, 0, 0])
    upper_red = np.array([255, 255, 255])
    maskR = cv2.inRange(hsv, lower_red, upper_red)

    # 将蒙版与图像结合，得到只透红色的图像
    res = cv2.bitwise_and(im, im, mask=maskR)

    # 画出图像中红色图形的外轮廓
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)  # 将处理后的图像转换为灰度图，便于二值化
    re, thresh = cv2.threshold(gray, 1, 1, cv2.THRESH_TOZERO)  # 将图像二值化，此函数只接受灰度图
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 在二值化的基础上找出轮廓
    # image为thresh图像，contours为轮廓点的坐标，hierarchy为轮廓索引
    # 目前为止，hierarchy属性显得非常鸡肋，因为是动态视频，每次屏幕刷新索引号都不一样，所以无法应用索引号去噪

    if contours == []:
        """
        这里一定要先判断轮廓是否为空，cv2.moments()函数是不能接收空轮廓的。
        后面这里可以加上输出接口，向右转、向后退、板子收起。
        """
        continue
    else:
        """
        同样，这里也可以加入输出接口
        """
        M = cv2.moments(contours[0])
        # 返回一个字典
        if int(M['m00']) == 0:  # 这个判断会屏蔽中点在边缘的情况
            continue
        else:  # 找出中点坐标
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            print(cx, cy)
            """
            # 根据中心点输出左转右转指令
            # 这里有一个需要考虑的地方，是否可以该成根据cx与视频中心线差距来判断拐弯幅度。
            if cx < im.shape[1] / 2:
                print(Turn(1),Go_on(1))
            elif cx > im.shape[1] / 2:
                print(Turn(-1),Go_on(1))
            elif cx == im.shape[1] / 2:
                print(Turn(0),Go_on(1))
            """

    # 在画布上画出轮廓，做出中心点
    img = cv2.drawContours(pri, contours, -1, (255, 255, 255), 1)
    cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1)
    cv2.imshow("video", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()