import cv2
import numpy as np

# 在这里填入你要标注的视频文件名
video_path = "test1.mp4" 

# 全局变量存储点击的坐标
points = []

def click_event(event, x, y, flags, params):
    # 鼠标左键点击事件
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"选中坐标: ({x}, {y})")
        
        # 在图像上画个圈标记
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image', img)

    # 鼠标右键点击事件（结束并打印结果）
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("\n=== 最终坐标数组 (请复制到 server.py) ===")
        print("region_polygon = np.array([")
        for p in points:
            print(f"    ({p[0]}, {p[1]}),")
        # 自动闭合多边形（首尾相连）
        if len(points) > 0:
            print(f"    ({points[0][0]}, {points[0][1]}),")
        print("], np.int32)")
        print("=========================================\n")

# 读取视频的第一帧
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"错误: 无法打开视频 {video_path}")
    exit()

ret, img = cap.read()
cap.release()

if not ret:
    print("错误: 无法读取视频帧")
    exit()

# 显示图像并绑定鼠标事件
print("=== 操作说明 ===")
print("1. 鼠标【左键】点击画面：选择多边形的顶点（按顺时针或逆时针顺序）")
print("2. 鼠标【右键】点击画面：完成选择，并在控制台打印出可直接复制的代码")
print("3. 按任意键退出窗口")
print("================\n")

cv2.imshow('Image', img)
cv2.setMouseCallback('Image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
