import cv2
import numpy as np

# 视频源
video_source = "test1.mp4"

# 检查视频
import os
if not os.path.exists(video_source):
    print(f"Error: 找不到文件 {video_source}")
    exit()

cap = cv2.VideoCapture(video_source)
ret, frame = cap.read()
if not ret:
    print("无法读取视频帧")
    exit()

# 获取原始尺寸
orig_h, orig_w = frame.shape[:2]
print(f"原始分辨率: {orig_w} x {orig_h}")

# 缩放比例 (如果宽度大于 1200，就缩小显示)
scale_factor = 1.0
if orig_w > 1200:
    scale_factor = 1200.0 / orig_w
    print(f"画面过大，将缩小 {scale_factor:.2f} 倍显示以适应屏幕。")
    print("放心选！选完后我会自动帮你还原回真实坐标。")

# 缩小后的尺寸
display_w = int(orig_w * scale_factor)
display_h = int(orig_h * scale_factor)

points = []

def mouse_callback(event, x, y, flags, param):
    global points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"添加点 (显示坐标): ({x}, {y})")
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            p = points.pop()
            print(f"撤销点: {p}")

cv2.namedWindow("Select Polygon", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select Polygon", display_w, display_h)
cv2.setMouseCallback("Select Polygon", mouse_callback)

print("=======================================================")
print("  【多边形选择指南】")
print("  1. 【左键点击】添加顶点。")
print("  2. 【右键点击】撤销上一个点。")
print("  3. 画完后，按【ENTER (回车键)】确认。")
print("=======================================================")

# 预先缩放一帧用于显示
small_frame = cv2.resize(frame, (display_w, display_h))

while True:
    display_img = small_frame.copy()
    
    if len(points) > 0:
        for p in points:
            cv2.circle(display_img, p, 5, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.polylines(display_img, [np.array(points)], False, (0, 255, 0), 2)
            
    cv2.imshow("Select Polygon", display_img)
    
    key = cv2.waitKey(20) & 0xFF
    if key == 13: # Enter
        if len(points) < 3:
            print("❌ 至少需要 3 个点！")
            continue
        break
    elif key == 27 or key == ord('c'): # Esc
        points = []
        break

cv2.destroyAllWindows()
cap.release()

if len(points) >= 3:
    # 还原坐标到原始分辨率
    real_points = []
    for (px, py) in points:
        rx = int(px / scale_factor)
        ry = int(py / scale_factor)
        real_points.append((rx, ry))
        
    print("\n✅ 你选择的真实坐标（已还原）：")
    print(real_points)
    print("-" * 30)
    print(f"代码复制用: region_polygon = {real_points}")
    print("-" * 30)
else:
    print("\n❌ 未选择有效区域。")
