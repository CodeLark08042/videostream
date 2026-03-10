import os
import threading
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("monitor.html")

GLOBAL_DATA = {
    "fps": {
        "video_fps": 0.0,    # 原视频读取帧率
        "inference_fps": 0.0 # 模型推理/展示帧率
    },
    "horizontal": {
        "avg_speed": 0.0,
        "density": 0.0,
        "vehicle_count": 0,
        "status": "检测中...",
        "warning": "",
        "vehicle_list": [] # 用于滚动展示 individual speed
    },
    "vertical": {
        "main_road_status": "检测中...",
        "merge_gap": 0.0,
        "risk_level": "中",
        "advice": "分析中...",
        "can_merge": False,
    },
    "id_analysis": {
        "selected_id": -1,
        "ahead_count": 0,
        "ahead_density": 0.0,
        "direction": "未知"
    }
}

GLOBAL_FRAME = None
FRAME_LOCK = threading.Lock()

RAW_FRAME = None
RAW_LOCK = threading.Lock()

ANALYSIS_FRAME = None
ANALYSIS_FRAME_ID = 0 # 增加帧 ID 追踪
ANALYSIS_LOCK = threading.Lock()

# 全局视频帧率，用于跨线程同步测速计算
ACTUAL_VIDEO_FPS = 30.0 

class PixelMeterBands:
    def __init__(self, h: int, mpp_values: list[float]):
        self.h = h
        self.mpp_values = mpp_values
        n = len(mpp_values)
        self.bounds = []
        step = h / n
        for i in range(n):
            y1 = i * step
            y2 = (i + 1) * step if i < n - 1 else float(h)
            self.bounds.append((y1, y2, mpp_values[i]))

    def mpp_at(self, y: float) -> float:
        y = max(0, min(self.h, y))
        for y1, y2, mpp in self.bounds:
            if y1 <= y <= y2:
                return mpp
        return self.bounds[-1][2]

    def length_m(self, y_min: float, y_max: float) -> float:
        y_min = max(0, min(self.h, y_min))
        y_max = max(0, min(self.h, y_max))
        if y_max <= y_min:
            return 0.0
        s = 0.0
        for y1, y2, mpp in self.bounds:
            a = max(y1, y_min)
            b = min(y2, y_max)
            if b > a:
                s += (b - a) * mpp
        return s


class VideoCaptureThread(threading.Thread):
    def __init__(self, video_source: str, capture_fps: float, region_polygon: np.ndarray):
        super().__init__()
        self.video_source = video_source
        self.capture_fps = capture_fps
        self.region_polygon = region_polygon
        self.daemon = True
        self.running = True

    def run(self):
        # 修改：使用 RTMP 流地址
        print(f"[Info] 正在连接 RTMP 流: {self.video_source}")
        cap = cv2.VideoCapture(self.video_source)
        
        # 尝试获取视频本身的 FPS
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        global ACTUAL_VIDEO_FPS, ANALYSIS_FRAME_ID 
        if actual_fps > 0:
            print(f"[Info] Video source actual FPS: {actual_fps}")
            sleep_s = 1.0 / actual_fps
            ACTUAL_VIDEO_FPS = actual_fps
        else:
            # RTMP 流可能无法获取准确 FPS，默认设为 25 或 30
            print("[Warning] 无法获取 RTMP 流 FPS，默认使用 25.0")
            sleep_s = 1.0 / 25.0
            ACTUAL_VIDEO_FPS = 25.0

        global GLOBAL_FRAME, ANALYSIS_FRAME, RAW_FRAME
        frame_count = 0
        start_time = time.time()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                # RTMP 流断开时，尝试重连
                print("[Warning] RTMP 流中断，3秒后尝试重连...")
                time.sleep(3)
                cap.release()
                cap = cv2.VideoCapture(self.video_source)
                continue

            with ANALYSIS_LOCK:
                ANALYSIS_FRAME = frame.copy()
                ANALYSIS_FRAME_ID += 1 
            
            with RAW_LOCK:
                RAW_FRAME = frame.copy()
            
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                GLOBAL_DATA["fps"]["video_fps"] = round(frame_count / elapsed, 1)
                frame_count = 0
                start_time = time.time()

            # RTMP 是实时流，通常不需要手动 sleep，read() 本身会阻塞
            # 但为了防止 CPU 占用过高，保留微小 sleep
            time.sleep(0.005)

        cap.release()


class TrafficAnalysisThread(threading.Thread):
    def __init__(
        self,
        model_name: str,
        device: str,
        mpp_values: list[float],
        region_polygon: np.ndarray,
        base_fps: float,
        orig_res: tuple[int, int] = (1920, 1080) # 默认之前的标注是基于 1080p 的
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.mpp_values = mpp_values
        self.base_region_polygon = region_polygon.copy()
        self.region_polygon = region_polygon
        self.orig_res = orig_res
        self.base_fps = base_fps
        self.daemon = True
        self.running = True

        self.track_history = defaultdict(list)
        self.speed_hist = defaultdict(lambda: deque(maxlen=30))
        self.avg_gap_history = deque(maxlen=15)
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.last_processed_id = -1 
        self.res_initialized = False

    def run(self):
        import torch
        print(f"DEBUG: Requested device: {self.device}")
        if self.device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available! Fallback to CPU.")
            self.device = "cpu"
        elif self.device == "cuda":
            print(f"SUCCESS: Using GPU: {torch.cuda.get_device_name(0)}")
            
        print(f"Loading YOLO model: {self.model_name}")
        model = YOLO(self.model_name)
        bands = None
        roi_len_m = 0.0

        while self.running:
            frame = None
            with ANALYSIS_LOCK:
                if ANALYSIS_FRAME is not None and ANALYSIS_FRAME_ID > self.last_processed_id:
                    frame = ANALYSIS_FRAME.copy()
                    self.last_processed_id = ANALYSIS_FRAME_ID
                
            if frame is None:
                time.sleep(0.01) 
                continue

            # 自动调整标注框到当前分辨率
            if not self.res_initialized:
                curr_h, curr_w = frame.shape[:2]
                orig_w, orig_h = self.orig_res
                if curr_w != orig_w or curr_h != orig_h:
                    scale_x = curr_w / orig_w
                    scale_y = curr_h / orig_h
                    print(f"[Info] 自动调整坐标缩放: X*{scale_x:.2f}, Y*{scale_y:.2f} (源 {orig_w}x{orig_h} -> 当前 {curr_w}x{curr_h})")
                    
                    # 缩放 ROI 坐标
                    self.region_polygon = self.base_region_polygon.copy().astype(np.float32)
                    self.region_polygon[:, 0] *= scale_x
                    self.region_polygon[:, 1] *= scale_y
                    self.region_polygon = self.region_polygon.astype(np.int32)
                    
                    # 缩放 MPP (Meters Per Pixel) 
                    # 如果分辨率缩小，每个像素代表的实际距离变大
                    self.mpp_values = [mpp / scale_y for mpp in self.mpp_values]
                self.res_initialized = True

            elapsed_since_last = time.time() - self.fps_start_time
            min_interval = 1.0 / ACTUAL_VIDEO_FPS
            if elapsed_since_last < min_interval:
                time.sleep(min_interval - elapsed_since_last)

            if bands is None:
                h = frame.shape[0]
                bands = PixelMeterBands(h, self.mpp_values)
                roi_y_min = int(np.min(self.region_polygon[:, 1]))
                roi_y_max = int(np.max(self.region_polygon[:, 1]))
                roi_len_m = bands.length_m(roi_y_min, roi_y_max)

            start_time = time.time()
            try:
                classes_to_track = [2, 3, 5, 7] 
                if "best" in self.model_name: 
                     classes_to_track = None 

                is_engine = self.model_name.endswith(".engine")
                use_half = False if is_engine else True
                
                results = model.track(
                    frame, 
                    persist=True, 
                    verbose=False, 
                    classes=classes_to_track, 
                    device=self.device,
                    imgsz=640,
                    half=use_half
                )
                
                annotated_frame = results[0].plot()
                cv2.polylines(annotated_frame, [self.region_polygon], True, (0, 255, 0), 2)
                
                # 注意：这里移除了 GLOBAL_FRAME 的赋值，移到了后面 ID 分析之后
                
                infer_time = (time.time() - start_time) * 1000 
                self.fps_frame_count += 1
                now = time.time()
                elapsed = now - self.fps_start_time
                if elapsed >= 1.0:
                    GLOBAL_DATA["fps"]["inference_fps"] = round(self.fps_frame_count / elapsed, 1)
                    self.fps_frame_count = 0
                    self.fps_start_time = now

                if self.frame_count % 30 == 0: 
                    print(f"[GPU Analysis] Inference: {infer_time:.1f}ms | FPS: {1000/infer_time:.1f}")
                self.frame_count += 1
                
            except Exception as e:
                results = None

            if not results or results[0].boxes.id is None:
                GLOBAL_DATA["horizontal"] = {
                    "avg_speed": 0.0,
                    "density": 0.0,
                    "vehicle_count": 0,
                    "status": "检测中...",
                    "warning": "",
                }
                time.sleep(0.02)
                continue

            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            names = results[0].names
            active_vehicles = []

            # 获取选中的 ID
            selected_id = GLOBAL_DATA["id_analysis"]["selected_id"]
            selected_vehicle = None

            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                x, y, w_box, h_box = box
                c = (float(x), float(y))

                if cv2.pointPolygonTest(self.region_polygon, c, False) < 0:
                    continue

                track = self.track_history[track_id]
                track.append(c)
                if len(track) > 30:
                    track.pop(0)

                v_mps = 0.0
                frame_step = 5
                if len(track) > frame_step:
                    p0 = track[-(frame_step + 1)]
                    p1 = track[-1]
                    pixel_dist = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
                    if pixel_dist > 1.0:
                        avg_y = (p0[1] + p1[1]) / 2.0
                        mpp = bands.mpp_at(avg_y)
                        v_mps = (pixel_dist * mpp) * (ACTUAL_VIDEO_FPS / frame_step)

                if v_mps > 0:
                    self.speed_hist[track_id].append(v_mps)
                avg_v_mps = (
                    sum(self.speed_hist[track_id]) / len(self.speed_hist[track_id])
                    if self.speed_hist[track_id]
                    else 0.0
                )

                v_info = {
                    "id": track_id, 
                    "x": c[0],
                    "y": c[1], 
                    "speed": avg_v_mps,
                    "class_name": names[cls_id]
                }
                active_vehicles.append(v_info)
                
                if track_id == selected_id:
                    selected_vehicle = v_info

            # 分析选中 ID 的前方车辆
            ahead_count = 0
            ahead_density = 0.0
            direction_str = "静止"
            
            if selected_vehicle:
                track = self.track_history[selected_id]
                if len(track) > 5:
                    p_old = track[-6]
                    p_new = track[-1]
                    vx = p_new[0] - p_old[0]
                    vy = p_new[1] - p_old[1]
                    
                    # 在画面上画出方向箭头
                    arrow_len = 50
                    dist = np.hypot(vx, vy)
                    if dist > 1.0:
                        dx = int(vx / dist * arrow_len)
                        dy = int(vy / dist * arrow_len)
                        cv2.arrowedLine(
                            annotated_frame, 
                            (int(p_new[0]), int(p_new[1])), 
                            (int(p_new[0] + dx), int(p_new[1] + dy)), 
                            (0, 0, 255), 3, tipLength=0.3
                        )
                        # 简单判定方向
                        if abs(vy) > abs(vx):
                            direction_str = "纵向" + ("前进" if vy < 0 else "后退")
                        else:
                            direction_str = "横向" + ("左行" if vx < 0 else "右行")

                    # 判定前方车辆：简化逻辑，判定在运动向量方向上的车
                    # 假设主要沿 Y 轴运动（交通流常见情况）
                    is_moving_up = vy < 0 
                    
                    ahead_vehicles = []
                    for v in active_vehicles:
                        if v["id"] == selected_id: continue
                        
                        # 判定是否在前方（Y轴方向）且距离不太远（300像素内）
                        if is_moving_up:
                            if v["y"] < selected_vehicle["y"] and (selected_vehicle["y"] - v["y"]) < 300:
                                ahead_vehicles.append(v)
                        else:
                            if v["y"] > selected_vehicle["y"] and (v["y"] - selected_vehicle["y"]) < 300:
                                ahead_vehicles.append(v)
                    
                    ahead_count = len(ahead_vehicles)
                    # 计算局部密度（假设前方 50 米范围）
                    # 这里简化处理，直接用数量/固定长度
                    ahead_density = ahead_count / 0.05 # 辆/公里
                    
                    # 在画面上高亮选中车辆
                    cv2.circle(annotated_frame, (int(selected_vehicle["x"]), int(selected_vehicle["y"])), 20, (255, 0, 0), 2)

            GLOBAL_DATA["id_analysis"].update({
                "ahead_count": ahead_count,
                "ahead_density": round(ahead_density, 1),
                "direction": direction_str
            })
            
            # 最后将带有分析结果（箭头和高亮圈）的画面交给展示线程
            with FRAME_LOCK:
                global GLOBAL_FRAME
                GLOBAL_FRAME = annotated_frame

            if active_vehicles:
                speeds = [v["speed"] for v in active_vehicles]
                avg_speed_kmh = (sum(speeds) / len(speeds)) * 3.6
                density = len(active_vehicles) / (roi_len_m / 1000.0) if roi_len_m > 1 else 0.0
            else:
                avg_speed_kmh = 0.0
                density = 0.0

            h_status = "畅通"
            h_warning = ""
            if density > 80:
                h_status = "拥堵"
                h_warning = "路段拥堵，汇入车辆风险提升，谨慎通行"
            elif avg_speed_kmh < 20 and density > 40:
                h_status = "缓行"
                h_warning = "前方T型路口，支路盲区可能有车辆汇入，请注意减速观察"

            GLOBAL_DATA["horizontal"] = {
                "avg_speed": round(avg_speed_kmh, 1),
                "density": round(density, 1),
                "vehicle_count": len(active_vehicles),
                "status": h_status,
                "warning": h_warning,
                "vehicle_list": sorted(
                    [
                        {
                            "id": v["id"], 
                            "speed": round(v["speed"] * 3.6, 1),
                            "class_name": v["class_name"]
                        } 
                        for v in active_vehicles
                    ],
                    key=lambda x: x["id"],
                    reverse=True
                )[:10] 
            }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/api/select_id", methods=["POST"])
def select_id():
    from flask import request
    data = request.json
    selected_id = data.get("id", -1)
    GLOBAL_DATA["id_analysis"]["selected_id"] = int(selected_id)
    print(f"[Info] 已选中目标 ID: {selected_id}")
    return jsonify({"ok": True, "selected_id": selected_id})


@app.route("/api/horizontal", methods=["GET"])
def get_horizontal_data():
    return jsonify({
        "data": GLOBAL_DATA["horizontal"],
        "id_analysis": GLOBAL_DATA["id_analysis"],
        "fps": GLOBAL_DATA["fps"]
    })


@app.route("/api/vertical", methods=["GET"])
def get_vertical_data():
    return jsonify({
        "data": GLOBAL_DATA["vertical"],
        "fps": GLOBAL_DATA["fps"]
    })


def _encode_stream_frame(frame: np.ndarray, max_w: int, quality: int) -> bytes:
    h, w = frame.shape[:2]
    if w > max_w:
        scale = max_w / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return buffer.tobytes() if ok else b""


def generate_frames():
    last_sent_time = 0
    while True:
        with FRAME_LOCK:
            frame = GLOBAL_FRAME
        
        now = time.time()
        min_send_interval = 1.0 / (ACTUAL_VIDEO_FPS + 2) 
        if now - last_sent_time < min_send_interval:
            time.sleep(0.01)
            continue

        if frame is None:
            time.sleep(0.01)
            continue
        
        payload = _encode_stream_frame(frame, max_w=960, quality=50)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
        )
        last_sent_time = now


def generate_raw_frames():
    last_sent_time = 0
    while True:
        with RAW_LOCK:
            frame = RAW_FRAME
        
        now = time.time()
        min_send_interval = 1.0 / (ACTUAL_VIDEO_FPS + 2)
        if now - last_sent_time < min_send_interval:
            time.sleep(0.01)
            continue

        if frame is None:
            time.sleep(0.01)
            continue
            
        payload = _encode_stream_frame(frame, max_w=960, quality=50)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
        )
        last_sent_time = now


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_raw")
def video_raw():
    return Response(generate_raw_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/frame.jpg")
def frame_jpg():
    with FRAME_LOCK:
        frame = GLOBAL_FRAME
    if frame is None:
        blank = np.zeros((360, 480, 3), dtype=np.uint8)
        payload = _encode_stream_frame(blank, max_w=480, quality=30)
    else:
        payload = _encode_stream_frame(frame, max_w=480, quality=30)
    resp = Response(payload, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


if __name__ == "__main__":
    detected_model = "yolov8n.pt" 
    
    env_model = os.environ.get("MODEL_NAME")
    
    if env_model:
        detected_model = env_model
        print(f"[Info] Using model from environment: {detected_model}")
    else:
        files = os.listdir(".")
        engine_models = [f for f in files if f.endswith(".engine")]
        onnx_models = [f for f in files if f.endswith(".onnx")]
        pt_models = [f for f in files if f.endswith(".pt") and "yolov8n.pt" not in f]
        
        if engine_models:
            detected_model = engine_models[0]
            print(f"[Info] Auto-detected TensorRT model: {detected_model} (Fastest)")
        elif onnx_models:
            detected_model = onnx_models[0]
            print(f"[Info] Auto-detected ONNX model: {detected_model}")
        elif pt_models:
            detected_model = pt_models[0]
            print(f"[Info] Auto-detected PT model: {detected_model}")
        else:
            print(f"[Info] No custom model found, using default: {detected_model}")
            
    # 修改默认视频源为 RTMP 流地址
    video_source = os.environ.get("VIDEO_SOURCE", "rtmp://localhost/live/stream")
    model_name = detected_model
    device = os.environ.get("DEVICE", "cpu")
    capture_fps = float(os.environ.get("CAPTURE_FPS", "30"))
    base_fps = float(os.environ.get("BASE_FPS", "30"))

    region_polygon = np.array(
        [
            (832, 1054), (897, 500), (995, 347), (1115, 342), (1129, 419), (1668, 1028), (833, 1056),
        ],
        np.int32,
    )

    capture_thread = VideoCaptureThread(
        video_source=video_source, capture_fps=capture_fps, region_polygon=region_polygon
    )
    capture_thread.start()

    analysis_thread = TrafficAnalysisThread(
        model_name=model_name,
        device=device,
        mpp_values=[0.05, 0.025, 0.01],
        region_polygon=region_polygon,
        base_fps=base_fps,
    )
    analysis_thread.start()

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "6006")) # 默认改为 6006，避免与 5000 冲突
    print(f"Starting Flask server on port {port}...")
    app.run(host=host, port=port, threaded=True)
