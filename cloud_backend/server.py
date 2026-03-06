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
        cap = cv2.VideoCapture(self.video_source)
        # 尝试获取视频本身的 FPS
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        global ACTUAL_VIDEO_FPS, ANALYSIS_FRAME_ID # 引用全局变量
        if actual_fps > 0:
            print(f"[Info] Video source actual FPS: {actual_fps}")
            sleep_s = 1.0 / actual_fps
            ACTUAL_VIDEO_FPS = actual_fps
        else:
            sleep_s = 1.0 / self.capture_fps if self.capture_fps > 0 else 0.03
            ACTUAL_VIDEO_FPS = self.capture_fps

        global GLOBAL_FRAME, ANALYSIS_FRAME, RAW_FRAME
        frame_count = 0
        start_time = time.time()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # 增加追赶逻辑：如果处理线程太慢，导致 ANALYSIS_FRAME 积压，
            # 那么视频读取线程应该跳过一些帧，确保模型拿到的总是最新的帧
            with ANALYSIS_LOCK:
                ANALYSIS_FRAME = frame.copy()
                ANALYSIS_FRAME_ID += 1 # 每读取一帧，ID 自增一次
            
            with RAW_LOCK:
                RAW_FRAME = frame.copy()
            
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                GLOBAL_DATA["fps"]["video_fps"] = round(frame_count / elapsed, 1)
                frame_count = 0
                start_time = time.time()

            time.sleep(sleep_s)

        cap.release()


class TrafficAnalysisThread(threading.Thread):
    def __init__(
        self,
        model_name: str,
        device: str,
        mpp_values: list[float],
        region_polygon: np.ndarray,
        base_fps: float,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.mpp_values = mpp_values
        self.region_polygon = region_polygon
        self.base_fps = base_fps
        self.daemon = True
        self.running = True

        self.track_history = defaultdict(list)
        self.speed_hist = defaultdict(lambda: deque(maxlen=30))
        self.avg_gap_history = deque(maxlen=15)
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.last_processed_id = -1 # 追踪上一次处理的帧 ID

    def run(self):
        # Explicitly check GPU availability
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
            # 只在有新帧时才处理，且强制对齐视频原生帧率，防止 GPU 过载
            with ANALYSIS_LOCK:
                if ANALYSIS_FRAME is not None and ANALYSIS_FRAME_ID > self.last_processed_id:
                    frame = ANALYSIS_FRAME.copy()
                    self.last_processed_id = ANALYSIS_FRAME_ID
                
            if frame is None:
                time.sleep(0.01) 
                continue

            # 额外的精准休眠：确保推理频率不超过视频原生 FPS
            # 15 帧视频大约每 66ms 一帧，如果推理只用了 10ms，就多等一下
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

            # 2. Run YOLO
            start_time = time.time()
            try:
                # Use results for plotting
                # Check if we are using a custom classes list
                classes_to_track = [2, 3, 5, 7] # Default COCO: car, motorcycle, bus, truck
                if "best" in self.model_name: # Detect custom model by name (best.pt/onnx/engine)
                     # If using custom model, we might want to track all classes or specific ones
                     classes_to_track = None 

                # Speed Optimization: imgsz=640
                # For .pt/.onnx, we use half=True (FP16) for speed on GPU.
                # For .engine (TensorRT), precision is baked in during export, so we shouldn't force half=True.
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
                
                # Plotting: Draw YOLO annotations directly onto the frame
                # This returns a BGR numpy array
                annotated_frame = results[0].plot()
                
                # Update GLOBAL_FRAME with the ANNOTATED version
                # Note: We need to draw the ROI polygon on top (or before) as well
                cv2.polylines(annotated_frame, [self.region_polygon], True, (0, 255, 0), 2)
                
                with FRAME_LOCK:
                    global GLOBAL_FRAME
                    GLOBAL_FRAME = annotated_frame
                
                # Performance Logging
                infer_time = (time.time() - start_time) * 1000 # ms
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
                        # 使用检测到的真实视频 FPS 进行测速校准
                        v_mps = (pixel_dist * mpp) * (ACTUAL_VIDEO_FPS / frame_step)

                if v_mps > 0:
                    self.speed_hist[track_id].append(v_mps)
                avg_v_mps = (
                    sum(self.speed_hist[track_id]) / len(self.speed_hist[track_id])
                    if self.speed_hist[track_id]
                    else 0.0
                )

                active_vehicles.append({
                    "id": track_id, 
                    "y": c[1], 
                    "speed": avg_v_mps,
                    "class_name": names[cls_id]
                })

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
                )[:10] # 最多展示10个最新ID
            }

            # REMOVED sleep to maximize FPS
            # time.sleep(0.02)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/api/horizontal", methods=["GET"])
def get_horizontal_data():
    return jsonify({
        "data": GLOBAL_DATA["horizontal"],
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
        
        # 智能限速：发送频率不要超过 15 FPS
        # 如果发太快，网络缓冲区会堆积，导致巨大的延时
        now = time.time()
        min_send_interval = 1.0 / (ACTUAL_VIDEO_FPS + 2) # 允许稍微多一点，保证流畅
        if now - last_sent_time < min_send_interval:
            time.sleep(0.01)
            continue

        if frame is None:
            time.sleep(0.01)
            continue
        
        # 码率控制：针对手机端，进一步压缩画质提升流畅度
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
    # Auto-detect model: Look for .onnx first, then .pt
    detected_model = "yolov8n.pt" # Default
    
    # Priority: Env Var > .onnx in cwd > .pt in cwd > Default
    env_model = os.environ.get("MODEL_NAME")
    
    if env_model:
        detected_model = env_model
        print(f"[Info] Using model from environment: {detected_model}")
    else:
        # Search for models in current directory
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
            
    video_source = os.environ.get("VIDEO_SOURCE", "test1.mp4")
    model_name = detected_model
    device = os.environ.get("DEVICE", "cpu")
    capture_fps = float(os.environ.get("CAPTURE_FPS", "30"))
    base_fps = float(os.environ.get("BASE_FPS", "30"))

    region_polygon = np.array(
        [
            (876, 696), (924, 412), (1003, 336), (1089, 328), (1144, 441), (1339, 644), (881, 696),
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
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port)
