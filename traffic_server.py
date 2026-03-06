import threading
import time
import cv2
import numpy as np
from collections import defaultdict, deque
from flask import Flask, jsonify, Response
from flask_cors import CORS
from ultralytics import YOLO

print("DEBUG: Script starting...")

app = Flask(__name__)
CORS(app)

# --- Global Shared Data ---
GLOBAL_DATA = {
    "horizontal": {
        "avg_speed": 0.0,
        "density": 0.0,
        "vehicle_count": 0,
        "status": "检测中...",
        "warning": ""
    },
    "vertical": {
        "main_road_status": "检测中...",
        "merge_gap": 0.0,
        "risk_level": "中",
        "advice": "分析中...",
        "can_merge": False
    }
}

# --- Frame Buffers ---
# GLOBAL_FRAME: The frame shown to the user (High FPS)
GLOBAL_FRAME = None
FRAME_LOCK = threading.Lock()

# ANALYSIS_FRAME: The frame used for YOLO analysis (Decoupled FPS)
ANALYSIS_FRAME = None
ANALYSIS_LOCK = threading.Lock()

# --- Helper Classes ---
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

# --- Thread 1: Video Capture (High FPS) ---
class VideoCaptureThread(threading.Thread):
    def __init__(self, video_source="test1.mp4"):
        super().__init__()
        self.video_source = video_source
        self.daemon = True
        self.running = True
        
        # Region (User defined polygon) - Visualization Only
        self.region_polygon = np.array([(960, 393), (1096, 395), (1564, 1009), (920, 1040), (960, 393)], np.int32)

    def run(self):
        print(f"Opening video source: {self.video_source}...")
        cap = cv2.VideoCapture(self.video_source)
        
        print("Starting Video Capture Loop (High FPS)...")
        global GLOBAL_FRAME, ANALYSIS_FRAME
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Video ended. Looping...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # 1. Update Display Frame immediately (for smooth video)
            # Draw ROI on display frame
            display_frame = frame.copy()
            cv2.polylines(display_frame, [self.region_polygon], True, (0, 255, 0), 2)
            
            with FRAME_LOCK:
                GLOBAL_FRAME = display_frame
            
            # 2. Update Analysis Frame (Consumer will pick it up)
            with ANALYSIS_LOCK:
                ANALYSIS_FRAME = frame.copy() # Raw frame for analysis
            
            # Control Capture FPS (e.g. 30 FPS)
            time.sleep(0.03)
            
        cap.release()

# --- Thread 2: Traffic Analysis (Lower FPS) ---
class TrafficAnalysisThread(threading.Thread):
    def __init__(self, model_name="yolov8n.pt"):
        super().__init__()
        self.model_name = model_name
        self.daemon = True
        self.running = True
        
        # Config parameters
        self.mpp_values = [0.05, 0.025, 0.01] 
        self.speed_window_s = 5
        self.conf = 0.3
        
        # State
        self.track_history = defaultdict(lambda: [])
        self.speed_hist = defaultdict(lambda: deque(maxlen=30))
        self.avg_gap_history = deque(maxlen=15)
        
        # Region (User defined polygon)
        self.region_polygon = np.array([(960, 393), (1096, 395), (1564, 1009), (920, 1040), (960, 393)], np.int32)

    def run(self):
        print(f"Loading YOLO model: {self.model_name}...")
        model = YOLO(self.model_name)
        
        # Initialize Bands (Assume 1080p for simplicity or wait for first frame)
        # We'll init with standard 1080p height
        h = 1080 
        bands = PixelMeterBands(h, self.mpp_values)
        
        # Calculate ROI length
        roi_y_min = min([p[1] for p in self.region_polygon])
        roi_y_max = max([p[1] for p in self.region_polygon])
        roi_len_m = bands.length_m(roi_y_min, roi_y_max)
        print(f"ROI Polygon Set. Length approx: {roi_len_m:.1f}m")
        
        print("Starting Analysis Loop (Decoupled FPS)...")
        global ANALYSIS_FRAME
        
        # FPS estimation
        fps = 30.0 # Assumed base FPS for speed calc
        
        while self.running:
            # 1. Get latest frame
            frame = None
            with ANALYSIS_LOCK:
                if ANALYSIS_FRAME is not None:
                    frame = ANALYSIS_FRAME.copy()
            
            if frame is None:
                time.sleep(0.1)
                continue
                
            # 2. Run YOLO
            start_time = time.time()
            try:
                results = model.track(frame, persist=True, verbose=False, classes=[2, 3, 5, 7], device='cpu')
            except Exception as e:
                print(f"Tracking error: {e}")
                results = None
                
            # 3. Process Results
            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                active_vehicles = []
                
                for box, track_id in zip(boxes, track_ids):
                    x, y, w_box, h_box = box
                    c = (float(x), float(y))
                    
                    is_inside = cv2.pointPolygonTest(self.region_polygon, c, False) >= 0
                    if not is_inside: continue

                    track = self.track_history[track_id]
                    track.append(c)
                    if len(track) > 30: track.pop(0)
                    
                    # Calculate Speed
                    v_mps = 0.0
                    frame_step = 5 
                    if len(track) > frame_step:
                        p0 = track[-(frame_step + 1)]
                        p1 = track[-1]
                        pixel_dist = np.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)
                        if pixel_dist > 1.0:
                            avg_y = (p0[1] + p1[1]) / 2
                            mpp = bands.mpp_at(avg_y)
                            v_mps = (pixel_dist * mpp) * (fps / frame_step)
                    
                    if v_mps > 0:
                         self.speed_hist[track_id].append(v_mps)
                    
                    if self.speed_hist[track_id]:
                        avg_v_mps = sum(self.speed_hist[track_id]) / len(self.speed_hist[track_id])
                    else:
                        avg_v_mps = 0.0
                    
                    active_vehicles.append({
                        "id": track_id,
                        "y": c[1],
                        "speed": avg_v_mps
                    })
                
                # Metrics
                if active_vehicles:
                    speeds = [v["speed"] for v in active_vehicles]
                    avg_speed_kmh = (sum(speeds) / len(speeds)) * 3.6
                    density = len(active_vehicles) / (roi_len_m / 1000.0) if roi_len_m > 1 else 0
                else:
                    avg_speed_kmh = 0.0
                    density = 0.0
                
                # Update Global Data
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
                    "warning": h_warning
                }
            
            # Control Analysis FPS (Don't hog CPU)
            # If YOLO takes 0.2s, this sleep is negligible.
            # If YOLO is fast, this limits it to ~10 FPS to save CPU for video thread
            time.sleep(0.05)


# --- Flask Routes ---
@app.route('/api/horizontal', methods=['GET'])
def get_horizontal_data():
    return jsonify(GLOBAL_DATA["horizontal"])

@app.route('/api/vertical', methods=['GET'])
def get_vertical_data():
    return jsonify(GLOBAL_DATA["vertical"])

def generate_frames():
    while True:
        with FRAME_LOCK:
            if GLOBAL_FRAME is None:
                time.sleep(0.1)
                continue
            
            # Streaming Optimization:
            # 1. Resize to max 480px width
            frame_stream = GLOBAL_FRAME
            h, w = frame_stream.shape[:2]
            if w > 480:
                scale = 480 / w
                frame_stream = cv2.resize(frame_stream, (int(w*scale), int(h*scale)))
            
            # 2. Compress aggressively (Quality 30)
            ret, buffer = cv2.imencode('.jpg', frame_stream, [cv2.IMWRITE_JPEG_QUALITY, 30])
            frame_bytes = buffer.tobytes()
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/frame.jpg')
def frame_jpg():
    with FRAME_LOCK:
        frame = GLOBAL_FRAME
        if frame is None:
            blank = np.zeros((360, 480, 3), dtype=np.uint8)
            ok, buffer = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 30])
        else:
            h, w = frame.shape[:2]
            # Aggressively resize for mobile streaming (Max width 480px)
            if w > 480:
                scale = 480 / w
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            # Aggressively compress (Quality 30)
            ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 30])
            
    data = buffer.tobytes() if ok else b""
    resp = Response(data, mimetype='image/jpeg')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    return resp

if __name__ == '__main__':
    import os
    source = "test1.mp4"
    if not os.path.exists(source):
        source = 0
    
    # Start Threads
    capture_thread = VideoCaptureThread(video_source=source)
    capture_thread.start()
    
    analysis_thread = TrafficAnalysisThread()
    analysis_thread.start()
    
    print("Starting Flask Server (Dual-Thread Mode)...")
    app.run(host='0.0.0.0', port=5000)
