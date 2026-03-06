import time
import os
from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors

DEFAULT_CONFIG = {
    "model": "yolov8n.pt",
    "source": ".\\test1.mp4",
    "classes": None,
    "region": None,
    "mpp_values": [0.30, 0.18, 0.10],
    "speed_window_s": 5,  # Reduced window for faster response
    "conf": 0.3,
    "iou": 0.5,
    "device": None,
    "show": True,
    "tracker": "bytetrack.yaml",
    "fps": 30.0,
    "name_filter": ["car", "person","bicycle","motorcycle","bus","truck"],
}


class PixelMeterBands:
    def __init__(self, h: int, mpp_values: list[float]):
        self.h = h
        self.mpp_values = mpp_values
        n = len(mpp_values)
        self.bounds = []
        step = h / n  # Use float division
        for i in range(n):
            y1 = i * step
            y2 = (i + 1) * step if i < n - 1 else float(h)
            self.bounds.append((y1, y2, mpp_values[i]))

    def mpp_at(self, y: float) -> float:
        y = max(0, min(self.h, y))
        for y1, y2, mpp in self.bounds:
            if y1 <= y <= y2:  # Inclusive upper bound for last segment
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


class TrafficSpeedDensity(BaseSolution):
    def __init__(self, **kwargs):
        self.mpp_values = kwargs.pop("mpp_values", [0.30, 0.18, 0.10])
        self.speed_window_s = kwargs.pop("speed_window_s", 30)
        # Normalize name filter to lowercase
        nf = kwargs.pop("name_filter", None)
        self.name_filter = set(n.lower() for n in nf) if nf else None
        
        super().__init__(**kwargs)
        
        self.region_initialized = False
        self.margin = self.line_width * 2
        self.bands = None
        self.fps = self.CFG.get("fps", 30.0)
        self.window_len = max(1, int(self.fps * self.speed_window_s))
        self.speed_hist = defaultdict(lambda: deque(maxlen=self.window_len))
        self.active_ids = set()
        
        # Speed smoothing
        self.smooth_alpha = 0.3  # Exponential moving average factor

    def initialize_region(self):
        # Override BaseSolution's initialize_region to avoid default small box
        # This will be called in process() after we set self.region to full screen
        super().initialize_region()

    def process(self, im0: np.ndarray) -> SolutionResults:
        h, w = im0.shape[:2]
        
        # Initialize region to full screen if not provided
        if not self.region_initialized:
            if self.region is None:
                print(f"Debug: Initializing full screen region: {w}x{h}")
                self.region = [(0, 0), (w, 0), (w, h), (0, h)]
            self.initialize_region()
            self.region_initialized = True

        if self.bands is None:
            self.bands = PixelMeterBands(h, self.mpp_values)

        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)
        
        # Draw region (optional, can comment out if distracting)
        annotator.draw_region(reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width)
        
        # Draw calibration lines
        for _, y2, mpp in self.bands.bounds[:-1]:
            y = int(y2)
            cv2.line(im0, (0, y), (w, y), (200, 200, 200), 1)
            cv2.putText(im0, f"MPP: {mpp:.2f}", (10, y - 5), 0, 0.5, (200, 200, 200), 1)

        active = set()
        current_frame_speeds = []

        # Debug: Print detected classes
        # if self.frame_no % 30 == 0:
        #     detected = [self.names[c] for c in self.clss]
        #     print(f"Frame {self.frame_no}: Detected {len(detected)} objects: {Counter(detected)}")

        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            class_name = self.names[cls]
            # Name filter check (case-insensitive)
            if self.name_filter is not None:
                if class_name.lower() not in self.name_filter:
                    continue

            annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(cls, True))
            self.store_tracking_history(track_id, box)
            
            c = self.track_history[track_id][-1]
            
            # Check if inside region
            in_region = True
            if len(self.region) >= 3:
                if cv2.pointPolygonTest(np.array(self.region, np.int32), (int(c[0]), int(c[1])), False) < 0:
                    in_region = False
            
            if in_region:
                active.add(track_id)
                
                # Calculate speed
                if len(self.track_history[track_id]) > 1:
                    p0 = np.array(self.track_history[track_id][-2], dtype=np.float32)
                    p1 = np.array(self.track_history[track_id][-1], dtype=np.float32)
                    
                    pixel_dist = float(np.linalg.norm(p1 - p0))
                    
                    # Filter small jitters
                    if pixel_dist < 1.0: 
                        v_mps = 0.0
                    else:
                        # Use average Y for MPP lookup
                        avg_y = (p0[1] + p1[1]) * 0.5
                        mpp = float(self.bands.mpp_at(avg_y))
                        v_mps = pixel_dist * mpp * self.fps
                    
                    # Smooth speed
                    prev_hist = self.speed_hist[track_id]
                    if prev_hist:
                        # Simple moving average or exponential smoothing could be used
                        # Here we just append to history and average later
                        pass
                    
                    self.speed_hist[track_id].append(v_mps)

        self.active_ids = active
        
        # Calculate Metrics
        y_vals = [p[1] for p in self.region]
        roi_len_m = self.bands.length_m(min(y_vals), max(y_vals)) if len(self.region) >= 3 else 0.0
        
        avg_speed_kmh = 0.0
        valid_speed_count = 0
        
        for tid in self.active_ids:
            hist = self.speed_hist[tid]
            if len(hist) >= 5: # Require at least 5 frames of history for stable speed
                s = sum(hist) / len(hist)
                current_frame_speeds.append(s)
        
        if current_frame_speeds:
            avg_speed_kmh = (sum(current_frame_speeds) / len(current_frame_speeds)) * 3.6
        
        # Density: vehicles per km
        # ROI length is in meters. density = count / (len_m / 1000)
        density = 0.0
        if roi_len_m > 1.0:
            density = len(self.active_ids) / (roi_len_m / 1000.0)
            
        status = self.get_congestion_state(avg_speed_kmh, density)
        
        labels = {
            "Avg Speed (km/h)": f"{avg_speed_kmh:.1f}",
            "Density (veh/km)": f"{density:.1f}",
            "Status": status,
            "Count": len(self.active_ids)
        }
        
        # Draw custom dashboard instead of using annotator.display_analytics
        self.draw_info_panel(im0, labels)
        
        # Debug: Print metrics to console occasionally
        if self.frame_no % 30 == 0:
            print(f"Frame {self.frame_no}: {labels}")

        self.display_output(im0)
        
        return SolutionResults(
            plot_im=im0,
            in_count=0,
            out_count=0,
            classwise_count={},
            total_tracks=len(self.track_ids),
        )

    def draw_info_panel(self, im, labels):
        """Draw a semi-transparent info panel on the image."""
        font_scale = 0.7
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        gap = 30
        margin = 20
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        
        # Calculate panel size
        max_text_w = 0
        for k, v in labels.items():
            txt = f"{k}: {v}"
            (w, h), _ = cv2.getTextSize(txt, font, font_scale, thickness)
            max_text_w = max(max_text_w, w)
            
        panel_w = max_text_w + 2 * margin
        panel_h = len(labels) * gap + margin
        
        h, w = im.shape[:2]
        
        # Position: Top-Right
        x1 = w - panel_w - 20
        y1 = 20
        x2 = w - 20
        y2 = y1 + panel_h
        
        # Draw semi-transparent background
        overlay = im.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0, im)
        
        # Draw text
        y_text = y1 + margin + 10
        for k, v in labels.items():
            txt = f"{k}: {v}"
            cv2.putText(im, txt, (x1 + margin, y_text), font, font_scale, text_color, thickness)
            y_text += gap

    def get_congestion_state(self, avg_speed: float, density: float) -> str:
        # Simple heuristic
        if density > 80: # Very high density
            return "Congested" # 拥堵
        if avg_speed < 20 and density > 40:
            return "Slow" # 缓行
        return "Free Flow" # 通畅

def run(config: dict | None = None):
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
        
    src = cfg.get("source", ".\\test1.mp4")
    if isinstance(src, str):
        if src.strip() == "0":
            src = 0
        elif not os.path.exists(src):
            print(f"Error: Source file '{src}' not found.")
            # Don't return, let OpenCV try or fail, but print warning
    
    classes_cfg = cfg.get("classes", None)
    if classes_cfg is None:
        cls = None
    elif isinstance(classes_cfg, str):
        cls = [int(x) for x in classes_cfg.split(",") if x.strip()]
    else:
        cls = [int(x) for x in classes_cfg]
    mpp_cfg = cfg.get("mpp_values", [0.30, 0.18, 0.10])
    mpp_values = (
        [float(x) for x in mpp_cfg.split(",") if x.strip()]
        if isinstance(mpp_cfg, str)
        else [float(x) for x in mpp_cfg]
    )
    cap = cv2.VideoCapture(src)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = cap_fps if cap_fps and cap_fps > 1 else cfg.get("fps", 30.0)
    tm = TrafficSpeedDensity(
        model=cfg.get("model", None),
        classes=cls,
        region=cfg.get("region", None),
        mpp_values=mpp_values,
        speed_window_s=cfg.get("speed_window_s", 30),
        conf=cfg.get("conf", 0.3),
        iou=cfg.get("iou", 0.5),
        device=cfg.get("device", None),
        show=cfg.get("show", True),
        tracker=cfg.get("tracker", "bytetrack.yaml"),
        fps=fps,
        name_filter=cfg.get("name_filter", ["car", "person","bicycle","motorcycle","bus","truck"]),
    )
    if cfg.get("show"):
        cv2.namedWindow("Ultralytics Solutions", cv2.WINDOW_NORMAL)
        # Optional: Set a default size (e.g., 1280x720) if you want it to start smaller
        # cv2.resizeWindow("Ultralytics Solutions", 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tm.process(frame)
        
        # Check if window is closed (user pressed 'q' or clicked close button)
        if cfg.get("show"):
            # Note: WND_PROP_VISIBLE checks if window is visible. 
            # If user pressed 'q', BaseSolution destroys it, so it becomes invalid/invisible.
            try:
                if cv2.getWindowProperty("Ultralytics Solutions", cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                # In case window is already destroyed
                break
                
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
