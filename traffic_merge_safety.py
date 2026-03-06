import time
import os
from collections import defaultdict, deque
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors

DEFAULT_CONFIG = {
    "model": "yolov8n.pt",
    "source": ".\\test1.mp4",
    "classes": None,
    "region": None,
    "mpp_values": [0.30, 0.18, 0.10],
    "speed_window_s": 5,
    "conf": 0.3,
    "iou": 0.5,
    "device": None,
    "show": True,
    "tracker": "bytetrack.yaml",
    "fps": 30.0,
    "name_filter": ["car", "truck", "bus", "motorcycle"],
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

    def y_to_meter(self, y: float) -> float:
        """Convert Y pixel coordinate to meters from top (y=0)."""
        return self.length_m(0, y)


class TrafficMergeSafety(BaseSolution):
    def __init__(self, **kwargs):
        self.mpp_values = kwargs.pop("mpp_values", [0.30, 0.18, 0.10])
        self.speed_window_s = kwargs.pop("speed_window_s", 5)
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
        
        # Smooth metrics
        self.avg_gap_history = deque(maxlen=30)  # Smooth gap over 1 second

    def initialize_region(self):
        super().initialize_region()

    def process(self, im0: np.ndarray) -> SolutionResults:
        h, w = im0.shape[:2]
        
        if not self.region_initialized:
            if self.region is None:
                # Default to full screen
                self.region = [(0, 0), (w, 0), (w, h), (0, h)]
            self.initialize_region()
            self.region_initialized = True

        if self.bands is None:
            self.bands = PixelMeterBands(h, self.mpp_values)

        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)
        
        # Draw region
        annotator.draw_region(reg_pts=self.region, color=(104, 0, 123), thickness=1)
        
        # Draw calibration lines
        for _, y2, mpp in self.bands.bounds[:-1]:
            y = int(y2)
            cv2.line(im0, (0, y), (w, y), (200, 200, 200), 1)
            
        active_vehicles = [] # List of (y_pos, speed_mps, track_id)

        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            class_name = self.names[cls]
            if self.name_filter is not None:
                if class_name.lower() not in self.name_filter:
                    continue

            annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(cls, True))
            self.store_tracking_history(track_id, box)
            
            c = self.track_history[track_id][-1]
            
            # Check region
            in_region = True
            if len(self.region) >= 3:
                if cv2.pointPolygonTest(np.array(self.region, np.int32), (int(c[0]), int(c[1])), False) < 0:
                    in_region = False
            
            if in_region:
                self.active_ids.add(track_id)
                
                # Calculate speed
                v_mps = 0.0
                if len(self.track_history[track_id]) > 1:
                    p0 = np.array(self.track_history[track_id][-2], dtype=np.float32)
                    p1 = np.array(self.track_history[track_id][-1], dtype=np.float32)
                    
                    pixel_dist = float(np.linalg.norm(p1 - p0))
                    
                    if pixel_dist >= 1.0:
                        avg_y = (p0[1] + p1[1]) * 0.5
                        mpp = float(self.bands.mpp_at(avg_y))
                        v_mps = pixel_dist * mpp * self.fps
                    
                    self.speed_hist[track_id].append(v_mps)
                
                # Get smoothed speed
                avg_v = 0.0
                if self.speed_hist[track_id]:
                    avg_v = sum(self.speed_hist[track_id]) / len(self.speed_hist[track_id])
                
                active_vehicles.append({
                    "id": track_id,
                    "y": c[1],
                    "speed": avg_v,
                    "box": box
                })

        # --- Analysis Logic ---
        
        # 1. Sort vehicles by Y position (assuming flow direction matters)
        # We assume standard view where Y increases downwards.
        # Sorting by Y helps find adjacent vehicles.
        active_vehicles.sort(key=lambda x: x["y"])
        
        # 2. Calculate Gaps (Time Headway)
        gaps = []
        for i in range(len(active_vehicles) - 1):
            # Vehicle i is "above" Vehicle i+1 (visually)
            # Distance between them
            v_rear = active_vehicles[i]
            v_front = active_vehicles[i+1]
            
            # Calculate physical distance between centers
            # Note: This is simplified. Real gap should be bumper-to-bumper.
            # But centers are easier to get.
            y_rear = v_rear["y"]
            y_front = v_front["y"]
            
            dist_m = self.bands.length_m(y_rear, y_front)
            
            # Use rear vehicle speed to calculate time gap (time to reach front vehicle position)
            # Avoid division by zero
            speed = max(v_rear["speed"], 1.0) # Min speed 1m/s to avoid infinite gap
            
            time_gap = dist_m / speed
            gaps.append(time_gap)

        # 3. Compute Metrics
        
        # Average Speed
        speeds = [v["speed"] for v in active_vehicles]
        avg_speed_kmh = (sum(speeds) / len(speeds) * 3.6) if speeds else 0.0
        
        # Safe Merge Gap (Average of current gaps, or a default large value if empty)
        if gaps:
            avg_gap = sum(gaps) / len(gaps)
        else:
            # If 0 or 1 car, gap is effectively infinite (very safe)
            avg_gap = 10.0 
            
        self.avg_gap_history.append(avg_gap)
        smooth_gap = sum(self.avg_gap_history) / len(self.avg_gap_history)
        
        # Density (for congestion check)
        y_vals = [p[1] for p in self.region]
        roi_len_m = self.bands.length_m(min(y_vals), max(y_vals)) if len(self.region) >= 3 else 0.0
        density = (len(active_vehicles) / (roi_len_m / 1000.0)) if roi_len_m > 1.0 else 0.0

        # 4. Determine States based on Rules
        
        # Congestion State
        congestion_status = "畅通" # Free Flow
        if density > 80:
            congestion_status = "拥堵" # Congested
        elif avg_speed_kmh < 20 and density > 40:
            congestion_status = "缓行" # Slow
        elif avg_speed_kmh < 40 and density > 20:
            congestion_status = "缓行"
            
        # Risk Level
        # Gap > 4s -> Low
        # Gap 2-4s -> Medium
        # Gap < 2s -> High
        if smooth_gap > 4.0:
            risk_level = "低" # Low
            advice = "可安全汇入" # Safe to Merge
            advice_color = (0, 255, 0) # Green
        elif smooth_gap > 2.0:
            risk_level = "中" # Medium
            advice = "谨慎汇入" # Caution
            advice_color = (0, 255, 255) # Yellow
        else:
            risk_level = "高" # High
            advice = "禁止汇入" # Do Not Merge
            advice_color = (0, 0, 255) # Red

        # Prepare Display Data
        labels = {
            "路况状态": congestion_status,
            "汇入间隙": f"{smooth_gap:.1f} 秒",
            "风险等级": risk_level,
            "系统建议": advice
        }
        
        # Draw Dashboard
        self.draw_dashboard(im0, labels, advice_color)
        
        # Debug Console
        if self.frame_no % 30 == 0:
            print(f"Frame {self.frame_no}: Speed={avg_speed_kmh:.1f}km/h, Gap={smooth_gap:.1f}s, Risk={risk_level}")

        self.display_output(im0)
        
        return SolutionResults(plot_im=im0)

    def draw_dashboard(self, im, labels, status_color):
        """Draw a professional dashboard overlay with Chinese support."""
        h, w = im.shape[:2]
        
        # Panel Config
        panel_w = 300
        panel_h = 180
        margin = 20
        x1 = w - panel_w - margin
        y1 = margin
        x2 = w - margin
        y2 = margin + panel_h
        
        # Background
        overlay = im.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, im, 0.3, 0, im)
        
        # Convert to PIL Image for Chinese text
        img_pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Try to load a Chinese font (SimHei is common on Windows)
        try:
            title_font = ImageFont.truetype("simhei.ttf", 24)
            text_font = ImageFont.truetype("simhei.ttf", 20)
            value_font = ImageFont.truetype("simhei.ttf", 22)
        except IOError:
            # Fallback if simhei not found (e.g. use default)
            print("Warning: SimHei font not found, Chinese may not display correctly.")
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default()
            value_font = ImageFont.load_default()
            
        # Title
        draw.text((x1 + 10, y1 + 15), "汇入安全分析", font=title_font, fill=(255, 255, 255))
        draw.line([(x1 + 10, y1 + 50), (x2 - 10, y1 + 50)], fill=(100, 100, 100), width=1)
        
        # Metrics
        y_cur = y1 + 60
        gap = 35
        
        # Helper to draw row
        def draw_row(label, value, value_color=(255, 255, 255)):
            draw.text((x1 + 10, y_cur), label + ":", font=text_font, fill=(200, 200, 200))
            draw.text((x1 + 110, y_cur), value, font=value_font, fill=value_color)
            
        # 1. Congestion
        draw_row("路况状态", labels["路况状态"])
        y_cur += gap
        
        # 2. Gap
        draw_row("汇入间隙", labels["汇入间隙"])
        y_cur += gap
        
        # 3. Risk & Advice - use dynamic color
        # PIL uses RGB, status_color is BGR (from OpenCV logic above)
        # Convert BGR tuple to RGB for PIL
        status_rgb = (status_color[2], status_color[1], status_color[0])
        
        draw.text((x1 + 10, y_cur), labels["系统建议"], font=title_font, fill=status_rgb)
        
        # Convert back to OpenCV
        im[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


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
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = cfg.get("fps", 30.0)
        
    safety_monitor = TrafficMergeSafety(
        model=cfg.get("model", None),
        classes=cls,
        region=cfg.get("region", None),
        mpp_values=mpp_values,
        speed_window_s=cfg.get("speed_window_s", 5),
        conf=cfg.get("conf", 0.3),
        iou=cfg.get("iou", 0.5),
        device=cfg.get("device", None),
        show=cfg.get("show", True),
        tracker=cfg.get("tracker", "bytetrack.yaml"),
        fps=fps,
        name_filter=cfg.get("name_filter", ["car", "truck", "bus", "motorcycle"]),
    )
    
    if cfg.get("show"):
        cv2.namedWindow("Ultralytics Solutions", cv2.WINDOW_NORMAL)
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        safety_monitor.process(frame)
        
        if cfg.get("show"):
            try:
                if cv2.getWindowProperty("Ultralytics Solutions", cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break
                
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
