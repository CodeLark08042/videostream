import time
import random
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Simulation State
class TrafficSimulation:
    def __init__(self):
        self.start_time = time.time()
        # Cycle duration in seconds
        self.cycle_duration = 60 
    
    def get_status(self):
        """
        Simulate a traffic cycle:
        0-20s: Free Flow (High Speed, Low Density, Safe)
        20-40s: Slow (Medium Speed, Medium Density, Caution)
        40-60s: Congested (Low Speed, High Density, Danger)
        """
        elapsed = int(time.time() - self.start_time) % self.cycle_duration
        
        if elapsed < 20:
            return "free_flow"
        elif elapsed < 40:
            return "slow"
        else:
            return "congested"

sim = TrafficSimulation()

@app.route('/api/horizontal', methods=['GET'])
def get_horizontal_data():
    state = sim.get_status()
    
    if state == "free_flow":
        avg_speed = random.uniform(60, 80)
        density = random.uniform(10, 30)
        status = "畅通"
        warning = "" # No warning
    elif state == "slow":
        avg_speed = random.uniform(30, 50)
        density = random.uniform(40, 60)
        status = "缓行"
        warning = "前方T型路口，支路盲区可能有车辆汇入，请注意减速观察"
    else: # congested
        avg_speed = random.uniform(5, 20)
        density = random.uniform(80, 100)
        status = "拥堵"
        warning = "路段拥堵，汇入车辆风险提升，谨慎通行"
        
    return jsonify({
        "avg_speed": round(avg_speed, 1),
        "density": round(density, 1),
        "status": status,
        "warning": warning
    })

@app.route('/api/vertical', methods=['GET'])
def get_vertical_data():
    state = sim.get_status()
    
    # Vertical data depends on horizontal state
    if state == "free_flow":
        main_status = "畅通"
        merge_gap = random.uniform(6.0, 10.0)
        risk_level = "低"
        advice = "可安全汇入"
        can_merge = True
    elif state == "slow":
        main_status = "缓行"
        merge_gap = random.uniform(2.5, 4.5)
        risk_level = "中"
        advice = "谨慎汇入"
        can_merge = True # Conditional
    else: # congested
        main_status = "拥堵"
        merge_gap = random.uniform(0.5, 1.8)
        risk_level = "高"
        advice = "禁止汇入"
        can_merge = False
        
    return jsonify({
        "main_road_status": main_status,
        "merge_gap": round(merge_gap, 1),
        "risk_level": risk_level,
        "advice": advice,
        "can_merge": can_merge
    })

if __name__ == '__main__':
    # Listen on all interfaces
    print("Starting Traffic Simulation Server on port 5000...")
    app.run(host='0.0.0.0', port=5000)
