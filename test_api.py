import urllib.request
import json
import time

try:
    print("Testing Horizontal API...")
    with urllib.request.urlopen('http://127.0.0.1:5000/api/horizontal') as response:
        data = json.loads(response.read().decode('utf-8'))
        print(json.dumps(data, indent=2, ensure_ascii=False))

    print("\nTesting Vertical API...")
    with urllib.request.urlopen('http://127.0.0.1:5000/api/vertical') as response:
        data = json.loads(response.read().decode('utf-8'))
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
except Exception as e:
    print(f"Error: {e}")
