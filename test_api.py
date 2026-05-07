import requests 
import base64 
import cv2 
import numpy as np 
frames = [] 
for i in range(16): 
    frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) 
    _, buffer = cv2.imencode('.jpg', frame) 
    frames.append(base64.b64encode(buffer).decode('utf-8')) 
r = requests.post('http://localhost:8000/predict', json={'frames': frames}) 
print('Status:', r.status_code) 
if r.status_code == 200: 
    print('Result:', r.json()) 
else: 
    print('Error:', r.text) 
