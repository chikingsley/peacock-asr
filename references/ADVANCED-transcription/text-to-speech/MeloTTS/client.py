import requests
from base64 import b64encode, b64decode
import json
import numpy as np
from scipy.io.wavfile import write
import time

url = "http://127.0.0.1:8000/connection"

payload = json.dumps({
  "voice_id": "EN-US",
  "text": "This  ",
  "sr": 8000
})

headers = {
  'Content-Type': 'application/json'
}

start_time = time.time()  # Record the start time

response = requests.request("POST", url, headers=headers, data=payload)
if response.ok:
    response_dict = json.loads(response.text)
    audio = b64decode(response_dict["audio"])
    print(f"Server processing time: {response_dict['time']} seconds")
    with open("text3.wav", 'wb') as file:
        file.write(audio)

    end_time = time.time()  # Record the end time
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
else:
    print(f"Request failed with status code: {response.status_code}")
    print(f"Error message: {response.text}")