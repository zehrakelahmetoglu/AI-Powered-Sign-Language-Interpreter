import requests
import json

url = "http://127.0.0.1:8000/api/tahmin_yap"
data = {
    "oturum_id": "test_oturum_123"
}

print(f"Sending request to {url}...")
try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
except Exception as e:
    print(f"Error: {e}")
