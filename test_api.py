#!/usr/bin/env python3
import requests
import json

# Test through FastAPI
url = "http://localhost:8800/v1/chat/completions"
data = {
    "model": "llama-3.2-1b",
    "messages": [{"role": "user", "content": "Hi mom!"}],
    "max_tokens": 50,
    "temperature": 0.7
}

print("Testing FastAPI endpoint...")
try:
    response = requests.post(url, json=data, timeout=30)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print("Error:", response.text)
except Exception as e:
    print(f"Error: {e}")

# Also test direct connection
print("\n\nTesting direct connection to mini1...")
url = "http://192.168.2.13:8000/v1/chat/completions"
try:
    response = requests.post(url, json=data, timeout=30)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print("Error:", response.text)
except Exception as e:
    print(f"Error: {e}")