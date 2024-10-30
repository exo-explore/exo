import requests
import traceback

class ErrorReporter:    
    def __init__(self, logging_url: str):
        self.logging_url = logging_url
    
    def report_error(self, error: Exception) -> None:
        headers = { 
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        data = {
            "commit_id": "string",
            "action": "string",
            "device_id": "string",
            "session_id": "string",
            "topology": {},
            "data": {"error": str(error), "stacktrace": traceback.format_exc()}
        }
        r = requests.post(self.logging_url, headers=headers, json=data)
        if r.status_code != 200:
            print(f"Error reporting error: {r.status_code} {r.text}")
        else:
            print("Error reported successfully\n")

    