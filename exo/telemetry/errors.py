import requests
import traceback
def report_error(error: Exception) -> None:
    URL = "https://exo-logging-api-dev.foobar.dev/api/v1/logs/"
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
    r = requests.post(URL, headers=headers, json=data)
    if r.status_code != 200:
        print(f"Error reporting error: {r.status_code} {r.text}")
    else:
        print("Error reported successfully\n")

    