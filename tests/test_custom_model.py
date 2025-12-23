from typing import Iterable
import argparse
import requests
import json
import sys
from typing import Any, Dict, List, cast

def main() -> None:
    parser = argparse.ArgumentParser(description="Test custom MLX model with EXO")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2.5-0.5B-Instruct-4bit", help="Model ID to test")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?", help="Prompt to send")
    parser.add_argument("--api-url", type=str, default="http://localhost:52415/v1/chat/completions", help="EXO API URL")
    
    args = parser.parse_args()
    
    # Cast args attributes to string to satisfy linter
    api_url: str = cast(str, args.api_url)
    model: str = cast(str, args.model)
    prompt: str = cast(str, args.prompt)

    base_url: str = api_url.rsplit("/v1/", 1)[0]
    place_url = f"{base_url}/place_instance"
    state_url = f"{base_url}/state"

    def check_for_instance(model_id: str) -> bool:
        try:
            state_resp = requests.get(state_url)
            state_resp.raise_for_status()
            # Explicitly cast requests.json() return to Dict[str, Any]
            state_data = cast(Dict[str, Any], state_resp.json())
            instances = cast(Dict[str, Dict[str,Dict[str,Any]]], state_data.get("instances", {}))
            
            for inst_wrapper in list(instances.values()):
                for actual_inst in list(inst_wrapper.values()):
                    shard_assignments = cast(Dict[str, Any], actual_inst.get("shardAssignments", {}))
                    if shard_assignments.get("modelId") == model_id:
                        return True
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to EXO. Make sure 'uv run exo' is running in the background.")
            sys.exit(1)
        except Exception as e:
            print(f"Warning: Error checking state: {e}")
        return False

    # 1. Check if instance already exists
    if check_for_instance(model):
        print(f"Instance for {model} already exists. Skipping placement.")
    else:
        # 2. Trigger Download/Placement
        print(f"Placing instance for {model}...")
        try:
            place_resp = requests.post(place_url, json={"model_id": model})
            place_resp.raise_for_status()
            print(f"Placement command sent: {place_resp.json()}")
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to EXO. Make sure 'uv run exo' is running.")
            sys.exit(1)
        except Exception as e:
            print(f"Error placing instance: {e}")
            return

    # 3. Wait for instance to be ready
    print("Waiting for instance to be ready (downloading)...")
    while True:
        if check_for_instance(model):
            print("\nInstance ready!")
            break
        
        print(".", end="", flush=True)
        import time
        time.sleep(2)

    # 4. Chat
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    print(f"Sending request to {api_url}...")
    print(f"Model: {model}")
    print("-" * 40)
    
    try:
        response = requests.post(api_url, json=payload, stream=True)
        response.raise_for_status() 
        # response.iter_lines() creates a Stream of Bytes,"
        # cast each line to bytes and decode it to utf-8
        response = cast(Iterable[bytes], response.iter_lines())

        print("Response:")
        for line in response:
            if line:
                decoded: str = line.decode('utf-8')
                if decoded.startswith("data: "):
                    content: str = decoded[6:]
                    if content == "[DONE]":
                        break
                    try:
                        chunk = cast(Dict[str, Any], json.loads(content))
                        choices = cast(List[Dict[str, Any]], chunk.get("choices", [{}]))
                        delta = cast(Dict[str, Any], choices[0].get("delta", {}))
                        delta_content: str = cast(str, delta.get("content", ""))
                        print(delta_content, end="", flush=True)
                    except json.JSONDecodeError:
                        pass
        print("\n" + "-" * 40)
        print("Success!")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to EXO. Make sure 'uv run exo' is running.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
