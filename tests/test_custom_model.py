"""
Test script to verify custom model is working with EXO
"""
import argparse
import requests
import json
import sys
import os
import time
import pytest
from typing import Any, Dict, List, cast, Iterable

def is_exo_running(api_url: str) -> bool:
    base_url = api_url.rsplit("/v1/", 1)[0]
    state_url = f"{base_url}/state"
    try:
        resp = requests.get(state_url, timeout=1)
        resp.raise_for_status()
        return True
    except:
        return False

@pytest.mark.skipif(
    not is_exo_running("http://localhost:52415/v1/chat/completions"), 
    reason="Exo instance not running at http://localhost:52415"
)
def test_custom_model_chat():
    """Integration test: Placing a custom model and chatting with it."""
    # When running via pytest, use defaults or env vars
    model = os.environ.get("MODEL_ID", "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    prompt = "What is the capital of France?"
    api_url = os.environ.get("EXO_API_URL", "http://localhost:52415/v1/chat/completions")
    
    run_custom_model_test(model, prompt, api_url)

def run_custom_model_test(model: str, prompt: str, api_url: str):
    base_url: str = api_url.rsplit("/v1/", 1)[0]
    place_url = f"{base_url}/place_instance"
    state_url = f"{base_url}/state"

    def check_for_instance(model_id: str) -> bool:
        try:
            state_resp = requests.get(state_url, timeout=5)
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
            if "pytest" not in sys.modules:
                sys.exit(1)
            else:
                pytest.fail("Could not connect to EXO")
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
            place_resp = requests.post(place_url, json={"model_id": model}, timeout=10)
            place_resp.raise_for_status()
            print(f"Placement command sent: {place_resp.json()}")
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to EXO. Make sure 'uv run exo' is running.")
            if "pytest" not in sys.modules:
                sys.exit(1)
            else:
                pytest.fail("Could not connect to EXO")
        except Exception as e:
            print(f"Error placing instance: {e}")
            if "pytest" in sys.modules:
                pytest.fail(f"Error placing instance: {e}")
            return

    # 3. Wait for instance to be ready
    print("Waiting for instance to be ready (downloading)...")
    wait_count = 0
    max_wait = 300 # scripts had infinite loop. keeping it somewhat bound.
    while True:
        if check_for_instance(model):
            print("\nInstance ready!")
            break
        
        print(".", end="", flush=True)
        time.sleep(2)
        wait_count += 2
        
        # If running in test mode, don't wait forever
        if "pytest" in sys.modules and wait_count > max_wait:
             pytest.fail("Timeout waiting for instance to be ready")

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
        response = requests.post(api_url, json=payload, stream=True, timeout=60)
        response.raise_for_status() 
        # response.iter_lines() creates a Stream of Bytes,"
        # cast each line to bytes and decode it to utf-8
        response_iter = cast(Iterable[bytes], response.iter_lines())

        print("Response:")
        full_response = ""
        for line in response_iter:
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
                        full_response += delta_content
                    except json.JSONDecodeError:
                        pass
        print("\n" + "-" * 40)
        print("Success!")
        
        # Basic assertion that we got something back
        if not full_response and "pytest" in sys.modules:
             pytest.fail("No response received from model")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to EXO. Make sure 'uv run exo' is running.")
        if "pytest" not in sys.modules:
            sys.exit(1)
        else:
             pytest.fail("Could not connect to EXO")
    except Exception as e:
        print(f"Error: {e}")
        if "pytest" not in sys.modules:
            sys.exit(1)
        else:
             pytest.fail(f"Error chatting: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test custom MLX model with EXO")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2.5-0.5B-Instruct-4bit", help="Model ID to test")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?", help="Prompt to send")
    parser.add_argument("--api-url", type=str, default="http://localhost:52415/v1/chat/completions", help="EXO API URL")
    
    args = parser.parse_args()
    
    # Cast args attributes to string to satisfy linter
    api_url_arg: str = cast(str, args.api_url)
    model_arg: str = cast(str, args.model)
    prompt_arg: str = cast(str, args.prompt)
    
    run_custom_model_test(model_arg, prompt_arg, api_url_arg)
