import pytest
import asyncio
from aiohttp import web
import sys
import os
from pathlib import Path
import json
from unittest import mock
from aiohttp.test_utils import make_mocked_request
from typing import Union
from pathlib import Path as PathLike

# Add the parent directory to Python path to import exo
sys.path.append(str(Path(__file__).parent.parent))

from exo.api.chatgpt_api import ChatGPTAPI
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader
from exo import DEBUG
from exo.models import model_base_shards

# Mock callback class
class MockCallback:
    def __init__(self):
        pass

    def register(self, callback_id):
        return self

    def deregister(self, callback_id):
        return True

    async def wait(self, callback, timeout=None):
        await asyncio.sleep(0.1)  # Short sleep to simulate waiting
        return None, [1, 2, 3], True  # Mock token output

# Mock the tokenizer
class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 0
        self.special_tokens_map = {"eos_token_id": 0}
        self._tokenizer = None

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, tokens):
        return "test response"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "test prompt"

async def mock_resolve_tokenizer(model_id: Union[str, PathLike]) -> MockTokenizer:
    return MockTokenizer()

# Mock the model_base_shards
test_shard = Shard(model_id="test_model", start_layer=0, end_layer=1, n_layers=1)
model_base_shards["dummy"] = {"dummy": test_shard}

class DummyNode:
    def __init__(self, has_downloads=False):
        self.has_downloads = has_downloads
        self.prompt_calls = []
        self.cancelled = False
        self._on_token = MockCallback()
        
    def has_active_downloads(self):
        return self.has_downloads
        
    async def process_prompt(self, shard, prompt, image_str=None, request_id=None):
        self.prompt_calls.append((shard, prompt, request_id))
        # Simulate long-running process
        try:
            await asyncio.sleep(2)
            return []
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    @property
    def on_token(self):
        return self._on_token

async def create_mock_request(body_dict):
    """Helper function to create a properly mocked request"""
    if DEBUG >= 2:
        print(f"Creating mock request with body: {body_dict}")
    
    protocol = mock.Mock()
    transport = mock.Mock()
    protocol.transport = transport
    
    request = make_mocked_request(
        'POST', 
        '/chat/completions',
        headers={'Content-Type': 'application/json'},
        protocol=protocol
    )
    
    request._transport = mock.Mock()
    request._transport.get_extra_info = mock.Mock(return_value=("127.0.0.1", 8000))
    
    async def mock_json():
        if DEBUG >= 2:
            print(f"Mock json() called, returning: {body_dict}")
        return body_dict
    request.json = mock_json
    
    return request

@pytest.fixture(autouse=True)
def mock_tokenizer(monkeypatch):
    """Patch both the tokenizer module and the function"""
    # Mock the entire module
    mock_tokenizers_module = mock.MagicMock()
    mock_tokenizers_module.resolve_tokenizer = mock_resolve_tokenizer
    mock_tokenizers_module._resolve_tokenizer = mock_resolve_tokenizer
    
    # Patch both the module and the specific functions
    monkeypatch.setattr("exo.inference.tokenizers", mock_tokenizers_module)
    monkeypatch.setattr("exo.api.chatgpt_api.resolve_tokenizer", mock_resolve_tokenizer)
    monkeypatch.setattr("exo.inference.tokenizers.resolve_tokenizer", mock_resolve_tokenizer)
    monkeypatch.setattr("exo.inference.tokenizers._resolve_tokenizer", mock_resolve_tokenizer)

@pytest.mark.asyncio
async def test_chat_completion_no_downloads():
    """Test that requests can be cancelled when no downloads are active"""
    if DEBUG >= 2:
        print("\nStarting test_chat_completion_no_downloads")
    
    node = DummyNode(has_downloads=False)
    api = ChatGPTAPI(node, "dummy", response_timeout=1)
    
    request = await create_mock_request({
        "model": "dummy",
        "messages": [{"role": "user", "content": "test"}],
        "temperature": 0.7,
        "stream": False
    })
    
    response = await api.handle_post_chat_completions(request)
    
    if DEBUG >= 2:
        print(f"Response status: {response.status}")
        print(f"Node cancelled: {node.cancelled}")
        print(f"Prompt calls: {node.prompt_calls}")
    
    assert response.status == 408  # Timeout status
    assert node.cancelled == True  # Task was cancelled
    assert len(node.prompt_calls) == 1  # Process was started

@pytest.mark.asyncio
async def test_chat_completion_with_downloads():
    """Test that requests cannot be cancelled when downloads are active"""
    if DEBUG >= 2:
        print("\nStarting test_chat_completion_with_downloads")
    
    node = DummyNode(has_downloads=True)
    api = ChatGPTAPI(node, "dummy", response_timeout=1)
    
    request = await create_mock_request({
        "model": "dummy",
        "messages": [{"role": "user", "content": "test"}],
        "temperature": 0.7,
        "stream": False
    })
    
    response = await api.handle_post_chat_completions(request)
    
    if DEBUG >= 2:
        print(f"Response status: {response.status}")
        print(f"Node cancelled: {node.cancelled}")
        print(f"Prompt calls: {node.prompt_calls}")
    
    assert response.status == 408  # Timeout status
    assert node.cancelled == False  # Task was not cancelled due to downloads
    assert len(node.prompt_calls) == 1  # Process was started

@pytest.mark.asyncio
async def test_chat_completion_normal_operation():
    """Test normal operation without timeouts"""
    if DEBUG >= 2:
        print("\nStarting test_chat_completion_normal_operation")
    
    node = DummyNode(has_downloads=False)
    api = ChatGPTAPI(node, "dummy", response_timeout=5)  # Longer timeout
    
    request = await create_mock_request({
        "model": "dummy",
        "messages": [{"role": "user", "content": "test"}],
        "temperature": 0.7,
        "stream": False
    })
    
    response = await asyncio.wait_for(
        api.handle_post_chat_completions(request),
        timeout=3
    )
    
    if DEBUG >= 2:
        print(f"Response status: {response.status}")
        print(f"Node cancelled: {node.cancelled}")
        print(f"Prompt calls: {node.prompt_calls}")
    
    assert response.status == 200  # Success status
    assert node.cancelled == False  # Task should not be cancelled
    assert len(node.prompt_calls) == 1  # Process was started

if __name__ == "__main__":
    pytest.main([__file__, "-v"])