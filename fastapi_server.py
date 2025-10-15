#!/usr/bin/env python3
"""
FastAPI server for Exo cluster
Provides a unified API interface for the distributed inference cluster
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Union
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import httpx


# Request/Response Models
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "llama-3.2-3b"
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1000, ge=1)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class CompletionRequest(BaseModel):
    model: str = "llama-3.2-3b"
    prompt: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1000, ge=1)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    echo: bool = False
    n: int = Field(default=1, ge=1, le=10)


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "exo-cluster"


# Configuration
EXO_NODES = [
    {"name": "mini1", "url": "http://192.168.2.13:8000"},
    {"name": "mini2", "url": "http://192.168.5.2:8003"},
]

# Available models in the cluster
AVAILABLE_MODELS = [
    "llama-3.2-1b",
    "llama-3.2-3b",
    "llama-3.1-8b",
    "llama-3.1-70b",
    "llama-3.1-405b",
    "mistral-nemo-2407",
    "mistral-large-2407",
    "deepseek-coder-v2-lite",
    "deepseek-coder-v2",
    "qwen-2.5-72b",
    "qwen-2.5-coder-32b",
]


# Global HTTP client
http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global http_client
    # Startup
    # Increase timeout for model inference
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
    yield
    # Shutdown
    if http_client:
        await http_client.aclose()


# Create FastAPI app
app = FastAPI(
    title="Exo Cluster API",
    description="Unified API for distributed inference with Exo",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper functions
async def check_node_health(node_url: str) -> bool:
    """Check if an exo node is healthy by testing the models endpoint"""
    try:
        # Exo doesn't have a /health endpoint, so we check /v1/models instead
        response = await http_client.get(f"{node_url}/v1/models", timeout=2.0)
        return response.status_code == 200
    except:
        return False


async def get_available_node() -> Optional[Dict]:
    """Get the first available node"""
    for node in EXO_NODES:
        if await check_node_health(node["url"]):
            return node
    return None


async def forward_to_node_stream(node_url: str, endpoint: str, data: dict):
    """Forward streaming request to an exo node"""
    url = f"{node_url}{endpoint}"
    
    async with http_client.stream("POST", url, json=data) as response:
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Node request failed")
        
        async for line in response.aiter_lines():
            if line:
                yield f"{line}\n"


async def forward_to_node(node_url: str, endpoint: str, data: dict):
    """Forward request to an exo node"""
    url = f"{node_url}{endpoint}"
    
    response = await http_client.post(url, json=data)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Node request failed")
    return response.json()


# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Exo Cluster API",
        "version": "1.0.0",
        "nodes": EXO_NODES,
        "endpoints": {
            "chat": "/v1/chat/completions",
            "completions": "/v1/completions",
            "models": "/v1/models",
            "health": "/health",
            "cluster_status": "/cluster/status"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/cluster/status")
async def cluster_status():
    """Get cluster status"""
    status = {
        "nodes": [],
        "available_models": AVAILABLE_MODELS,
        "timestamp": time.time()
    }
    
    for node in EXO_NODES:
        node_status = {
            "name": node["name"],
            "url": node["url"],
            "healthy": await check_node_health(node["url"])
        }
        status["nodes"].append(node_status)
    
    status["cluster_healthy"] = any(n["healthy"] for n in status["nodes"])
    return status


@app.get("/v1/models")
async def list_models():
    """List available models"""
    models = []
    for model_id in AVAILABLE_MODELS:
        models.append(ModelInfo(
            id=model_id,
            created=int(time.time())
        ))
    
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Chat completions endpoint (OpenAI compatible)"""
    node = await get_available_node()
    if not node:
        raise HTTPException(status_code=503, detail="No available nodes in cluster")
    
    # Build clean request for exo
    request_data = {
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }
    
    # Only add optional fields if they're set and valid
    if request.top_p is not None and request.top_p != 1.0:
        request_data["top_p"] = request.top_p
    
    if request.stop is not None:
        # Handle stop sequences properly
        if isinstance(request.stop, str) and request.stop != "string":
            request_data["stop"] = request.stop
        elif isinstance(request.stop, list):
            request_data["stop"] = request.stop
    
    if request.stream:
        request_data["stream"] = True
        return StreamingResponse(
            forward_to_node_stream(node["url"], "/v1/chat/completions", request_data),
            media_type="text/event-stream"
        )
    else:
        response = await forward_to_node(node["url"], "/v1/chat/completions", request_data)
        return response


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Text completions endpoint (OpenAI compatible)"""
    node = await get_available_node()
    if not node:
        raise HTTPException(status_code=503, detail="No available nodes in cluster")
    
    request_data = request.dict(exclude_unset=True)
    
    if request.stream:
        return StreamingResponse(
            forward_to_node_stream(node["url"], "/v1/completions", request_data),
            media_type="text/event-stream"
        )
    else:
        response = await forward_to_node(node["url"], "/v1/completions", request_data)
        return response


@app.post("/chat/completions")
async def chat_completions_alt(request: ChatCompletionRequest):
    """Alternative chat completions endpoint"""
    return await chat_completions(request)


# Custom inference endpoint
@app.post("/inference")
async def inference(
    prompt: str,
    model: str = "llama-3.2-3b",
    max_tokens: int = 100,
    temperature: float = 0.7,
    node_preference: Optional[str] = None
):
    """Simple inference endpoint with node selection"""
    # Select node based on preference
    target_node = None
    if node_preference:
        for node in EXO_NODES:
            if node["name"] == node_preference:
                target_node = node
                break
    
    if not target_node:
        target_node = await get_available_node()
    
    if not target_node:
        raise HTTPException(status_code=503, detail="No available nodes")
    
    # Create request
    request_data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    
    response = await forward_to_node(target_node["url"], "/v1/completions", request_data)
    return {
        "node_used": target_node["name"],
        "response": response
    }


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Get cluster metrics"""
    metrics = {
        "timestamp": time.time(),
        "nodes": {}
    }
    
    for node in EXO_NODES:
        try:
            # You could add actual metrics collection here
            metrics["nodes"][node["name"]] = {
                "healthy": await check_node_health(node["url"]),
                "url": node["url"]
            }
        except:
            metrics["nodes"][node["name"]] = {
                "healthy": False,
                "url": node["url"]
            }
    
    return metrics


def main():
    """Run the FastAPI server"""
    print("Starting Exo Cluster FastAPI Server...")
    print(f"Configured nodes: {[n['name'] for n in EXO_NODES]}")
    print("Server will be available at: http://localhost:8800")
    print("API docs available at: http://localhost:8800/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8800,
        log_level="info"
    )


if __name__ == "__main__":
    main()