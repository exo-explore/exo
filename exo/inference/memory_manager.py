"""
GPU Memory Manager for Exo

This module provides GPU memory prioritization and fallback mechanisms for model loading.
It ensures that models are loaded into GPU memory when possible, falling back to system memory
when GPU memory is insufficient.
"""

from typing import Optional, Tuple, Dict, Any
from exo.topology.device_capabilities import DeviceCapabilities
from exo.helpers import DEBUG
import platform
import os


class MemoryAllocationStrategy:
    """Defines memory allocation strategies for different scenarios"""
    
    def __init__(self, device_caps: DeviceCapabilities):
        self.device_caps = device_caps
        self.gpu_memory_mb = device_caps.available_gpu_memory
        self.system_memory_mb = device_caps.available_system_memory
        
    def get_allocation_strategy(self, model_size_mb: int) -> Dict[str, Any]:
        """
        Determine the best memory allocation strategy for a given model size.
        
        Args:
            model_size_mb: Size of the model in MB
            
        Returns:
            Dictionary containing allocation strategy parameters
        """
        strategy = {
            'use_gpu': False,
            'use_system_fallback': False,
            'gpu_layers': 0,
            'memory_mapping': True,
            'low_vram_mode': False,
            'batch_size': 512,
            'cache_type': 'auto'
        }
        
        # Calculate available memory with safety margin
        gpu_available = max(0, self.gpu_memory_mb - 1024)  # Reserve 1GB for OS
        system_available = max(0, self.system_memory_mb - 2048)  # Reserve 2GB for OS
        
        if DEBUG >= 1:
            print(f"Memory allocation for {model_size_mb}MB model:")
            print(f"  GPU available: {gpu_available}MB")
            print(f"  System available: {system_available}MB")
        
        # Strategy 1: Full GPU allocation (preferred)
        if gpu_available >= model_size_mb:
            strategy.update({
                'use_gpu': True,
                'gpu_layers': -1,  # All layers on GPU
                'low_vram_mode': False,
                'batch_size': min(1024, max(256, gpu_available // 16)),
                'cache_type': 'gpu'
            })
            if DEBUG >= 1:
                print("  Strategy: Full GPU allocation")
                
        # Strategy 2: Partial GPU + System memory
        elif gpu_available >= model_size_mb * 0.3:  # At least 30% fits in GPU
            strategy.update({
                'use_gpu': True,
                'use_system_fallback': True,
                'gpu_layers': int((gpu_available / model_size_mb) * 80),  # Conservative estimate
                'low_vram_mode': True,
                'batch_size': min(512, max(128, gpu_available // 32)),
                'cache_type': 'mixed'
            })
            if DEBUG >= 1:
                print(f"  Strategy: Partial GPU ({strategy['gpu_layers']} layers) + System fallback")
                
        # Strategy 3: System memory only
        elif system_available >= model_size_mb:
            strategy.update({
                'use_gpu': False,
                'use_system_fallback': True,
                'gpu_layers': 0,
                'memory_mapping': True,
                'batch_size': min(256, max(64, system_available // 64)),
                'cache_type': 'system'
            })
            if DEBUG >= 1:
                print("  Strategy: System memory only")
                
        # Strategy 4: Insufficient memory (error case)
        else:
            if DEBUG >= 1:
                print("  Strategy: Insufficient memory - model may not load properly")
                
        return strategy


class GPUMemoryManager:
    """Manages GPU memory allocation and fallback strategies"""
    
    def __init__(self):
        self.device_caps: Optional[DeviceCapabilities] = None
        self.allocation_strategy: Optional[MemoryAllocationStrategy] = None
        
    def initialize(self, device_caps: DeviceCapabilities):
        """Initialize with device capabilities"""
        self.device_caps = device_caps
        self.allocation_strategy = MemoryAllocationStrategy(device_caps)
        
        if DEBUG >= 1:
            print(f"GPU Memory Manager initialized:")
            print(f"  GPU Memory: {device_caps.available_gpu_memory}MB")
            print(f"  System Memory: {device_caps.available_system_memory}MB")
    
    def get_memory_config(self, model_size_mb: int) -> Dict[str, Any]:
        """Get memory configuration for a specific model size"""
        if not self.allocation_strategy:
            raise RuntimeError("MemoryManager not initialized")
            
        return self.allocation_strategy.get_allocation_strategy(model_size_mb)
    
    def estimate_model_size(self, model_path: str) -> int:
        """Estimate model size in MB from file path"""
        try:
            if os.path.isfile(model_path):
                size_bytes = os.path.getsize(model_path)
                return size_bytes // (1024 * 1024)
            elif os.path.isdir(model_path):
                total_size = 0
                for root, dirs, files in os.walk(model_path):
                    for file in files:
                        if file.endswith(('.safetensors', '.bin', '.gguf', '.pth')):
                            file_path = os.path.join(root, file)
                            total_size += os.path.getsize(file_path)
                return total_size // (1024 * 1024)
        except Exception as e:
            if DEBUG >= 1:
                print(f"Failed to estimate model size: {e}")
        
        # Conservative fallback estimate
        return 8192  # 8GB default estimate
    
    def optimize_for_gpu_inference(self) -> Dict[str, Any]:
        """Get optimal settings for GPU inference"""
        if not self.device_caps:
            return {}
            
        config = {}
        
        # Platform-specific optimizations
        if platform.system() == "Darwin":  # macOS
            config.update({
                'metal_gpu': True,
                'unified_memory': True,
                'gpu_memory_fraction': 0.9,  # Use most of unified memory
            })
        elif platform.system() == "Windows" or platform.system() == "Linux":
            if self.device_caps.available_gpu_memory > 0:
                config.update({
                    'cuda_gpu': True,
                    'gpu_memory_fraction': 0.8,  # Leave some VRAM for OS
                    'allow_growth': True,
                })
        
        return config
    
    def should_use_gpu_acceleration(self) -> bool:
        """Determine if GPU acceleration should be used"""
        if not self.device_caps:
            return False
            
        # Always prefer GPU if available
        return self.device_caps.available_gpu_memory > 1024  # At least 1GB VRAM


# Global memory manager instance
memory_manager = GPUMemoryManager()


def initialize_memory_manager(device_caps: DeviceCapabilities):
    """Initialize the global memory manager"""
    memory_manager.initialize(device_caps)


def get_memory_config_for_model(model_path: str) -> Dict[str, Any]:
    """Get memory configuration for a specific model"""
    model_size = memory_manager.estimate_model_size(model_path)
    return memory_manager.get_memory_config(model_size)


def should_prioritize_gpu() -> bool:
    """Check if GPU should be prioritized for inference"""
    return memory_manager.should_use_gpu_acceleration()


def get_gpu_optimization_config() -> Dict[str, Any]:
    """Get GPU optimization configuration"""
    return memory_manager.optimize_for_gpu_inference()
