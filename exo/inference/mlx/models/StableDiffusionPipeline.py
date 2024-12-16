# Adapted from https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/__init__.py

import time
from typing import Optional, Tuple
import inspect

import mlx.core as mx
import mlx.nn as nn
from pathlib import Path

from tqdm import tqdm

from .sd_models.vae import ModelArgs as VAEArgs
from .sd_models.vae import Autoencoder
from .sd_models.tokenizer import load_tokenizer
from .sd_models.clip import CLIPTextModel
from .sd_models.clip import ModelArgs as CLIPArgs
from .sd_models.unet import UNetConfig, UNetModel

from dataclasses import dataclass, field
from exo.inference.shard import Shard

@dataclass
class DiffusionConfig:
    beta_schedule: str = "scaled_linear"
    beta_start: float = 0.00085
    beta_end: float = 0.012
    num_train_steps: int = 1000

    @classmethod
    def from_dict(cls, params):
        return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})


#Sampler
def _linspace(a, b, num):
    x = mx.arange(0, num) / (num - 1)
    return (b - a) * x + a


def _interp(y, x_new):
    """Interpolate the function defined by (arange(0, len(y)), y) at positions x_new."""
    x_low = x_new.astype(mx.int32)
    x_high = mx.minimum(x_low + 1, len(y) - 1)

    y_low = y[x_low]
    y_high = y[x_high]
    delta_x = x_new - x_low
    y_new = y_low * (1 - delta_x) + delta_x * y_high

    return y_new

class SimpleEulerSampler:
    """A simple Euler integrator that can be used to sample from our diffusion models.

    The method ``step()`` performs one Euler step from x_t to x_t_prev.
    """

    def __init__(self, config: DiffusionConfig):
        # Compute the noise schedule
        if config.beta_schedule == "linear":
            betas = _linspace(
                config.beta_start, config.beta_end, config.num_train_steps
            )
        elif config.beta_schedule == "scaled_linear":
            betas = _linspace(
                config.beta_start**0.5, config.beta_end**0.5, config.num_train_steps
            ).square()
        else:
            raise NotImplementedError(f"{config.beta_schedule} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = mx.cumprod(alphas)

        self._sigmas = mx.concatenate(
            [mx.zeros(1), ((1 - alphas_cumprod) / alphas_cumprod).sqrt()]
        )

    @property
    def max_time(self):
        return len(self._sigmas) - 1

    def sample_prior(self, shape, dtype=mx.float32, key=None):
        noise = mx.random.normal(shape, key=key)
        return (
            noise * self._sigmas[-1] * (self._sigmas[-1].square() + 1).rsqrt()
        ).astype(dtype)

    def add_noise(self, x, t, key=None):
        noise = mx.random.normal(x.shape, key=key)
        s = self.sigmas(t)
        return (x + noise * s) * (s.square() + 1).rsqrt()

    def sigmas(self, t):
        return _interp(self._sigmas, t)

    def timesteps(self, num_steps: int, start_time=None, dtype=mx.float32):
        start_time = start_time or (len(self._sigmas) - 1)
        assert 0 < start_time <= (len(self._sigmas) - 1)
        steps = _linspace(start_time, 0, num_steps + 1).astype(dtype)
        return list(zip(steps, steps[1:]))

    def current_timestep(self, step, total_steps, start_time=None):
        if step < total_steps:
            steps = self.timesteps(total_steps, start_time)
            return steps[step]
        else:
            return mx.array(0),mx.array(0)

    def step(self, eps_pred, x_t, t, t_prev):
        sigma = self.sigmas(t).astype(eps_pred.dtype)
        sigma_prev = self.sigmas(t_prev).astype(eps_pred.dtype)

        dt = sigma_prev - sigma
        x_t_prev = (sigma.square() + 1).sqrt() * x_t + eps_pred * dt

        x_t_prev = x_t_prev * (sigma_prev.square() + 1).rsqrt()

        return x_t_prev

@dataclass
class ShardConfig:
    model_id:str
    start_layer:int
    end_layer:int
    n_layers:int

@dataclass
class StableDiffusionConfig:
    model_type:str
    vae:VAEArgs
    text_encoder:CLIPArgs
    scheduler:DiffusionConfig
    unet:UNetConfig
    shard:ShardConfig
    
    @classmethod
    def from_dict(cls, params):
        return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})

@dataclass
class ModelArgs(StableDiffusionConfig):
    shard:Shard = field(default_factory=lambda: Shard("", 0, 0, 0))

    def __post_init__(self):
        if isinstance(self.shard, dict):
            self.shard = Shard(**self.shard)

        if not isinstance(self.shard, Shard):
            raise TypeError(f"Expected shard to be a Shard instance or a dict, got {type(self.shard)} instead")


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model_type = config.model_type
        self.config = config
        self.model_path = config.vae['path'].split('/vae')[0]
        self.shard = config.shard
        self.shard_clip, self.shard_encoder, self.shard_unet, self.shard_decoder  = model_shards(config.shard)
        self.config_clip=CLIPArgs.from_dict(config.text_encoder['config'])
        if self.shard_clip.start_layer != -1:
            self.text_encoder = CLIPTextModel(self.config_clip, shard=self.shard_clip)
        else:
            self.text_encoder = nn.Identity()    
        self.tokenizer = load_tokenizer(Path(self.model_path), "vocab.json", "merges.txt")
        self.diffusion_config = DiffusionConfig.from_dict(config.scheduler['config'])
        self.sampler = SimpleEulerSampler(self.diffusion_config)
        if self.shard_unet.start_layer!=-1:
            self.config_unet = UNetConfig.from_dict(config.unet['config'])
            self.unet = UNetModel(self.config_unet, self.shard_unet)
        else:
            self.unet = nn.Identity()
        self.config_vae=VAEArgs.from_dict(config.vae['config'])
        if self.shard_encoder.start_layer != -1:
            self.encoder=Autoencoder(self.config_vae, self.shard_encoder, "vae_encoder") 
        else:
            self.encoder = nn.Identity()            
        if self.shard_decoder.start_layer != -1:
            self.decoder=Autoencoder(self.config_vae, self.shard_decoder, "vae_decoder") 
        else:
            self.decoder = nn.Identity()            

    def __call__(self,x, step= 0, cfg_weight: float = 7.5,total_steps=50,conditioning=None,mask=None,residual=None,x_t_prev=None,is_finished=False,is_step_finished=False, image=None, strength=0.65, start_step=None):
        t, t_prev = self.sampler.current_timestep(step=step, total_steps=total_steps, start_time=start_step)
        is_finished = False
        is_step_finished = False
        if t.item()==1000:
            if self.shard_clip.start_layer == 0:
                conditioning = x
            if self.shard_clip.start_layer != -1:
                conditioning, mask= self.text_encoder(conditioning,mask)
            seed = int(time.time()) 
            mx.random.seed(seed)
            if image is None:
                if self.shard_encoder.is_last_layer():
                    x = self.sampler.sample_prior((1, *(64, 64), self.config_vae.latent_channels_in), dtype=mx.float32)
                    x_t_prev=x
                    start_step = self.sampler.max_time
            else:
                if self.shard_encoder.start_layer != -1:
                    image= self.encoder.encode(image)
                    if self.shard_encoder.is_last_layer():
                        start_step = self.sampler.max_time*strength
                        total_steps = int(total_steps*strength)
                        image = mx.broadcast_to(image, (1,) + image.shape[1:])
                        x_t_prev=self.sampler.add_noise(image, mx.array(start_step))
                        image = None
                        t, t_prev = self.sampler.current_timestep(step=step, total_steps=total_steps, start_time=start_step)
        # Perform the denoising loop
        if self.shard_unet.start_layer != -1:
            with tqdm(total=total_steps,initial=step+1) as pbar:
                if step<total_steps:
                    x = x_t_prev
                    if self.shard_unet.is_first_layer():
                        x_t_unet = mx.concatenate([x] * 2, axis=0) if cfg_weight> 1 else x
                    else:
                        x_t_unet = x
                    t_unet = mx.broadcast_to(t, [len(x_t_unet)])
                    x, residual= self.unet(x_t_unet, t_unet, encoder_x=conditioning, residuals=residual)
                    if self.shard_unet.is_last_layer():
                        if cfg_weight > 1:
                            eps_text, eps_neg = x.split(2)
                            eps_pred = eps_neg + cfg_weight * (eps_text - eps_neg)
                        x = self.sampler.step(eps_pred, x_t_prev, t, t_prev)
                        x_t_prev=x
                    mx.eval(x)
                    
        if self.shard_decoder.is_last_layer():
            is_step_finished=True
            if self.shard_decoder.start_layer != -1:
                x=self.decoder.decode(x)
            if self.shard_decoder.is_last_layer():
                x = mx.clip(x / 2 + 0.5, 0, 1)
                B, H, W, C = x.shape
                x = x.reshape(1, B // 1, H, W, C).transpose(0, 2, 1, 3, 4)
                x = x.reshape(1 * H, B // 1 * W, C)
                x = (x * 255).astype(mx.uint8)
                if t_prev.item() ==0:
                    is_finished=True   
        mx.eval(x)
         
        return x, {'conditioning':conditioning, 'mask':mask,'residual':residual,'x_t_prev':x_t_prev,'is_finished':is_finished,'is_step_finished':is_step_finished, 'step':step, 'total_steps':total_steps, 'start_step':start_step, 'image':image}
    

    def load(self):
        if self.shard_encoder.start_layer != -1:    
            vae_weights =  mx.load(self.config_vae.weight_files[0])
            vae_weights = self.encoder.sanitize(vae_weights)
            self.encoder.load_weights(list(vae_weights.items()), strict=True)
        if self.shard_decoder.start_layer != -1:
            vae_weights =  mx.load(self.config_vae.weight_files[0])
            vae_weights = self.decoder.sanitize(vae_weights)
            self.decoder.load_weights(list(vae_weights.items()), strict=True)
        if self.shard_clip.start_layer != -1:
            clip_weights = mx.load(self.config_clip.weight_files[0])
            clip_weights = self.text_encoder.sanitize(clip_weights)
            self.text_encoder.load_weights(list(clip_weights.items()), strict=True)
        if self.shard_unet.start_layer !=-1:
            unet_weights = mx.load(self.config_unet.weight_files[0])
            unet_weights = self.unet.sanitize(unet_weights)
            self.unet.load_weights(list(unet_weights.items()), strict=True)

def model_shards(shard:ShardConfig):
    def create_shard(shard, model_ranges):
        start_layer = shard.start_layer
        end_layer = shard.end_layer
        
        shards = {}
        
        for model_name, (range_start, range_end) in model_ranges.items():
            if start_layer < range_end and end_layer >= range_start:
                # Calculate the overlap with the model range
                overlap_start = max(start_layer, range_start)
                overlap_end = min(end_layer, range_end - 1)

                # Adjust the layers relative to the model's range
                relative_start = overlap_start - range_start
                relative_end = overlap_end - range_start
                shards[model_name] = Shard(model_name, relative_start, relative_end, range_end - range_start)
            else:
                # If no overlap, create a zero-layer shard
                shards[model_name] = Shard(model_name, -1, -1, range_end - range_start)
        
        return shards

    # Define the ranges for different models
    model_ranges = {
        'clip': (0, 12),
        'vae_encoder':(12,17),
        'unet':(17,26),
        'vae_decoder': (26, 31) # Example range for unet
    }

    # Call the function and get the shards for all models
    shards = create_shard(shard, model_ranges)

    # Access individual shards
    shard_clip = shards['clip']
    shard_encoder = shards['vae_encoder']
    shard_unet = shards['unet']
    shard_decoder = shards['vae_decoder']
    
    return shard_clip, shard_encoder, shard_unet, shard_decoder



