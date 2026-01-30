# Missed things
[X] Log EXO_LIBP2P_NAMESPACE on start in exo/main.py
[X] Ordering of warmup was changed, which is wrong. It was changed to rank < n-1, then rank=n-1. It should be rank!=0 then rank=0 (this matches the auto_parallel implementation. NOTE: we use a different convention to mlx-lm, our terminal rank is rank=n-1 whereas mlx-lm is rank=0 hence i can see why this was changed wrongly).
[X] Downloads keying by model_id not shard_metadata (worker/plan.py, worker/main.py).
[X] Fetching download status of all models on start
[X] Deduplication of tasks in plan_step.
[X] resolve_allow_patterns should just be wildcard now.
[X] no mx_barrier in genreate.py mlx_generate at the end.
[] cache assertion not needed in auto_parallel.py PipelineLastLayer.
[X] GPTOSS support dropped in auto_parallel.py.
[X] sharding changed "all-to-sharded" became _all_to_sharded in auto_parallel.py.
[X] same as above with "sharded-to-all" became _sharded_to_all in auto_parallel.py.
[X] Dropped support for Ministral3Model, DeepseekV32Model, Glm4MoeModel, Qwen3NextModel, GptOssMode in auto_parallel.py.
[] Dropped prefill/decode code in auto_parallel.py and utils_mlx.py.
[X] KV_CACHE_BITS should be None to disable quantized KV cache.
[X] Dropped _set_nofile_limit in utils_mlx.py.
[X] We have group optional in load_mlx_items in utils_mlx.py.
[] Dropped add_missing_chat_templates for GptOss in load_mlx_items in utils_mlx.py.
[X] Dropped model.make_cache in make_kv_cache in utils_mlx.py.
[X] We put cache limit back in utils_mlx.py.
[] topology.py remove_node removes the connections after checking if node is is in self._node_id_to_rx_id_map. on beta_1 it checks after, so would remove stale connections I guess?
[] Missing Glm 4.7 model cards (this isn't ready yet but should be picked up, probably create an issue... the blocker is transforemrs version doesn't support the tokenizer for Glm 4.7. rc-1 does but we can't upgrade as it breaks other things.)
[] try-except in _command_processor only excepts ValueError. This was silently failing leading to un-debuggable errors (we had a KeyError that was happening ). Changed this to catch Exception instead of ValueError. See exo-v2 89ae38405e0052e3c22405daf094b065878aa873 and fb99fea69b5a39017efc90c5dad0072e677455f0.
[X] In placement.py, place_instance no longer looks at model_meta.supports_tensor and check if this tensor parallel number of nodes is supported by the model's tensor dimensions.
[X] In placement.py, place_instanec, we no longer have the special case to exclude DeepSeek v3.1 pipeline parallel (it doesn't work).
[] logger.warning("You have likely selected ibv for a single node instance; falling back to MlxRing") was changed to debug. That will spam this warning since it happens every time we query instance previews.
[X] In placement_utils.py, get_mlx_jaccl_coordinators, We no longer prioritise Jaccl Coordinator IP. Now it picks the first one, which is unstable (Jaccl coordinator over TB5 is unstable).



[X] Downloads keying by model_id not shard_metadata (worker/plan.py, worker/main.py).
[X] Fetching download status of all models on start
[X] Deduplication of tasks in plan_step.
[X] resolve_allow_patterns should just be wildcard now.
[X] KV_CACHE_BITS should be None to disable quantized KV cache.
[X] We put cache limit back in utils_mlx.py.
[X] In placement.py, place_instance no longer looks at model_meta.supports_tensor and check if this tensor parallel number of nodes is supported by the model's tensor dimensions.
[X] In placement.py, place_instanec, we no longer have the special case to exclude DeepSeek v3.1 pipeline parallel (it doesn't work).
[X] In placement_utils.py, get_mlx_jaccl_coordinators, We no longer prioritise Jaccl Coordinator IP. Now it picks the first one, which is unstable (Jaccl coordinator over TB5 is unstable).


