1. Currently EXO just doesn't start cleanly a lot of the time. I see two kinds of issues:
    b. EXO starts but then after creating an instance that instance never loads (either gets stuck in Loading of Inactive).
2. Currently a lot of requests from the API are timing out, but we still process those requests internally. If an API request times out, we should cancel all corresponding tasks to that API request (why process a request with nobody listening).
4. I'd like to see profiled network latency / bandwidth.
5. I'd like to see how much bandwidth each link is using.
6. We should handle the case where one machine doesn't have the model downloaded and then other machines are waiting on it. In this case we get loads of timeout errors because the others are waiting for the one that needs to download the model.
7. Solve the problem of in continuous batching when a new prompt comes in, it will block decode of the current batch until the prefill is complete.
8. We want people to be able to copy models over to a new device without ever connecting EXO to the internet. Right now EXO require internet connection once to cache some files to check if a download is complete. Instead, we should simply check if there is a non-empty model folder locally with no .partial files. This indicates it's a fully downloaded model that can be loaded.
10. More granular control over how to deploy instances.
12. Nix is great but installing it is a pain and we have ended up in a lot of cases having PATH issues or installation issues. For example, after rebooting mike it seemed to no longer have a nix installation and needed reinstalling. It has a bunch of broken symlinks left over from nix that caused ssh to fail, making it even harder to debug. We need consistent environments (perhaps MDM) so we can guarantee nix is installed properly on each machine.
13. Memory pressure instead of memory used.
14. Show the type of each connection (TB5, Ethernet, etc.) in the UI. Refer to old exo: https://github.com/exo-explore/exo/blob/56f783b38dc6b08ce606b07a5386dc40dae00330/exo/helpers.py#L251
15. Prioritise certain connection types (or by latency). TB5 > Ethernet > WiFi. Refer to old exo: https://github.com/exo-explore/exo/blob/56f783b38dc6b08ce606b07a5386dc40dae00330/exo/helpers.py#L251
16. Dynamically switch to higher priority connection when it becomes available. Probably bring back InstanceReplacedAtomically.
17. Faster model loads by streaming model from other devices in cluster.
18. Add support for specifying the type of network connection to use in a test. Depends on 15/16.
19. Fix mx.distributed.Group typing.
20. Add chat completion cancellations (e.g OpenWebUI has something for cancelling an ongoing request).
21. Make two separate things: tensor or pipeline, and ring or ibv.
22. When downloading for the first time, stuff times out and I think the model never ends up actually loading into memory, or something.
23. Do we need cache_limit? We went back and forth on that a lot because we thought it might be causing issues. One problem is it sets it relative to model size. So if you have multiple models loaded in it will take the most recent model size for the cache_limit. This is problematic if you launch DeepSeek -> Llama for example.
24. Task cancellation. When API http request gets cancelled, it should cancel corresponding task.

Potential refactors:

1. Make ForwarderEvent typed
2. Topology can be simplified
3. Get rid of InstanceReplacedAtomically

Random errors we've run into:

1. exo.shared.types.worker.common.RunnerError: RuntimeError: [ibv] Couldn't connect (error: 60). Traceback: Traceback (most recent call last):
  File "/Users/puffin4/actions-runner/_work/exo/exo/src/exo/worker/runner/runner.py", line 54, in main
    model, tokenizer, sampler, group = await loop.run_in_executor(
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<8 lines>...
    )
    ^
  File "/nix/store/s7ik6dazn4nd2jdg9l36qf5q0z18sjyk-python3-3.13.8/lib/python3.13/concurrent/futures/thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
  File "/Users/puffin4/actions-runner/_work/exo/exo/src/exo/engines/mlx/utils_mlx.py", line 149, in initialize_mlx
    group = mlx_distributed_init(
        model_shard_meta.device_rank,
    ...<4 lines>...
        or (mlx_ibv_devices is not None and len(mlx_ibv_devices) > 1),
    )
  File "/Users/puffin4/actions-runner/_work/exo/exo/src/exo/engines/mlx/utils_mlx.py", line 124, in mlx_distributed_init
    group = mx.distributed.init(
        backend="ring" if hosts is not None else "ibv",
        strict=strict,
    )
RuntimeError: [ibv] Couldn't connect (error: 60)

2. 