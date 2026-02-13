3. Task cancellation. When API http request gets cancelled, it should cancel corresponding task.
4. I'd like to see profiled network latency / bandwidth.
5. I'd like to see how much bandwidth each link is using.
7. Solve the problem of in continuous batching when a new prompt comes in, it will block decode of the current batch until the prefill is complete.
8. We want people to be able to copy models over to a new device without ever connecting EXO to the internet. Right now EXO require internet connection once to cache some files to check if a download is complete. Instead, we should simply check if there is a non-empty model folder locally with no .partial files. This indicates it's a fully downloaded model that can be loaded.
13. Memory pressure instead of memory used.
14. Show the type of each connection (TB5, Ethernet, etc.) in the UI. Refer to old exo: https://github.com/exo-explore/exo/blob/56f783b38dc6b08ce606b07a5386dc40dae00330/exo/helpers.py#L251
15. Prioritise certain connection types (or by latency). TB5 > Ethernet > WiFi. Refer to old exo: https://github.com/exo-explore/exo/blob/56f783b38dc6b08ce606b07a5386dc40dae00330/exo/helpers.py#L251
16. Dynamically switch to higher priority connection when it becomes available. Probably bring back InstanceReplacedAtomically.
17. Faster model loads by streaming model from other devices in cluster.
18. Add support for specifying the type of network connection to use in a test. Depends on 15/16.
25. Rethink retry logic
27. Log cleanup - per-module log filters and default to DEBUG log levels
28. Validate RDMA connections with ibv_devinfo in the info gatherer
