# Perf improvements

Target: 460 tok/sec
- removing sample goes from 369 -> 402
- performance degrades as we generate more tokens
- make mlx inference engien synchronous, removing thread pool executor: 402 -> 413
- remove self.on_opaque_status.trigger_all: 413 -> 418
