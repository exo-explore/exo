from dataclasses import dataclass


@dataclass
class RunnerOpts:
    fast_synch_override: bool | None
    trust_remote_code_override: bool | None
