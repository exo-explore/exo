#!/usr/bin/env python3
import subprocess, pickle, shlex, sys, os
from typing import Dict, List, Tuple
from tinygrad.engine.graph import print_tree
from tinygrad.helpers import colored
from tinygrad.ops import LazyOp

def _run(name:str, cmd:List[str], env:Dict[str, str]) -> List[Tuple[LazyOp, ...]]:
  commit = subprocess.check_output(["git", "rev-parse", name], encoding="utf-8").strip()
  subprocess.run(["git", "checkout", commit], check=True)
  subprocess.run(cmd, env={**env, "SAVE_SCHEDULE_PATH": f"{commit}.pkl"})
  return pickle.load(open(f"./{commit}.pkl", "rb"))

def _get_cmd():
  parts, env = shlex.split(sys.argv[1]), {**os.environ, "SAVE_SCHEDULE": "1", "CAPTURE_AST": "1"}
  env.update({k: v for p in parts if "=" in p for k, v in [p.split("=")]})
  return [p for p in parts if "=" not in p], env

if __name__ == "__main__":
  cmd, env = _get_cmd()
  feat = _run("HEAD", cmd, env)
  master = _run("master", cmd, env)

  assert len(master) == len(feat)
  for m, f in zip(master, feat):
    try: assert m == f
    except AssertionError as e:
      print(colored("FAILED FOR AST: ", "red"))
      print("expected:")
      for op in m: print_tree(op)
      print("got:")
      for op in f: print_tree(op)
      raise e
