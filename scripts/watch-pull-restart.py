#!/usr/bin/env python3

"""
watch-pull-restart.py  —  Unix-only

Runs a command, periodically checks git upstream, pulls if upstream is ahead,
and gracefully restarts the command. Watcher logs go to STDERR; your app's
output goes straight to the console (STDOUT/STDERR).

Assumptions:
  - current branch tracks an upstream (i.e., @{u} exists)
  - pulls must be fast-forward (remote-ahead workflow)

Arguments:
  - cmd: Command to run/manage (e.g. './run.sh' or 'python -m app').
  - restart-cmd: Optional hook to run after a successful pull (e.g., systemctl restart).
  - sleep-secs: Poll interval while up-to-date.
  - grace-secs: Seconds to wait after SIGTERM before SIGKILL.
  - debounce-secs: Coalesce multiple pulls before restart.

Usage:
  ./watch-pull-restart.py --cmd "./run.sh" --sleep-secs 1
  ./watch-pull-restart.py --cmd "python -m app" --restart-cmd "systemctl --user restart myapp"
  ./watch-pull-restart.py --restart-cmd "systemctl --user restart myapp"   # no managed child; only trigger hook
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from types import FrameType
from typing import Optional


# ---------- logging helpers (to STDERR) ----------
def log(msg: str):
    sys.stderr.write(msg.rstrip() + "\n")
    sys.stderr.flush()


def sep(title: str = ""):
    """Big visual separator for state transitions (to STDERR)."""
    sys.stderr.write("\n\n")
    if title:
        sys.stderr.write(f"===== [watch] {title} =====\n")
    else:
        sys.stderr.write("===== [watch] =====\n")
    sys.stderr.flush()


def run_capture(cmd: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run and capture output; for git plumbing."""
    return subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=check,
    )


# ---------- shell helpers ----------
def is_up_to_date() -> bool:
    subprocess.run("git fetch --quiet",
                   shell=True)  # Quiet fetch; ignore network errors (we'll just try again next tick)
    try:
        current = run_capture("git rev-parse HEAD", check=True).stdout.strip()
        upstream = run_capture("git rev-parse @{u}", check=True).stdout.strip()
        return current == upstream
    except subprocess.CalledProcessError:
        return True  # No upstream or other git error; treat as up-to-date to avoid thrash


def pull_ff_only() -> bool:
    """Returns True if pull applied changes, False if already up-to-date."""
    try:
        cp = run_capture("git pull --ff-only --no-rebase", check=True)
        return "Already up to date" not in cp.stdout and cp.returncode == 0  # Git prints "Already up to date." on no-op; cheap heuristic
    except subprocess.CalledProcessError as e:
        log("[watch] git pull failed:")
        if e.stdout:  # pyright: ignore[reportAny]
            log(e.stdout)  # pyright: ignore[reportAny]
        if e.stderr:  # pyright: ignore[reportAny]
            log(e.stderr)  # pyright: ignore[reportAny]
        return False


# ---------- managed processes ----------
class ManagedProc:
    def __init__(self, cmd: Optional[str], grace_secs: float):
        self.cmd = cmd
        self.grace = grace_secs
        self.child: Optional[subprocess.Popen[bytes]] = None

    def start(self):
        if not self.cmd:
            return
        if self.child and self.child.poll() is None:
            return
        sep("starting main cmd")
        log(f"[watch] starting: {self.cmd}")
        # New process group so we can signal the entire tree (shell + children)
        self.child = subprocess.Popen(
            self.cmd,
            shell=True,  # allow shell features in --cmd
            stdout=None,  # inherit parent's stdout (your app prints normally)
            stderr=None,  # inherit parent's stderr
            stdin=None,
            preexec_fn=os.setsid,  # create new session (PGID == child PID)
        )

    def stop_gracefully(self):
        if not self.child:
            return
        if self.child.poll() is not None:
            self.child = None
            return

        sep("stopping main cmd (SIGTERM)")
        try:
            os.killpg(self.child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        deadline = time.time() + self.grace
        while time.time() < deadline:
            if self.child.poll() is not None:
                self.child = None
                return
            time.sleep(0.1)

        sep("main cmd unresponsive; SIGKILL")
        try:
            os.killpg(self.child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.child = None

    def forward_signal(self, sig: int):
        if not self.child or self.child.poll() is not None:
            return
        try:
            os.killpg(self.child.pid, sig)
        except ProcessLookupError:
            pass


class OneShotHook:
    """
    One-shot hook command (e.g., systemctl restart).
    Runs to completion with inherited stdio so its output is visible.
    """

    def __init__(self, cmd: Optional[str], grace_secs: float):
        self.cmd = cmd
        self.grace = grace_secs
        self.child: Optional[subprocess.Popen[bytes]] = None

    def run(self) -> int:
        if not self.cmd:
            return 0
        sep("running restart hook")
        log(f"[watch] hook: {self.cmd}")
        self.child = subprocess.Popen(
            self.cmd,
            shell=True,
            stdout=None,  # inherit stdio
            stderr=None,
            stdin=None,
            preexec_fn=os.setsid,
        )
        # Wait with grace/kill if needed (rare for hooks, but symmetric)
        deadline = time.time() + self.grace
        while True:
            rc = self.child.poll()
            if rc is not None:
                self.child = None
                return rc
            if time.time() > deadline:
                sep("hook exceeded grace; SIGKILL")
                try:
                    os.killpg(self.child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.child = None
                return 137  # killed
            time.sleep(0.1)

    def forward_signal(self, sig: int):
        if not self.child or self.child.poll() is not None:
            return
        try:
            os.killpg(self.child.pid, sig)
        except ProcessLookupError:
            pass


# ---------- main loop ----------
def main():
    # CMD commands
    ap = argparse.ArgumentParser(description="Auto-pull & restart on upstream changes (Unix).")
    ap.add_argument("--cmd", help="Command to run/manage (e.g. './run.sh' or 'python -m app').")
    ap.add_argument("--restart-cmd", help="Optional hook to run after a successful pull (e.g., systemctl restart).")
    ap.add_argument("--sleep-secs", type=float, default=0.5, help="Poll interval while up-to-date.")
    ap.add_argument("--grace-secs", type=float, default=5.0, help="Seconds to wait after SIGTERM before SIGKILL.")
    ap.add_argument("--debounce-secs", type=float, default=0.5, help="Coalesce multiple pulls before restart.")
    args = ap.parse_args()

    # get CMD command values
    cmd = args.cmd  # pyright: ignore[reportAny]
    assert cmd is None or isinstance(cmd, str)
    restart_cmd = args.restart_cmd  # pyright: ignore[reportAny]
    assert cmd is None or isinstance(restart_cmd, str)
    sleep_secs = args.sleep_secs  # pyright: ignore[reportAny]
    assert sleep_secs is not None and isinstance(sleep_secs, float)
    grace_secs = args.grace_secs  # pyright: ignore[reportAny]
    assert sleep_secs is not None and isinstance(grace_secs, float)
    debounce_secs = args.debounce_secs  # pyright: ignore[reportAny]
    assert sleep_secs is not None and isinstance(debounce_secs, float)

    # start managed proc
    proc = ManagedProc(cmd, grace_secs)
    hook = OneShotHook(restart_cmd, grace_secs)

    # signal handling for graceful exit
    exiting = {"flag": False}

    def _handle(sig_num: int, _frame: Optional[FrameType]):
        sep(f"received signal {sig_num}; exiting")
        exiting["flag"] = True
        proc.forward_signal(sig_num)
        hook.forward_signal(sig_num)

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    # Initial start (if managing a process)
    proc.start()

    pending_restart = False
    last_change = 0.0
    while not exiting["flag"]:
        try:
            if not is_up_to_date():
                sep("upstream ahead; pulling")
                changed = pull_ff_only()
                if changed:
                    last_change = time.time()
                    pending_restart = True

            # handle debounce window
            if pending_restart and (time.time() - last_change) >= debounce_secs:
                # Optional hook first
                if restart_cmd:
                    rc = hook.run()
                    if rc != 0:
                        sep(f"hook exited with {rc}")
                # Then bounce managed process
                if cmd:
                    proc.stop_gracefully()
                    proc.start()
                pending_restart = False
                sep("restart cycle complete")

            # keep the child alive if it crashed without a pull
            if cmd and (proc.child is None or proc.child.poll() is not None):
                sep("main cmd exited; restarting")
                proc.start()

            time.sleep(sleep_secs)
        except Exception as e:
            sep("loop error")
            log(f"[watch] {e}")
            time.sleep(2.0)

    # graceful shutdown on exit
    proc.stop_gracefully()
    sep("bye")


if __name__ == "__main__":
    main()
