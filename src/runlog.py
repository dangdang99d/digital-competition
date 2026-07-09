"""Log the exact command used to launch a script, so it persists in the run log.

Every entry-point script calls `log_cmd()` right after arg-parsing; the invocation
then shows up in that run's log / sbatch output — no separate command log to maintain.
Note: env prefixes (e.g. CUDA_VISIBLE_DEVICES=0) are NOT part of sys.argv, so they
are not captured here.
"""
import shlex
import sys

from loguru import logger


def log_cmd():
    """Emit the full invocation (`python` + argv) to the log."""
    logger.info(f"CMD: python {shlex.join(sys.argv)}")
