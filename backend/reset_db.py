import subprocess as _subprocess

_original_subprocess_run = _subprocess.run

def _patched_subprocess_run(*args, **kwargs):
    cmd = args[0] if args else kwargs.get("args", "")
    cmd_str = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)

    if "pg_ctl" in cmd_str and kwargs.get("timeout") is not None:
        kwargs["timeout"] = max(kwargs["timeout"], 90)

    return _original_subprocess_run(*args, **kwargs)

_subprocess.run = _patched_subprocess_run
print("🔧 Patched subprocess.run: pg_ctl timeout extended to >=90s")

import pixeltable as pxt

print("Wiping old database...")
pxt.drop_dir("video_rag", force=True)
print("✅ Database successfully reset!")