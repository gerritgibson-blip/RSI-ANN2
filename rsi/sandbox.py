"""Execution sandbox for model-written code.

Model code runs in a separate `python -I` process as the unprivileged `sandbox`
user, in a throwaway directory, with CPU/memory/file/process rlimits, a stripped
environment and sockets disabled. (This container cannot create network
namespaces, so the socket block is best-effort, not a hard boundary.)

Grading uses a per-run random nonce printed only after every hidden assertion
passes, so a solution that calls sys.exit(0)/os._exit, or kills the harness,
cannot fake a pass.
"""
import os, pwd, resource, secrets, subprocess, tempfile, shutil

_PREAMBLE = r'''
import socket as _s
def _blocked(*a, **k): raise OSError("network disabled in sandbox")
_s.socket = _blocked; _s.create_connection = _blocked
del _s, _blocked
'''

try:
    _pw = pwd.getpwnam("sandbox")
    _UID, _GID = _pw.pw_uid, _pw.pw_gid
except KeyError:  # fall back to running as current user
    _UID = _GID = None


def _limits(cpu_s, mem_mb):
    def fn():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        os.setsid()
        if _GID is not None:
            os.setgid(_GID); os.setuid(_UID)
    return fn


def run_code(code: str, timeout: float = 10.0, mem_mb: int = 1024, stdin: str = "") -> dict:
    d = tempfile.mkdtemp(prefix="sbx_")
    try:
        path = os.path.join(d, "main.py")
        with open(path, "w") as f:
            f.write(_PREAMBLE + "\n" + code)
        if _UID is not None:
            os.chown(d, _UID, _GID); os.chown(path, _UID, _GID)
        env = {"PATH": "/usr/bin:/bin", "HOME": d, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            p = subprocess.run(["python3", "-I", path], cwd=d, env=env, input=stdin,
                               capture_output=True, text=True, timeout=timeout,
                               preexec_fn=_limits(int(timeout) + 1, mem_mb))
            out, err, rc, to = p.stdout, p.stderr, p.returncode, False
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            err, rc, to = "TIMEOUT", -9, True
        return {"stdout": out[-4000:], "stderr": err[-4000:], "returncode": rc, "timeout": to}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def grade(solution: str, test_code: str, timeout: float = 15.0) -> dict:
    """Run hidden tests against a solution. Pass only if the nonce is printed."""
    nonce = secrets.token_hex(16)
    prog = f"{solution}\n\n# ---- hidden tests ----\n{test_code}\n\nprint('{nonce}')\n"
    r = run_code(prog, timeout=timeout)
    passed = (r["returncode"] == 0) and (nonce in r["stdout"])
    # strip nonce so it can never leak back to the model
    r["stdout"] = r["stdout"].replace(nonce, "")
    r["passed"] = passed
    return r
