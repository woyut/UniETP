"""Start and stop the agent RPC process without leaving it bound after a crash."""

from __future__ import annotations

import os
import signal
import socket
import struct
import subprocess
import time
from multiprocessing.connection import Connection, answer_challenge, deliver_challenge
from typing import Optional

from Agent.remote.protocol import METHOD_SHUTDOWN_SERVER, make_request

_AGENT_SERVICE_MARKER = "Agent.remote.agent_service"
_HANDSHAKE_TIMEOUT_S = 2.0


def bind_lifetime_to_parent() -> None:
    """Exit with the parent process so a crashed launcher cannot leave a listener."""
    if os.name != "posix":
        return
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() == 1:
        raise SystemExit("agent service parent already exited")


def tcp_port_open(host: str, port: int) -> bool:
    """Return whether a process is listening on this port.

    This does not connect or bind. A raw connect breaks the auth handshake,
    and a bind fails while old connections are still in TIME_WAIT.
    """
    del host
    return bool(_listen_inodes(port))


def _pids_listening_on(port: int) -> list[int]:
    inodes = _listen_inodes(port)
    if not inodes:
        return []
    pids: list[int] = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        fd_dir = f"/proc/{name}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in inodes:
                pids.append(int(name))
                break
    return pids


def _listen_inodes(port: int) -> set[str]:
    hex_port = f"{int(port):04X}"
    inodes: set[str] = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = open(path, encoding="utf-8").read().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10 or parts[3] != "0A":
                continue
            if parts[1].rsplit(":", 1)[-1].upper() != hex_port:
                continue
            inodes.add(parts[9])
    return inodes


def _command_line(pid: int) -> str:
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace")


def _set_fd_timeout(fd: int, timeout_s: float) -> None:
    """Apply SO_RCVTIMEO/SO_SNDTIMEO. os.read ignores Python's socket timeout."""
    seconds = int(timeout_s)
    microseconds = int(round((timeout_s - seconds) * 1_000_000))
    timeval = struct.pack("ll", seconds, microseconds)
    held = socket.socket(fileno=fd)
    try:
        held.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, timeval)
        held.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO, timeval)
    finally:
        held.detach()


def connect_agent(host: str, port: int, authkey: bytes, timeout_s: float):
    """Connect with a bounded handshake, then clear the socket timeout."""
    timeout_s = max(0.2, float(timeout_s))
    sock = socket.create_connection((host, int(port)), timeout_s)
    sock.settimeout(None)
    _set_fd_timeout(sock.fileno(), timeout_s)
    connection = Connection(sock.detach())
    try:
        answer_challenge(connection, authkey)
        deliver_challenge(connection, authkey)
        _set_fd_timeout(connection.fileno(), 0)
        return connection
    except Exception:
        connection.close()
        raise


def _request_shutdown(
    host: str, port: int, authkey: bytes, timeout_s: float = _HANDSHAKE_TIMEOUT_S
) -> bool:
    try:
        connection = connect_agent(host, port, authkey, timeout_s)
    except Exception:
        return False
    try:
        connection.send(make_request(METHOD_SHUTDOWN_SERVER, {}))
        if not connection.poll(timeout_s):
            return False
        response = connection.recv()
        return response.get("status") == "ok"
    except Exception:
        return False
    finally:
        connection.close()


def _wait_until_closed(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not tcp_port_open(host, port):
            return True
        time.sleep(0.1)
    return not tcp_port_open(host, port)


def _stop_pids(pids: list[int]) -> None:
    for pid in pids:
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not any(_process_alive(pid) for pid in pids):
            return
        time.sleep(0.1)
    for pid in pids:
        if pid == os.getpid() or not _process_alive(pid):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _ppid(pid: int) -> Optional[int]:
    try:
        lines = open(f"/proc/{pid}/status", encoding="utf-8")
    except OSError:
        return None
    with lines:
        for line in lines:
            if line.startswith("PPid:"):
                return int(line.split()[1])
    return None


def _agent_service_pids() -> list[int]:
    pids = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == os.getpid():
            continue
        if _AGENT_SERVICE_MARKER in _command_line(pid):
            pids.append(pid)
    return pids


def _parent_is_dead(pid: int) -> bool:
    ppid = _ppid(pid)
    return ppid is None or ppid == 1 or not _process_alive(ppid)


def reap_orphan_agent_services() -> None:
    """Stop agent services whose parent has already exited."""
    orphans = [pid for pid in _agent_service_pids() if _parent_is_dead(pid)]
    if orphans:
        print(
            f"[Agent] Stopping orphan agent service(s): {', '.join(map(str, orphans))}",
            flush=True,
        )
        _stop_pids(orphans)


def reclaim_stale_agent_service(
    host: str, port: int, authkey: bytes, timeout_s: float = 5.0
) -> bool:
    """Try to free a port held by a previous agent service.

    Returns True when the port is free. A foreign listener is left running.
    The shutdown handshake is bounded so a stuck peer cannot block startup.
    """
    reap_orphan_agent_services()
    if not tcp_port_open(host, port):
        return True
    listeners = _pids_listening_on(port)
    live_owners = [
        pid
        for pid in listeners
        if _AGENT_SERVICE_MARKER in _command_line(pid) and not _parent_is_dead(pid)
    ]
    if live_owners:
        return False
    _request_shutdown(host, port, authkey, _HANDSHAKE_TIMEOUT_S)
    if _wait_until_closed(host, port, timeout_s):
        return True

    listeners = _pids_listening_on(port)
    orphans = [
        pid
        for pid in listeners
        if _AGENT_SERVICE_MARKER in _command_line(pid) and _parent_is_dead(pid)
    ]
    if orphans:
        print(
            f"[Agent] Stopping previous agent service on {host}:{port}: "
            f"{', '.join(map(str, orphans))}",
            flush=True,
        )
        _stop_pids(orphans)
        if _wait_until_closed(host, port, timeout_s):
            return True
    return not tcp_port_open(host, port)


def _ephemeral_port(host: str) -> int:
    bind_host = host if host not in ("0.0.0.0", "", "*") else "127.0.0.1"
    for _ in range(50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((bind_host, 0))
            port = int(sock.getsockname()[1])
        if not tcp_port_open(host, port):
            return port
    raise RuntimeError("Could not find a free port for the agent service.")


def choose_agent_port(host: str, preferred_port: int, authkey: bytes) -> int:
    """Use the preferred port, or a free one when it cannot be reclaimed.

    Only this process's own previous agent services are stopped. Another
    program listening on the preferred port is left alone.
    """
    preferred_port = int(preferred_port)
    if reclaim_stale_agent_service(host, preferred_port, authkey):
        return preferred_port
    port = _ephemeral_port(host)
    print(
        f"[Agent] {host}:{preferred_port} is in use; "
        f"starting the agent service on {host}:{port}.",
        flush=True,
    )
    return port


def shutdown_owned_service(
    host: str,
    port: int,
    authkey: bytes,
    process: Optional[subprocess.Popen],
    timeout_s: float = _HANDSHAKE_TIMEOUT_S,
) -> None:
    if process is None:
        return
    if process.poll() is None:
        _request_shutdown(host, port, authkey, timeout_s)
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout_s)


def popen_agent_service(
    python_exec: str,
    host: str,
    port: int,
    authkey: str,
    cwd: Optional[str] = None,
) -> subprocess.Popen:
    command = [
        python_exec,
        "-m",
        _AGENT_SERVICE_MARKER,
        "--host",
        host,
        "--port",
        str(port),
        "--authkey",
        authkey,
    ]
    # A new session keeps simulator shutdown signals in the behavior worker
    # from reaching this process. prctl(PDEATHSIG) still stops it when the
    # launcher exits, and the launcher also kills it in shutdown_owned_service.
    return subprocess.Popen(command, cwd=cwd, start_new_session=True)
