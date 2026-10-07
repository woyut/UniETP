from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import defaultdict
from Agent.remote.lifecycle import connect_agent, shutdown_owned_service
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from Agent.general_agent import Agent
from Agent.remote.protocol import (
    METHOD_CLOSE_AGENT,
    METHOD_CREATE_AGENT,
    METHOD_GET_STATS,
    METHOD_INIT_EPISODE,
    METHOD_PING,
    METHOD_PLAN,
    METHOD_SET_SAVE_DIR,
    METHOD_UPDATE_IMG_PATH,
    RPCError,
    STATUS_ERROR,
    make_request,
)
from Agent.remote.serialization import sanitize_additional_info_list


class RemoteAgentProxy(Agent):
    def __init__(
        self,
        save_dir: str,
        host: str,
        port: int,
        authkey: str,
        timeout_s: float = 300.0,
        auto_start_service: bool = False,
        service_python: Optional[str] = None,
        connect_retries: int = 20,
        retry_interval_s: float = 0.5,
        agent_module: str = "Agent.llm_planner",
        agent_class: str = "LLMAgent",
        agent_init_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(save_dir=save_dir)
        self.address: Tuple[str, int] = (host, int(port))
        self.authkey = authkey.encode("utf-8")
        self.timeout_s = timeout_s
        self._lock = Lock()
        self._conn = None
        self._service_proc: Optional[subprocess.Popen] = None
        self._owns_service = False

        self.agent_module = agent_module
        self.agent_class = agent_class
        self.agent_init_kwargs = dict(agent_init_kwargs or {})
        self.agent_init_kwargs["save_dir"] = save_dir

        self.rpc_stats = defaultdict(float)
        self.rpc_counts = defaultdict(int)

        if auto_start_service:
            self._start_service_process(service_python)
            self._owns_service = True

        try:
            self._connect(connect_retries, retry_interval_s)
            self._request(METHOD_PING, {})
            self._request(
                METHOD_CREATE_AGENT,
                {
                    "agent_module": self.agent_module,
                    "agent_class": self.agent_class,
                    "agent_init_kwargs": self.agent_init_kwargs,
                },
            )
        except BaseException:
            self.close()
            raise

    def _start_service_process(self, service_python: Optional[str]):
        from Agent.remote.lifecycle import choose_agent_port, popen_agent_service

        host, port = self.address
        port = choose_agent_port(host, port, self.authkey)
        self.address = (host, port)
        self._service_proc = popen_agent_service(
            service_python or sys.executable,
            host,
            port,
            self.authkey.decode("utf-8"),
            cwd=os.getcwd(),
        )

    def _connect(self, retries: int, retry_interval_s: float):
        last_error = None
        for _ in range(retries):
            if self._service_proc is not None and self._service_proc.poll() is not None:
                raise RuntimeError(
                    "Agent service exited before accepting connections "
                    f"(code {self._service_proc.returncode})."
                )
            try:
                self._conn = connect_agent(
                    self.address[0], self.address[1], self.authkey, retry_interval_s
                )
                return
            except (
                Exception
            ) as exc:  # pragma: no cover - platform dependent network error
                last_error = exc
                time.sleep(retry_interval_s)
        raise RuntimeError(
            f"Failed to connect to agent service at {self.address}: {last_error}"
        )

    def _record_timing(self, key: str, delta: float):
        self.rpc_stats[key] += delta
        self.rpc_counts[key] += 1

    def _request(self, method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._conn is None:
            raise RuntimeError("Agent service connection is not available.")

        request = make_request(method, payload)
        with self._lock:
            t_roundtrip_start = time.time()
            t_send_start = time.time()
            self._conn.send(request)
            t_send_end = time.time()
            ready = self._conn.poll(self.timeout_s)
            if not ready:
                raise TimeoutError(
                    f"RPC timeout for method {method} after {self.timeout_s}s"
                )
            t_recv_start = time.time()
            response = self._conn.recv()
            t_recv_end = time.time()

        self._record_timing("rpc_send", t_send_end - t_send_start)
        self._record_timing("rpc_wait_response", t_recv_start - t_send_end)
        self._record_timing("rpc_recv_decode", t_recv_end - t_recv_start)
        self._record_timing("rpc_roundtrip", t_recv_end - t_roundtrip_start)

        if response.get("status") == STATUS_ERROR:
            err = response.get("error", {})
            raise RPCError(
                message=err.get("message", "unknown remote error"),
                error_type=err.get("type", "RemoteError"),
                remote_traceback=err.get("traceback", ""),
            )
        return response.get("payload", {})

    def init_episode(self, instruction: str, **kwargs):
        super().init_episode(instruction, **kwargs)
        if "simulator_name" in kwargs:
            self.simulator_name = kwargs["simulator_name"]
        self._request(
            METHOD_INIT_EPISODE,
            {
                "instruction": instruction,
                "kwargs": kwargs,
            },
        )

    def plan(
        self,
        obs_list: List[Any],
        action_success_list: List[bool],
        feedback_list: List[str],
        additional_info_list: List[Dict[str, Any]],
    ):
        t_prepare_start = time.time()
        safe_additional_info = sanitize_additional_info_list(additional_info_list)
        t_prepare_end = time.time()
        self._record_timing("rpc_prepare_payload", t_prepare_end - t_prepare_start)

        payload = {
            "obs_list": obs_list,
            "action_success_list": action_success_list,
            "feedback_list": feedback_list,
            "additional_info_list": safe_additional_info,
        }
        resp = self._request(METHOD_PLAN, payload)
        return resp.get("actions", [])

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "rpc_timing_total_s": dict(self.rpc_stats),
            "rpc_call_counts": dict(self.rpc_counts),
        }
        try:
            remote = self._request(METHOD_GET_STATS, {})
            stats["remote_agent_stats"] = remote.get("agent_stats", {})
        except Exception as exc:  # pragma: no cover
            stats["remote_agent_stats_error"] = str(exc)
        return stats

    def close(self):
        if self._conn is not None:
            try:
                self._request(METHOD_CLOSE_AGENT, {})
            except Exception:
                pass
            self._conn.close()
            self._conn = None

        if self._owns_service and self._service_proc is not None:
            shutdown_owned_service(
                self.address[0],
                self.address[1],
                self.authkey,
                self._service_proc,
            )
            self._service_proc = None

    def update_img_path(self, img_path: str, img_h: int, img_w: int):
        super().update_img_path(img_path, img_h, img_w)
        self._request(
            METHOD_UPDATE_IMG_PATH,
            {"img_path": img_path, "img_h": img_h, "img_w": img_w},
        )

    def set_save_dir(self, save_dir: str):
        super().set_save_dir(save_dir)
        self.agent_init_kwargs["save_dir"] = save_dir
        self._request(METHOD_SET_SAVE_DIR, {"save_dir": save_dir})
