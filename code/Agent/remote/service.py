from __future__ import annotations

import importlib
import traceback
from multiprocessing.connection import Listener
from typing import Any, Dict, Optional, Tuple

from Agent.remote.protocol import (
    METHOD_CLOSE_AGENT,
    METHOD_CREATE_AGENT,
    METHOD_GET_STATS,
    METHOD_INIT_EPISODE,
    METHOD_PING,
    METHOD_PLAN,
    METHOD_SET_SAVE_DIR,
    METHOD_SHUTDOWN_SERVER,
    METHOD_UPDATE_IMG_PATH,
    PROTOCOL_VERSION,
    make_error_response,
    make_success_response,
)


class AgentService:
    def __init__(self, address: Tuple[str, int], authkey: bytes):
        self.address = address
        self.authkey = authkey
        self._listener: Optional[Listener] = None
        self._agent = None
        self._running = False

    def _create_agent(
        self, module_name: str, class_name: str, init_kwargs: Dict[str, Any]
    ):
        if self._agent is not None:
            if hasattr(self._agent, "close"):
                self._agent.close()
            self._agent = None

        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        self._agent = cls(**init_kwargs)

    def _handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        request_id = request.get("request_id", "")
        try:
            if request.get("protocol_version") != PROTOCOL_VERSION:
                raise ValueError(
                    f"Protocol mismatch: expected {PROTOCOL_VERSION}, got {request.get('protocol_version')}"
                )
            method = request.get("method")
            payload = request.get("payload", {})

            if method == METHOD_PING:
                return make_success_response(request_id, {"pong": True})

            if method == METHOD_CREATE_AGENT:
                self._create_agent(
                    module_name=payload["agent_module"],
                    class_name=payload["agent_class"],
                    init_kwargs=payload.get("agent_init_kwargs", {}),
                )
                return make_success_response(request_id, {"created": True})

            if method == METHOD_SHUTDOWN_SERVER:
                if self._agent is not None and hasattr(self._agent, "close"):
                    self._agent.close()
                self._agent = None
                self._running = False
                return make_success_response(request_id, {"shutdown": True})

            if self._agent is None:
                raise RuntimeError(
                    "Agent has not been created. Call create_agent first."
                )

            if method == METHOD_INIT_EPISODE:
                self._agent.init_episode(
                    payload["instruction"], **payload.get("kwargs", {})
                )
                return make_success_response(request_id, {})
            if method == METHOD_PLAN:
                actions = self._agent.plan(
                    payload["obs_list"],
                    payload["action_success_list"],
                    payload["feedback_list"],
                    payload["additional_info_list"],
                )
                return make_success_response(request_id, {"actions": actions})
            if method == METHOD_GET_STATS:
                stats = {}
                if hasattr(self._agent, "get_stats"):
                    got = self._agent.get_stats()
                    if isinstance(got, dict):
                        stats = got
                elif hasattr(self._agent, "stats"):
                    got = getattr(self._agent, "stats")
                    if isinstance(got, dict):
                        stats = got
                return make_success_response(request_id, {"agent_stats": stats})
            if method == METHOD_UPDATE_IMG_PATH:
                self._agent.update_img_path(
                    payload["img_path"], payload["img_h"], payload["img_w"]
                )
                return make_success_response(request_id, {})
            if method == METHOD_SET_SAVE_DIR:
                if not hasattr(self._agent, "set_save_dir"):
                    raise AttributeError("Agent does not support set_save_dir")
                self._agent.set_save_dir(payload["save_dir"])
                return make_success_response(request_id, {})
            if method == METHOD_CLOSE_AGENT:
                if hasattr(self._agent, "close"):
                    self._agent.close()
                self._agent = None
                return make_success_response(request_id, {"closed": True})

            raise ValueError(f"Unknown method: {method}")
        except Exception as exc:
            return make_error_response(
                request_id=request_id,
                message=str(exc),
                error_type=type(exc).__name__,
                traceback_text=traceback.format_exc(),
            )

    def serve_forever(self):
        self._listener = Listener(self.address, authkey=self.authkey)
        self._running = True
        while self._running:
            conn = self._listener.accept()
            try:
                while self._running:
                    try:
                        request = conn.recv()
                    except EOFError:
                        break
                    response = self._handle_request(request)
                    conn.send(response)
            finally:
                conn.close()

        if self._listener is not None:
            self._listener.close()
