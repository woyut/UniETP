from dataclasses import dataclass
from typing import Any, Dict, Optional
import uuid


PROTOCOL_VERSION = "v1"

METHOD_PING = "ping"
METHOD_CREATE_AGENT = "create_agent"
METHOD_INIT_EPISODE = "init_episode"
METHOD_PLAN = "plan"
METHOD_GET_STATS = "get_stats"
METHOD_CLOSE_AGENT = "close_agent"
METHOD_SHUTDOWN_SERVER = "shutdown_server"
METHOD_UPDATE_IMG_PATH = "update_img_path"
METHOD_SET_SAVE_DIR = "set_save_dir"

STATUS_OK = "ok"
STATUS_ERROR = "error"


@dataclass
class RPCError(Exception):
    message: str
    error_type: str = "RPCError"
    remote_traceback: Optional[str] = None

    def __str__(self) -> str:
        if self.remote_traceback:
            return f"{self.error_type}: {self.message}\n{self.remote_traceback}"
        return f"{self.error_type}: {self.message}"


def make_request(
    method: str, payload: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": str(uuid.uuid4()),
        "method": method,
        "payload": payload or {},
    }


def make_success_response(
    request_id: str, payload: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "status": STATUS_OK,
        "payload": payload or {},
    }


def make_error_response(
    request_id: str, message: str, error_type: str, traceback_text: str = ""
) -> Dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "status": STATUS_ERROR,
        "error": {
            "message": message,
            "type": error_type,
            "traceback": traceback_text,
        },
    }
