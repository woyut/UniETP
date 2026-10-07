import argparse
import os

from Agent.remote.lifecycle import bind_lifetime_to_parent
from Agent.remote.service import AgentService


def main():
    bind_lifetime_to_parent()
    parser = argparse.ArgumentParser(description="Start the UniETP agent RPC service.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55001)
    parser.add_argument(
        "--authkey",
        type=str,
        default=os.environ.get("UNIETP_AGENT_AUTHKEY", "unietp-agent-rpc"),
    )
    args = parser.parse_args()

    service = AgentService(
        address=(args.host, args.port), authkey=args.authkey.encode("utf-8")
    )
    service.serve_forever()


if __name__ == "__main__":
    main()
