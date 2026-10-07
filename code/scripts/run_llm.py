from __future__ import annotations

import copy
import gzip
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import argparse
from argparse import ArgumentParser, Namespace
import math
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from task.benchmark_splits import BENCHMARK_SPLITS

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

_SCORE_LINE = re.compile(
    r"^Score:\s*([^,\s]+)\s*,\s*environment steps:\s*\d+\s*$"
)


def _parse_args() -> Namespace:
    parser = ArgumentParser(description="Evaluate an LLM agent in a UniETP simulator.")
    parser.add_argument("--dataset_file", required=True)
    parser.add_argument(
        "--simulator_name",
        required=True,
        choices=["THOR", "VH", "Hab", "BEHAVIOR"],
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10_000)
    parser.add_argument("--agent_save_dir", default="./outputs")
    parser.add_argument("--model_type", default="api")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_config_file")
    parser.add_argument("--nav_action_mode", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--manip_action_mode", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--no_action_success", action="store_true")
    parser.add_argument("--no_bbox_coord", action="store_true")
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--max_failed_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--procthor_dataset_dir", default=os.environ.get("PROCTHOR_DATASET_DIR")
    )
    parser.add_argument(
        "--habitat_data_dir", default=os.environ.get("HABITAT_DATA_DIR")
    )
    parser.add_argument(
        "--vh_exec_file", default=os.environ.get("VIRTUALHOME_EXECUTABLE")
    )
    parser.add_argument("--unity_port", default="8080")
    parser.add_argument("--unity_timeout", type=float, default=180.0)

    parser.add_argument("--agent_mode", choices=["local", "remote"], default="local")
    parser.add_argument("--agent_host", default="127.0.0.1")
    parser.add_argument("--agent_port", type=int, default=55_001)
    parser.add_argument(
        "--agent_authkey",
        default=os.environ.get("UNIETP_AGENT_AUTHKEY", "unietp-agent-rpc"),
    )
    parser.add_argument("--agent_rpc_timeout", type=float, default=300.0)
    parser.add_argument("--agent_auto_start_service", action="store_true")
    parser.add_argument("--agent_service_python")
    parser.add_argument("--agent_service_module", default="Agent.llm_planner")
    parser.add_argument("--agent_service_class", default="LLMAgent")
    parser.add_argument("--agent_connect_retries", type=int, default=40)
    parser.add_argument("--agent_retry_interval_s", type=float, default=0.5)
    parser.add_argument("--thor_timeout_retries", type=int, default=3)
    parser.add_argument("--thor_restart_wait_sec", type=float, default=5.0)
    parser.add_argument("--behavior_worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _validate_args(args: Namespace) -> None:
    if args.start < 0 or args.end < args.start:
        raise ValueError("Expected 0 <= start <= end.")
    if args.nav_action_mode != args.manip_action_mode:
        raise ValueError("Navigation and manipulation modes must match.")
    if not Path(args.dataset_file).is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset_file}")
    if args.model_config_file and not Path(args.model_config_file).is_file():
        raise FileNotFoundError(f"Model config not found: {args.model_config_file}")

    if args.simulator_name == "THOR":
        if not args.procthor_dataset_dir:
            raise ValueError(
                "THOR requires --procthor_dataset_dir or PROCTHOR_DATASET_DIR."
            )
        if not Path(args.procthor_dataset_dir).is_dir():
            raise FileNotFoundError(
                f"ProcTHOR dataset directory not found: {args.procthor_dataset_dir}"
            )
    elif args.simulator_name == "VH":
        if not args.vh_exec_file:
            raise ValueError("VH requires --vh_exec_file or VIRTUALHOME_EXECUTABLE.")
        if not Path(args.vh_exec_file).is_file():
            raise FileNotFoundError(
                f"VirtualHome executable not found: {args.vh_exec_file}"
            )
    elif args.simulator_name == "Hab":
        if not args.habitat_data_dir:
            raise ValueError("Hab requires --habitat_data_dir or HABITAT_DATA_DIR.")
        if not Path(args.habitat_data_dir).is_dir():
            raise FileNotFoundError(
                f"Habitat data directory not found: {args.habitat_data_dir}"
            )


def _seed_everything(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def _instruction_text(task_item: dict[str, Any]) -> str:
    instruction = task_item.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        return instruction.strip()

    instruction_dict = task_item.get("instruction_dict")
    if isinstance(instruction_dict, dict):
        for key in ("L1_Direct", "L2_Referential", "L3_Syntactic"):
            value = instruction_dict.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise ValueError("Dataset entry does not contain a non-empty instruction.")


def _build_agent(args: Namespace, save_dir: str):
    init_kwargs = {
        "save_dir": save_dir,
        "model_type": args.model_type,
        "model_name": args.model_name,
        "model_config_file": args.model_config_file,
    }
    if args.agent_mode == "remote":
        from Agent.remote.proxy import RemoteAgentProxy

        return RemoteAgentProxy(
            save_dir=save_dir,
            host=args.agent_host,
            port=args.agent_port,
            authkey=args.agent_authkey,
            timeout_s=args.agent_rpc_timeout,
            auto_start_service=args.agent_auto_start_service,
            service_python=args.agent_service_python,
            connect_retries=args.agent_connect_retries,
            retry_interval_s=args.agent_retry_interval_s,
            agent_module=args.agent_service_module,
            agent_class=args.agent_service_class,
            agent_init_kwargs=init_kwargs,
        )
    from Agent.llm_planner import LLMAgent

    return LLMAgent(**init_kwargs)


def _load_procthor_dataset(dataset_dir: str):
    from prior import LazyJsonDataset
    from tqdm import tqdm

    proc_data = {}
    root = Path(dataset_dir)
    for split in ("train", "val", "test"):
        split_path = root / f"{split}.jsonl.gz"
        if not split_path.is_file():
            raise FileNotFoundError(f"ProcTHOR split not found: {split_path}")
        with gzip.open(split_path, "rb") as handle:
            houses = list(tqdm(handle, desc=f"Loading {split}"))
        proc_data[split] = LazyJsonDataset(
            data=houses,
            dataset="procthor-dataset",
            split=split,
        )
    return proc_data


def _create_vh_comm(args: Namespace):
    import VH.unity_launcher_patch  # noqa: F401
    from virtualhome.simulation.unity_simulator import comm_unity

    return comm_unity.UnityCommunication(
        port=args.unity_port,
        file_name=str(Path(args.vh_exec_file).resolve()),
        x_display=None,
        no_graphics=False,
        logging=False,
        timeout_wait=args.unity_timeout,
        docker_enabled=False,
    )


def _is_ai2thor_timeout_error(exc: Exception) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current)
        if isinstance(current, TimeoutError) and "AI2-THOR" in message:
            return True
        if "AI2-THOR backend timed out" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _cleanup_scene_dir_for_retry(save_dir: Path) -> None:
    if not save_dir.is_dir():
        return
    for path in save_dir.iterdir():
        if path.is_file() and path.name.endswith((".png", "_seg.png", ".json")):
            try:
                path.unlink()
            except OSError as exc:
                print(f"[Retry] Could not remove stale file {path}: {exc}")


def _scene_label(scene: dict[str, Any]) -> str:
    if "scene_id" in scene:
        return str(scene["scene_id"])
    scene_args = scene.get("args") or {}
    label = scene_args.get("scene") or scene_args.get("scene_model")
    return str(label) if label is not None else "unknown"


def _resolve_dataset_resource(dataset_file: str, resource: str) -> Path:
    """Resolve a dataset-owned resource relative to the dataset directory."""
    raw_path = Path(resource).expanduser()
    if raw_path.is_absolute():
        raise ValueError(
            f"Dataset resource must be relative to the dataset directory: {resource!r}"
        )

    dataset_dir = Path(dataset_file).resolve().parent
    resolved = (dataset_dir / raw_path).resolve()
    try:
        resolved.relative_to(dataset_dir)
    except ValueError as exc:
        raise ValueError(
            f"Dataset resource escapes the dataset directory: {resource!r}"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Dataset resource not found relative to the dataset directory: {resolved}"
        )
    return resolved


def _scene_metadata(args: Namespace, scene: dict[str, Any], proc_data, comm):
    metadata = copy.deepcopy(scene)
    metadata["simulator_name"] = args.simulator_name
    if "scene_id" in metadata:
        metadata["sceneID"] = metadata["scene_id"]
    elif args.simulator_name == "BEHAVIOR":
        metadata["sceneID"] = metadata["args"]["scene"]
    else:
        metadata["sceneID"] = metadata["scene_id"]
    metadata["max_steps"] = args.max_steps
    metadata["max_failed_steps"] = args.max_failed_steps

    if args.simulator_name == "THOR":
        metadata["proc_data"] = proc_data
        if isinstance(metadata["sceneID"], int):
            metadata["placement_randomization_args"] = None
            metadata["material_randomization_args"] = None
            metadata["lighting_randomization_args"] = None
    elif args.simulator_name == "VH":
        metadata["comm"] = comm
        metadata["scene_graph"] = metadata["vheg"]
        metadata["unity_port"] = args.unity_port
    elif args.simulator_name == "Hab":
        metadata["hab_data_dir"] = str(Path(args.habitat_data_dir).resolve())
        metadata["sceneID"] = metadata["sceneID"].split(".scene_instance.json")[0]
        metadata["img_w"] = 512
        metadata["img_h"] = 512
        metadata["add_object_placement"] = metadata["all_placement"]
        metadata["change_object_state"] = metadata["all_state_change"]
        metadata["change_object_placement"] = metadata["all_new_placement"]
    elif args.simulator_name == "BEHAVIOR":
        scene_args = metadata.get("args")
        if not isinstance(scene_args, dict):
            raise TypeError("BEHAVIOR scene metadata must contain an 'args' object.")
        scene_file = scene_args.get("scene_file")
        if not isinstance(scene_file, str) or not scene_file.strip():
            raise ValueError("BEHAVIOR scene metadata requires a non-empty scene_file.")
        scene_args["scene_file"] = str(
            _resolve_dataset_resource(args.dataset_file, scene_file)
        )
    return metadata


def _load_dataset(args: Namespace) -> list[Any]:
    with open(args.dataset_file, encoding="utf-8") as handle:
        dataset = json.load(handle)
    if not isinstance(dataset, list):
        raise TypeError("Dataset root must be a JSON list.")
    for task_index, task_item in enumerate(dataset):
        if not isinstance(task_item, dict):
            raise TypeError(f"Dataset task {task_index} must be a JSON object.")
        _task_split(args, task_item, task_index)
    return dataset


def _task_split(
    args: Namespace,
    task_item: dict[str, Any],
    task_index: int,
) -> str:
    """Return the task's required, explicitly declared benchmark split."""
    split = task_item.get("split")
    if split in BENCHMARK_SPLITS:
        return split
    raise ValueError(
        f"Dataset task {task_index} has missing or invalid split {split!r}; "
        f"expected an explicit value from {list(BENCHMARK_SPLITS)} in "
        f"{Path(args.dataset_file).name!r}."
    )


def _empty_split_stats() -> dict[str, dict[str, int]]:
    return {
        split: {"successful_episodes": 0, "total_episodes": 0}
        for split in BENCHMARK_SPLITS
    }


def _record_expected_episode(
    split_stats: dict[str, dict[str, int]],
    split: str,
) -> None:
    split_stats[split]["total_episodes"] += 1


def _record_score(
    split_stats: dict[str, dict[str, int]],
    split: str,
    score: float,
) -> None:
    if not math.isfinite(score):
        raise ValueError(f"Non-finite evaluation score for split {split!r}: {score}")
    if score == 1.0:
        split_stats[split]["successful_episodes"] += 1


def _print_evaluation_summary(
    args: Namespace,
    dataset: list[Any],
    split_stats: dict[str, dict[str, int]],
    behavior_no_score_episode_ids: list[int | str] | None = None,
) -> None:
    selected_indices = _selected_task_indices(dataset, args)
    effective_end = min(args.end, len(dataset))
    print("============== Evaluation summary ==============", flush=True)
    print(
        f"Task range: [{args.start}, {effective_end}); "
        f"selected tasks: {len(selected_indices)}",
        flush=True,
    )
    print(
        f"{'split':<22} {'success':>8} {'episodes':>10} {'success_rate':>14}",
        flush=True,
    )
    overall_success = 0
    overall_total = 0
    for split in BENCHMARK_SPLITS:
        successful = split_stats[split]["successful_episodes"]
        total = split_stats[split]["total_episodes"]
        overall_success += successful
        overall_total += total
        rate = f"{successful / total:.2%}" if total else "N/A"
        print(
            f"{split:<22} {successful:>8} {total:>10} {rate:>14}",
            flush=True,
        )
    overall_rate = f"{overall_success / overall_total:.2%}" if overall_total else "N/A"
    print(
        f"{'overall':<22} {overall_success:>8} {overall_total:>10} "
        f"{overall_rate:>14}",
        flush=True,
    )
    if behavior_no_score_episode_ids is not None:
        print(
            "BEHAVIOR episode IDs without Score: "
            f"{json.dumps(behavior_no_score_episode_ids, ensure_ascii=False)}",
            flush=True,
        )


def _score_from_line(line: str) -> float | None:
    match = _SCORE_LINE.match(line.strip())
    if match is None:
        return None
    try:
        score = float(match.group(1))
    except ValueError as exc:
        raise ValueError(f"Invalid Score line: {line.rstrip()!r}") from exc
    if not math.isfinite(score):
        raise ValueError(f"Non-finite Score line: {line.rstrip()!r}")
    return score


def _selected_task_indices(dataset: list[Any], args: Namespace) -> list[int]:
    return [index for index in range(len(dataset)) if args.start <= index < args.end]


def _behavior_child_command(args: Namespace, task_index: int) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--dataset_file",
        args.dataset_file,
        "--simulator_name",
        "BEHAVIOR",
        "--start",
        str(task_index),
        "--end",
        str(task_index + 1),
        "--agent_save_dir",
        args.agent_save_dir,
        "--model_type",
        args.model_type,
        "--model_name",
        args.model_name,
        "--nav_action_mode",
        str(args.nav_action_mode),
        "--manip_action_mode",
        str(args.manip_action_mode),
        "--max_steps",
        str(args.max_steps),
        "--max_failed_steps",
        str(args.max_failed_steps),
        "--seed",
        str(args.seed),
        "--unity_port",
        str(args.unity_port),
        "--unity_timeout",
        str(args.unity_timeout),
        "--agent_mode",
        args.agent_mode,
        "--agent_host",
        args.agent_host,
        "--agent_port",
        str(args.agent_port),
        "--agent_authkey",
        args.agent_authkey,
        "--agent_rpc_timeout",
        str(args.agent_rpc_timeout),
        "--agent_service_module",
        args.agent_service_module,
        "--agent_service_class",
        args.agent_service_class,
        "--agent_connect_retries",
        str(args.agent_connect_retries),
        "--agent_retry_interval_s",
        str(args.agent_retry_interval_s),
        "--thor_timeout_retries",
        str(args.thor_timeout_retries),
        "--thor_restart_wait_sec",
        str(args.thor_restart_wait_sec),
        "--behavior_worker",
    ]
    if args.model_config_file:
        command.extend(["--model_config_file", args.model_config_file])
    if args.procthor_dataset_dir:
        command.extend(["--procthor_dataset_dir", args.procthor_dataset_dir])
    if args.habitat_data_dir:
        command.extend(["--habitat_data_dir", args.habitat_data_dir])
    if args.vh_exec_file:
        command.extend(["--vh_exec_file", args.vh_exec_file])
    if args.agent_service_python:
        command.extend(["--agent_service_python", args.agent_service_python])
    if args.no_action_success:
        command.append("--no_action_success")
    if args.no_bbox_coord:
        command.append("--no_bbox_coord")
    return command


def _start_agent_service(args: Namespace) -> subprocess.Popen[bytes]:
    from Agent.remote.lifecycle import choose_agent_port, popen_agent_service

    args.agent_port = choose_agent_port(
        args.agent_host, args.agent_port, args.agent_authkey.encode("utf-8")
    )
    return popen_agent_service(
        args.agent_service_python or sys.executable,
        args.agent_host,
        args.agent_port,
        args.agent_authkey,
        cwd=str(REPOSITORY_ROOT),
    )


def _wait_for_agent(args: Namespace) -> None:
    from Agent.remote.lifecycle import connect_agent
    from Agent.remote.protocol import METHOD_PING, make_request

    address = (args.agent_host, int(args.agent_port))
    authkey = args.agent_authkey.encode("utf-8")
    last_error: Exception | None = None
    for _ in range(args.agent_connect_retries):
        try:
            connection = connect_agent(
                address[0], address[1], authkey, args.agent_retry_interval_s
            )
            try:
                connection.send(make_request(METHOD_PING, {}))
                connection.recv()
            finally:
                connection.close()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(args.agent_retry_interval_s)
    raise RuntimeError(
        f"Agent service at {address[0]}:{address[1]} did not accept a connection: {last_error}"
    )


def _stop_agent_service(
    args: Namespace, process: subprocess.Popen[bytes] | None
) -> None:
    if process is None:
        return
    from Agent.remote.lifecycle import shutdown_owned_service

    shutdown_owned_service(
        args.agent_host,
        int(args.agent_port),
        args.agent_authkey.encode("utf-8"),
        process,
    )


def _run_behavior_task(args: Namespace, task_index: int) -> tuple[list[float], int]:
    print(f"========== BEHAVIOR task {task_index} ==========", flush=True)
    environment = os.environ.copy()
    environment["OMNIGIBSON_HEADLESS"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        _behavior_child_command(args, task_index),
        cwd=os.getcwd(),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    scores: list[float] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        score = _score_from_line(line)
        if score is not None:
            scores.append(score)
    return_code = process.wait()
    if return_code != 0:
        print(
            f"[BEHAVIOR batch] task {task_index} failed (exit {return_code})",
            flush=True,
        )
    if not scores:
        print(
            f"[BEHAVIOR batch] task {task_index} produced no score",
            flush=True,
        )
    return scores, return_code


def _run_behavior_processes(args: Namespace, dataset: list[Any]) -> None:
    task_indices = _selected_task_indices(dataset, args)
    split_stats = _empty_split_stats()
    task_episode_ids: dict[int, list[int | str]] = {}
    for task_index in task_indices:
        task_item = dataset[task_index]
        scenes = task_item.get("sampled_scenes") or []
        if len(scenes) != 1:
            print(
                f"[BEHAVIOR batch] task {task_index} has {len(scenes)} sampled scenes; "
                "episode IDs will use 'task_index:scene_index'.",
                flush=True,
            )
        split = _task_split(args, task_item, task_index)
        episode_ids: list[int | str] = []
        for scene_index in range(len(scenes)):
            _record_expected_episode(split_stats, split)
            episode_ids.append(
                task_index if len(scenes) == 1 else f"{task_index}:{scene_index}"
            )
        task_episode_ids[task_index] = episode_ids

    agent_process = None
    if args.agent_mode == "remote" and args.agent_auto_start_service:
        agent_process = _start_agent_service(args)
        _wait_for_agent(args)

    failed: list[int] = []
    no_score_episode_ids: list[int | str] = []
    try:
        for task_index in task_indices:
            if (
                agent_process is not None
                and agent_process.poll() is not None
            ):
                print(
                    "[Agent] Service exited during the previous task; restarting it.",
                    flush=True,
                )
                agent_process = _start_agent_service(args)
                _wait_for_agent(args)
            scores, return_code = _run_behavior_task(args, task_index)
            task_item = dataset[task_index]
            split = _task_split(args, task_item, task_index)
            episode_ids = task_episode_ids[task_index]
            if len(scores) > len(episode_ids):
                raise RuntimeError(
                    f"BEHAVIOR task {task_index} produced {len(scores)} "
                    f"scores for {len(episode_ids)} episodes."
                )
            for score in scores:
                _record_score(split_stats, split, score)
            missing_episode_ids = episode_ids[len(scores) :]
            no_score_episode_ids.extend(missing_episode_ids)
            if return_code != 0 or missing_episode_ids:
                failed.append(task_index)
    finally:
        _stop_agent_service(args, agent_process)

    _print_evaluation_summary(
        args,
        dataset,
        split_stats,
        behavior_no_score_episode_ids=no_score_episode_ids,
    )

    if failed:
        print(f"Failed tasks: {failed}")
        raise SystemExit(1)


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    dataset = _load_dataset(args)
    if args.simulator_name == "BEHAVIOR" and not args.behavior_worker:
        _run_behavior_processes(args, dataset)
        return
    if args.simulator_name == "BEHAVIOR":
        os.environ["OMNIGIBSON_HEADLESS"] = "1"

    from runner import Runner

    _seed_everything(args.seed)

    proc_data = (
        _load_procthor_dataset(args.procthor_dataset_dir)
        if args.simulator_name == "THOR"
        else None
    )
    comm = _create_vh_comm(args) if args.simulator_name == "VH" else None
    dataset_name = Path(args.dataset_file).stem
    output_root = Path(args.agent_save_dir).resolve() / dataset_name
    bootstrap_dir = output_root / "_bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    agent = _build_agent(args, str(bootstrap_dir))
    split_stats = _empty_split_stats()

    try:
        for task_index, task_item in enumerate(dataset):
            if task_index < args.start:
                continue
            if task_index >= args.end:
                break
            print(f"============== Task {task_index} ==============")

            instruction = _instruction_text(task_item)
            print(f"Instruction: {instruction}")
            logic_dict = copy.deepcopy(task_item["logic_dict"])
            split = _task_split(args, task_item, task_index)

            for scene_index, scene in enumerate(task_item["sampled_scenes"]):
                _record_expected_episode(split_stats, split)
                print(
                    f"---------- Scene {scene_index} ({_scene_label(scene)}) ----------"
                )
                save_dir = output_root / str(task_index) / str(scene_index)
                save_dir.mkdir(parents=True, exist_ok=True)
                retry_limit = (
                    args.thor_timeout_retries if args.simulator_name == "THOR" else 0
                )
                timeout_count = 0
                scene_done = False

                while timeout_count <= retry_limit and not scene_done:
                    if timeout_count:
                        print(
                            f"[THOR] Retrying task {task_index}, scene {scene_index} "
                            f"({timeout_count}/{retry_limit})"
                        )
                        _cleanup_scene_dir_for_retry(save_dir)
                        time.sleep(args.thor_restart_wait_sec)

                    scene_seed = args.seed + task_index * 10_000 + scene_index
                    _seed_everything(scene_seed)
                    agent.set_save_dir(str(save_dir))
                    metadata = _scene_metadata(args, scene, proc_data, comm)
                    metadata["object_reference"] = copy.deepcopy(
                        metadata.get("object_reference", {})
                    )
                    task = {
                        "instruction": instruction,
                        "env_metadata": metadata,
                        "init_info_for_agent": {
                            "simulator_name": args.simulator_name,
                            "nav_action_mode": args.nav_action_mode,
                            "manip_action_mode": args.manip_action_mode,
                            "use_feedback": True,
                            "use_bbox_coord": not args.no_bbox_coord,
                            "use_action_success": not args.no_action_success,
                        },
                        "goal_info": {"logic_dict": logic_dict},
                    }

                    runner = None
                    try:
                        runner = Runner(task, agent, break_when_action_fail=False)
                        score, env_steps, agent_stats = runner.run()
                        print(f"Score: {score}, environment steps: {env_steps}")
                        if agent_stats:
                            print(f"Agent stats: {agent_stats}")
                        _record_score(split_stats, split, float(score))
                        scene_done = True
                    except Exception as exc:
                        if args.simulator_name == "THOR" and _is_ai2thor_timeout_error(
                            exc
                        ):
                            timeout_count += 1
                            print(
                                f"[THOR] Timeout in task {task_index}, scene {scene_index}: {exc}"
                            )
                            if timeout_count > retry_limit:
                                raise RuntimeError(
                                    "AI2-THOR timed out after "
                                    f"{retry_limit + 1} attempts in task {task_index}, "
                                    f"scene {scene_index}; aborting evaluation."
                                ) from exc
                            continue
                        raise
                    finally:
                        if runner is not None:
                            runner.close(close_agent=False)
                    sys.stdout.flush()
    finally:
        try:
            agent.close()
        finally:
            if comm is not None:
                comm.close()
            if not args.behavior_worker:
                _print_evaluation_summary(args, dataset, split_stats)


if __name__ == "__main__":
    main()
