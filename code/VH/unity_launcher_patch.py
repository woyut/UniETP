# Portions of this file are adapted from VirtualHome
# (https://github.com/xavierpuigf/virtualhome), in particular from
# virtualhome/simulation/unity_simulator/communication.py (UnityLauncher).
# Copyright (c) 2022 MIT. Licensed under the MIT License.
# The upstream code is itself based on Unity ML-Agents
# (https://github.com/Unity-Technologies/ml-agents) and the AI2-THOR controller
# (https://github.com/allenai/ai2thor), both licensed under the Apache License, Version 2.0.
# The adapted code has been modified for this project.

"""
Monkey-patch VirtualHome UnityLauncher to:
1. Run the Unity executable under xvfb when on Linux with no_graphics=False and no valid DISPLAY.
2. Redirect Unity -logFile to a configurable path (disabled by default).
3. close(): on Linux kill the whole process group (xvfb-run + linux_exec), not just the direct child.
   Safety: only our Popen child and its descendants are in that group (start_new_session=True).
   We call killpg only while the process is still alive (proc.poll() is None) to avoid PID reuse risk.

Apply by importing this module before any code that creates UnityCommunication, e.g. in vh_env.py:
    import VH.unity_launcher_patch  # noqa: F401
"""

import os
import atexit
import signal
from sys import platform
import subprocess
import glob

# xvfb-run args: -a auto display, -s server-args
XVFB_RUN_ARGS = ["xvfb-run", "-a", "-s", "-screen 0 1024x768x24"]


def _resolve_unity_log_path(work_dir, port_number):
    """Resolve Unity log path from env, with sensible fallbacks."""
    log_file = os.environ.get("VH_UNITY_LOG_FILE", "").strip()
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        return log_file

    log_dir = os.environ.get("VH_UNITY_LOG_DIR", "").strip()
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, f"VHLOG_{port_number}.txt")

    return os.devnull


def _patched_launch_executable(
    self,
    file_name,
    x_display=None,
    no_graphics=False,
    docker_enabled=False,
    logging=False,
    args=None,
):
    if args is None:
        args = []
    if docker_enabled:
        return
    cwd = os.getcwd()
    file_name = (
        file_name.strip()
        .replace(".app", "")
        .replace(".exe", "")
        .replace(".x86_64", "")
        .replace(".x86", "")
    )
    env = {}
    true_filename = os.path.basename(os.path.normpath(file_name))
    launch_string = None
    if platform == "linux" or platform == "linux2":
        if not docker_enabled:
            if x_display:
                env["DISPLAY"] = ":" + x_display
                self.check_x_display(env["DISPLAY"])
            elif "DISPLAY" not in os.environ or not os.environ.get("DISPLAY"):
                env["DISPLAY"] = ""

            self.check_port(self.port_number)

            candidates = glob.glob(os.path.join(cwd, file_name) + ".x86_64")
            if len(candidates) == 0:
                candidates = glob.glob(os.path.join(cwd, file_name) + ".x86")
            if len(candidates) == 0:
                candidates = glob.glob(file_name + ".x86_64")
            if len(candidates) == 0:
                candidates = glob.glob(file_name + ".x86")
            if len(candidates) > 0:
                launch_string = candidates[0]

    elif platform == "darwin":
        candidates = glob.glob(
            os.path.join(cwd, file_name + ".app", "Contents", "MacOS", true_filename)
        )
        if len(candidates) == 0:
            candidates = glob.glob(
                os.path.join(file_name + ".app", "Contents", "MacOS", true_filename)
            )
        if len(candidates) == 0:
            candidates = glob.glob(
                os.path.join(cwd, file_name + ".app", "Contents", "MacOS", "*")
            )
        if len(candidates) == 0:
            candidates = glob.glob(
                os.path.join(file_name + ".app", "Contents", "MacOS", "*")
            )
        if len(candidates) > 0:
            launch_string = candidates[0]

    elif platform == "windows" or platform == "win32":
        candidates = glob.glob(os.path.join(cwd, file_name) + ".exe")
        if len(candidates) > 0:
            launch_string = candidates[0]

    if launch_string is None:
        self.close()
        raise Exception(
            "Couldn't launch the {0} environment. "
            "Provided filename does not match any environments.".format(true_filename)
        )
    docker_training = False
    if not docker_training:
        subprocess_args = [launch_string]
        if self.batchmode:
            subprocess_args += ["-batchmode"]
        if no_graphics:
            subprocess_args += ["-nographics"]

        file_path = os.getcwd()
        # Patch: log to configurable path instead of Unity default Player.log
        unity_log_path = _resolve_unity_log_path(file_path, self.port_number)
        subprocess_args += [
            "-http-port=" + str(self.port_number),
            "-logFile",
            unity_log_path,
        ]
        subprocess_args += args

        # Patch: on Linux with graphics, run under xvfb when there is no valid DISPLAY
        use_xvfb = (
            (platform == "linux" or platform == "linux2")
            and not no_graphics
            and (not x_display or not x_display.strip())
            and (
                not os.environ.get("DISPLAY")
                or os.environ.get("DISPLAY", "").strip() == ""
            )
        )
        if use_xvfb:
            subprocess_args = XVFB_RUN_ARGS + subprocess_args
            # Give xvfb-run a full env so it can find xvfb; child Unity will get DISPLAY from xvfb-run
            env = os.environ.copy()

        if logging:
            f = open("{}/port_{}.txt".format(file_path, self.port_number), "w+")
        else:
            f = subprocess.DEVNULL
        try:
            self.proc = subprocess.Popen(
                subprocess_args,
                env=env,
                stdout=f,
                start_new_session=True,
            )
            atexit.register(lambda: self.close())
        except Exception:
            raise Exception("Error, environment was found but could not be launched")
    else:
        docker_ls = (
            f"exec xvfb-run --auto-servernum --server-args='-screen 0 640x480x24'"
            f" {launch_string} -http-port {self.port_number}"
        )
        self.proc = subprocess.Popen(
            docker_ls,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )
        raise Exception("Docker training is still not implemented")


def _patched_close(self):
    """Kill the launcher process and on Linux the whole process group (xvfb-run + linux_exec)."""
    if self.proc is None:
        return
    proc = self.proc
    self.proc = None  # clear first so we don't double-close
    try:
        # Only use killpg while our process is still alive. If it already exited,
        # the PID may have been reused by an unrelated process - do not signal by pgid.
        if proc.poll() is None and (platform == "linux" or platform == "linux2"):
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError, AttributeError):
                pass
        proc.kill()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
    except ProcessLookupError:
        pass
    except Exception:
        pass


def apply_patch():
    """Apply monkey-patch to UnityLauncher.launch_executable and close()."""
    from virtualhome.simulation.unity_simulator import communication

    communication.UnityLauncher.launch_executable = _patched_launch_executable
    communication.UnityLauncher.close = _patched_close


# Apply as soon as this module is imported (must run before UnityCommunication is created)
apply_patch()
