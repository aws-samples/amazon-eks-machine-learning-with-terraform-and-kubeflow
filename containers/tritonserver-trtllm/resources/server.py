# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import signal
import subprocess
import sys
import time

ERROR_EXIT_DELAY = 15
ERROR_CODE_FATAL = 255
EXIT_SUCCESS = 0

DELAY_BETWEEN_QUERIES = 60

def die(exit_code: int):
    if exit_code is None:
        exit_code = ERROR_CODE_FATAL

    write_error(f"Waiting {ERROR_EXIT_DELAY} second before exiting.")
    # Delay the process' termination to provide a small window for administrators 
    # to capture the logs before it exits and restarts.
    time.sleep(ERROR_EXIT_DELAY)

    exit(exit_code)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["leader", "worker"])
    parser.add_argument("--model", help="Model name", type=str, default="model")
    parser.add_argument("--model-repo", help="Model repository path", type=str, default="model_repo")
    parser.add_argument(
        "--dt",
        type=str,
        default="float16",
        choices=["bfloat16", "float16", "float32"],
        help="Tensor type.",
    )
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism.")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism.")
    parser.add_argument("--iso8601", action='store_true', type=bool, default=False)
    parser.add_argument("--verbose", action='store_true', type=bool, default=False)
    parser.add_argument(
        "--deployment", type=str, help="Name of the Kubernetes deployment."
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Namespace of the Kubernetes deployment.",
    )
    parser.add_argument("--multinode", action="count", default=0)

    return parser.parse_args()

def run_command(cmd_args: [str], omit_args: [int] = None):
    command = ""

    for i, arg in enumerate(cmd_args):
        command += " "
        if omit_args is not None and i in omit_args:
            command += "*****"
        else:
            command += arg

    write_output(f">{command}")
    write_output(" ")

    return subprocess.call(cmd_args, stderr=sys.stderr, stdout=sys.stdout)


def signal_handler(sig, frame):
    write_output(f"Signal {sig} detected, quitting.")
    exit(EXIT_SUCCESS)

def wait_for_workers(world_size: int, timeout: int=1800):
    if world_size is None or world_size <= 0:
        raise RuntimeError("Argument `world_size` must be greater than zero.")

    write_output("Begin waiting for worker pods.")

    cmd_args = [
        "kubectl",
        "get",
        "pods",
        "-n",
        f"{args.namespace}",
        "-l",
        f"app={args.deployment}",
        "-o",
        "jsonpath='{.items[*].metadata.name}'",
    ]
    command = " ".join(cmd_args)

    workers = []
    start_time = time.time()
    while len(workers) < world_size and ( time.time() - start_time) < timeout:
        time.sleep(DELAY_BETWEEN_QUERIES)
        write_output(f"> {command}")
        output = subprocess.check_output(cmd_args).decode("utf-8")
        write_output(output)
        output = output.strip("'")
        workers = output.split(" ")
        write_output(f"{len(workers)} workers of {world_size} workers ready in {time.time() - start_time} seconds")


    if workers is not None and len(workers) == world_size:
        write_output(f"All {world_size} workers are ready in {time.time() - start_time} seconds")
        workers.sort()

    return workers


def write_output(message: str):
    print(message, file=sys.stdout, flush=True)


def write_error(message: str):
    print(message, file=sys.stderr, flush=True)


def do_leader(args):
    world_size = args.tp * args.pp

    if world_size <= 0:
        raise Exception(
            "usage: Options --pp and --pp must both be equal to or greater than 1."
        )

    write_output(f"Executing Leader (world size: {world_size})")

    workers = wait_for_workers(world_size)

    if len(workers) != world_size:
        write_error(f"fatal: {len(workers)} found, expected {world_size}.")
        die(ERROR_EXIT_DELAY)

    cmd_args = [
        "mpirun",
        "--allow-run-as-root",
    ]

    cmd_args += [
        "--report-bindings",
        "-mca",
        "plm_rsh_agent",
        "kubessh",
        "-np",
        f"{world_size}",
        "--host",
        ",".join(workers),
    ]

    # Add per node command lines separated by ':'.
    for i in range(world_size):
        if i != 0:
            cmd_args += [":"]

        cmd_args += [
            "-n",
            "1",
            "tritonserver",
            "--allow-cpu-metrics=false",
            "--allow-gpu-metrics=false",
            "--disable-auto-complete-config",
            f"--id=rank{i}",
            "--model-load-thread-count=2",
            f"--model-repository={args.model_repo}",
        ]

        # Rank0 node needs to support metrics collection and web services.
        if i == 0:
            cmd_args += [
                "--allow-metrics=true",
                "--metrics-interval-ms=1000",
            ]

            if args.verbose:
                cmd_args += ["--log-verbose=1"]

            if args.iso8601:
                cmd_args += ["--log-format=ISO8601"]

        # Rank(N) nodes can disable metrics, web services, and logging.
        else:
            cmd_args += [
                "--allow-http=false",
                "--allow-grpc=false",
                "--allow-metrics=false",
                "--model-control-mode=explicit",
                f"--load-model={args.model}",
                "--log-info=false",
                "--log-warning=false",
            ]

    result = run_command(cmd_args)

    if result != 0:
        die(result)

    exit(result)


def do_worker(args):
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    write_output("Worker paused awaiting SIGINT or SIGTERM.")
    signal.pause()


# Parse options provided.
args = parse_arguments()

if args.mode == "leader":
    do_leader(args)
elif args.mode == "worker":
    do_worker(args)
