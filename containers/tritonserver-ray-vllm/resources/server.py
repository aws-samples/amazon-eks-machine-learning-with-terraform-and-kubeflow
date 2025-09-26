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
import socket
import subprocess
import sys
import time

ERROR_EXIT_DELAY = 15
ERROR_CODE_FATAL = 255

DELAY_BETWEEN_QUERIES = 60
WAIT_WORKER_TIMEOUT = 1800

def die(exit_code: int):
    if exit_code is None:
        exit_code = ERROR_CODE_FATAL

    write_error(f"Error code: {exit_code}. Waiting {ERROR_EXIT_DELAY} second before exiting.")
    # Delay the process' termination to provide a small window for administrators 
    # to capture the logs before it exits and restarts.
    time.sleep(ERROR_EXIT_DELAY)

    exit(exit_code)


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model-repo", help="Model repository path", type=str, default="model_repo")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism.")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism.")
    parser.add_argument("--group-key", type=str, help="LWS Group Key")
    parser.add_argument("--namespace", type=str, default="default", help="Namespace of the Kubernetes deployment.")
    parser.add_argument("--head_port", type=str, default='6379', help="Default ray head port")
    parser.add_argument('--grpc_port', type=str, help='tritonserver grpc port', default='8001')
    parser.add_argument('--http_port', type=str, help='tritonserver http port', default='8000')
    parser.add_argument('--metrics_port', type=str, help='tritonserver metrics port', default='8002')
    parser.add_argument('--log-file', type=str, help='path to triton log file', default='triton_log.txt')
    parser.add_argument('--wait-worker-timeout', type=int, help='Wait for worker pods timeout', default=WAIT_WORKER_TIMEOUT)

    return parser.parse_args()

def detect_local_gpus():
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"]).decode("utf-8")
        write_output(output)
        return len(output.strip().split("\n"))
    except subprocess.CalledProcessError:
        return 0
    
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

def wait_for_workers(num_workers: int, timeout: int):
    write_output("Begin waiting for worker pods.")

    cmd_args = [
        "kubectl",
        "get",
        "pods",
        "-n",
        f"{args.namespace}",
        "-l",
        f"leaderworkerset.sigs.k8s.io/group-key={args.group_key}",
        "--field-selector",
        "status.phase=Running",
        "-o",
        "jsonpath='{.items[*].metadata.name}'",
    ]
    command = " ".join(cmd_args)

    workers = []
    start_time = time.time()
    while len(workers) < num_workers and ( time.time() - start_time) < timeout:
        time.sleep(DELAY_BETWEEN_QUERIES)
        write_output(f">{command}")
        output = subprocess.check_output(cmd_args).decode("utf-8")
        write_output(output)
        output = output.strip("'")
        workers = output.split(" ")
        write_output(f"{len(workers)} workers of {num_workers} workers ready in {time.time() - start_time} seconds")

    if workers is not None and len(workers) == num_workers:
        write_output(f"All {num_workers} workers are ready in {time.time() - start_time} seconds")
        workers.sort()

    return workers


def write_output(message: str):
    print(message, file=sys.stdout, flush=True)


def write_error(message: str):
    print(message, file=sys.stderr, flush=True)


def start_ray_cluster(args):
    world_size = args.tp * args.pp
    assert world_size > 0, f"world_size: {world_size} must be greater than zero, tp: {args.tp}, pp: {args.pp}"
    write_output(f"Executing Leader (world size: {world_size})")

    num_local_gpus = detect_local_gpus()
    num_workers = world_size // num_local_gpus

    assert num_workers > 0, f"num_workers: {num_workers} must be greater than zero, world_size: {world_size}, num_local_gpus: {num_local_gpus}"

    workers = wait_for_workers(num_workers=num_workers, timeout=args.wait_worker_timeout)
    workers_with_mpi_slots = [worker + f":1" for worker in workers]

    ip_addrs = socket.gethostbyname(f'{workers[0]}')
    write_output(f"head node ip_addrs: {ip_addrs}")
  
    if len(workers) != num_workers:
        write_error(f"fatal: {len(workers)} found, expected {num_workers}.")
        die(ERROR_EXIT_DELAY)

    cmd_args = [
        "mpirun",
        "--allow-run-as-root",
    ]

    cmd_args += [
        "--report-bindings",
        "-map-by",
        "slot",
        "-mca",
        "btl_tcp_if_exclude",
        "lo,docker0",
        "-mca",
        "oob_tcp_if_exclude",
        "lo,docker0",
        "-mca",
        "plm_rsh_agent",
        "kubessh",
        "-np",
        f"{num_workers}",
        "--host",
        ",".join(workers_with_mpi_slots),
    ]
  
    # Add per node command lines separated by ':'.
    for i in range(num_workers):

        if i != 0:
            cmd_args += [":"]

        cmd_args += [
            "-n",
            "1",
            "ray",
            "start"
        ]
        if i == 0:
            cmd_args += [
                f'--head', 
                f'--node-ip-address={ip_addrs}',
                f'--port={args.head_port}',
            ]
        else:
            cmd_args += [
                "--address",
                f"{ip_addrs}:{args.head_port}"
            ]

    result = run_command(cmd_args)

    if result != 0:
        die(result)

def start_tritonserver(args):

    cmd_args = [
        "tritonserver",
        f"--model-repository={args.model_repo}",
        f'--grpc-port={args.grpc_port}', 
        f'--http-port={args.http_port}',
        f'--metrics-port={args.metrics_port}',
    ]
    cmd_args += ['--log-verbose=3', f'--log-file={args.log_file}']
    cmd_args += [
        "--allow-metrics=true",
        "--metrics-interval-ms=1000",
    ]
    result = run_command(cmd_args)

    if result != 0:
        die(result)

    exit(result)

if __name__ == "__main__":
    args = parse_arguments()
    start_ray_cluster(args)
    start_tritonserver(args)

