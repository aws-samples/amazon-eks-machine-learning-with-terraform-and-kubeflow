from kfp import dsl
from kfp import compiler

from typing import List, Dict

@dsl.component(base_image='public.ecr.aws/w0d9j2p3/eks/universal-client:1.0.0')
def helm_charts_component(chart_configs: List[Dict]) -> str:

    import os
    import signal
    import sys
    from tempfile import NamedTemporaryFile
    import time
    from pyhelm import repo
    import yaml
    import subprocess
    from kubernetes import client, config
    from kubernetes.client.exceptions import ApiException

    class HelmChartHandler:

        def __init__(self, chart_config: dict) -> str:
            self.chart_config = chart_config

            signal.signal(signal.SIGINT, self.__exit_gracefully)
            signal.signal(signal.SIGTERM, self.__exit_gracefully)
            signal.signal(signal.SIGABRT, self.__exit_gracefully)

        def __call__(self) -> int:
            exit_code = 0
            try:
                exit_code = self.install_chart()
                if exit_code == 0:
                    self.wait_for_pods()
            except Exception as e:
                print(str(e))
            finally:
                exit_code = self.uninstall_release()
            
            return exit_code
        
        def __exit_gracefully(self, signum, frame):
            self.uninstall_release()
            sys.exit(f"Signal: {signum}")

        @staticmethod
        def run_cmd(cmd:str) -> int:

            print(f"run command: {cmd}")
            exit_code = 0
            try:
                completed_process = subprocess.run(cmd,capture_output=True,check=True)
                if completed_process.stdout:
                    print(completed_process.stdout)
                if completed_process.stderr:
                    print(completed_process.stderr)
                exit_code = completed_process.returncode
            except subprocess.CalledProcessError as e:
                exit_code = e.returncode
                print(str(e))
            
            return exit_code

        def install_chart(self) -> int:

            print(f"Install chart: {self.chart_config}...")

            release_name = self.chart_config['release_name']
            namespace = self.chart_config.get('namespace', 'default')
            repo_url = self.chart_config['repo_url']

            path = self.chart_config.get('path', None)
            if path is not None:
                branch = self.chart_config.get('branch', 'master')
                local_chart_path = repo.git_clone(repo_url=repo_url, branch=branch, path=path)
            else:
                chart = self.chart_config['chart']
                version = self.chart_config.get('version', None)
                local_chart_path = repo.from_repo(repo_url, chart, version=version)

           
            cmd = ["helm",
                    "install",
                    "--debug",
                    release_name, 
                    local_chart_path, 
                    "--namespace",
                    namespace,
                    "--wait"
                ]
            
            timeout = self.chart_config.get('timeout', None)
            if timeout is not None:
                cmd += [ "--timeout", timeout]
            
            values_file = os.path.join(local_chart_path, "values.yaml")
            if os.path.exists(values_file):
                cmd += [ "--values", values_file]

            values = self.chart_config.get('values', None)
            if values:
                with NamedTemporaryFile(mode="w+", encoding="utf-8", \
                                        prefix="values", suffix=".yaml", delete=False) as values_files:
                    yaml.dump(values, values_files, default_flow_style=False)
                    values_files.close()

                    cmd += [ "--values", values_files.name]

            exit_code = self.run_cmd(cmd=cmd)
            if exit_code == 0:
                print(f"Release {release_name} successful")
            else:
                print(f"Release {release_name} failed")

            return exit_code

        def uninstall_release(self) -> int:
            release_name = self.chart_config['release_name']
            namespace = self.chart_config.get('namespace', 'default')

            print(f"Uninstall release: {release_name} in {namespace}")

            cmd = [
                    "helm",
                    "uninstall",
                    release_name,
                    "--namespace",
                    namespace
                ]
            
            exit_code = self.run_cmd(cmd=cmd)
            if exit_code == 0:
                print(f"Uninstall release: {release_name} successful")
            else:
                print(f"Uninstall release: {release_name} failed")

            return exit_code

        def wait_for_pods(self):
            
            release_name = self.chart_config['release_name']
            namespace = self.chart_config.get('namespace', 'default')

            print(f"Wait for pods in release {release_name} in namespace {namespace}")

            complete_timeout = self.chart_config.get('pod_complete_timeout', 7*24*3600)
            error_timeout = self.chart_config.get('pod_error_timeout', 1800)

            config.load_incluster_config() # config.load_kube_config() 
            v1 = client.CoreV1Api()

            max_retries = 5
            attempt = 0
            wait_pods = []
            while not wait_pods and (attempt < max_retries):
                attempt += 1

                time.sleep(60) # let pods start
                pods = v1.list_namespaced_pod(namespace=namespace, async_req=False, timeout_seconds=120)
            
                for pod in pods.items:
                    annotations = pod.metadata.annotations

                    managed_by = annotations.get("app.kubernetes.io/managed-by", None)
                    if managed_by == 'Helm':
                        release_name = annotations.get("app.kubernetes.io/instance", None)
                        if release_name == release_name:
                            wait_pods.append(pod.metadata.name)
            
            if not wait_pods:
                print("No pods to wait for...")
            else:
                print(f"Waiting for pods to complete: {wait_pods}")

            start = time.time()
            pod_check_secs = self.chart_config.get('pod_check_secs', 300)
        
            for name in wait_pods:
                phase = "Running"
                while (phase == "Running" and (time.time() - start) < complete_timeout) or \
                    (phase != "Completed" and (time.time() - start) < error_timeout):

                    try:
                        print(f"read pod status: {name}")
                        pod = v1.read_namespaced_pod_status(name=name, namespace=namespace, 
                                                            async_req=False, _request_timeout=120)
                        phase = pod.status.phase
                        print(f"Pod {name} phase: {phase}")
                    except ApiException as e:
                        if e.reason == 'Not Found':
                            print(f"Pod {name} Not Found")
                            phase = "Deleted"
                    
                    if phase == "Running":
                        print(f"Check pod in: {pod_check_secs} secs")
                        time.sleep(pod_check_secs)
                    elif phase != "Deleted":
                        print(f"Check pod in: 60 secs")
                        time.sleep(60)

            print(f"Waiting for pod complete: {wait_pods}")

    for chart_config in chart_configs:
        helm_chart_handler = HelmChartHandler(chart_config)
        exit_code = helm_chart_handler()
        if exit_code > 0:
            return "Failure"
        
    return "Success"


compiler.Compiler().compile(helm_charts_component, package_path='kfp/components/packages/helm_charts_component.yaml')