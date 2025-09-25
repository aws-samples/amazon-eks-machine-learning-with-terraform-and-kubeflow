import time
from kubernetes import client, config

# Load Kubernetes configuration
config.load_kube_config()
v1 = client.CoreV1Api()
custom_api = client.CustomObjectsApi()

def find_matching_helm_pods(release_name:str, 
                            namespace:str='kubeflow-user-example-com') -> list:
    """Find pods managed by a specific Helm release"""
    helm_pods = v1.list_namespaced_pod(
        namespace=namespace
    )

    matching_pods = []
    for pod in helm_pods.items:
        if (pod.metadata.annotations and
            pod.metadata.annotations.get('app.kubernetes.io/managed-by') == 'Helm' and 
            pod.metadata.annotations.get('app.kubernetes.io/instance') == release_name):
            matching_pods.append(pod)

    return matching_pods

def wait_for_helm_release_pods(release_name:str, 
                               namespace:str='kubeflow-user-example-com', 
                               timeout:float=3600):
    """Wait for all pods in a helm release to complete successfully"""
    print(f"Waiting for pods in release '{release_name}' to complete...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            matching_pods = find_matching_helm_pods(release_name, namespace)
            
            if not matching_pods:
                print(f"No pods found in Hem release: {release_name} waiting...")
                time.sleep(60)
                continue
            
            all_completed = True
            for pod in matching_pods:
                status = pod.status.phase
                print(f"Pod {pod.metadata.name}: {status}")
                
                if status in ['Pending', 'Running']:
                    all_completed = False
                elif status == 'Failed':
                    print(f"Pod {pod.metadata.name} failed!")
                    return False
            
            if all_completed:
                print("All pods completed successfully!")
                return True
                
        except Exception as e:
            print(f"Error checking pods: {e}")
        
        time.sleep(60)
    
    print(f"Timeout waiting for pods to complete")
    return False
    
def is_application_healthy(name:str, namespace: str) -> bool:
    try:
        # Get the RayService object
        rayservice = custom_api.get_namespaced_custom_object(
            group="ray.io",
            version="v1",
            namespace=namespace,
            plural="rayservices",
            name=name
        )
        
        status = rayservice.get("status", {})
        
        # Check overall service status
        service_status = status.get("serviceStatus")
        if service_status != "Running":
            return False
        
        # Check Ready condition
        conditions = status.get("conditions", [])
        ready_condition = next((c for c in conditions if c.get("type") == "Ready"), None)
        if not ready_condition or ready_condition.get("status") != "True":
            return False
        
        # Check application statuses
        active_service = status.get("activeServiceStatus", {})
        app_statuses = active_service.get("applicationStatuses", {})
        
        for _, app_status in app_statuses.items():
            if app_status.get("status") != "RUNNING":
                return False
            
            # Check deployment statuses within each application
            deployments = app_status.get("serveDeploymentStatuses", {})
            for _, deployment_status in deployments.items():
                if deployment_status.get("status") != "HEALTHY":
                    return False
        
        return True
        
    except Exception as e:
        print(f"Error checking Ray Serve application status: {e}")
        return False
    
def wait_for_rayservice_ready(release_name:str, 
                              namespace:str='kubeflow-user-example-com', 
                              timeout:float=1800) -> bool:
    """Wait for RayService to be ready and healthy"""
    print(f"Waiting for RayService '{release_name}' to be ready...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            # Check RayService status
            rayservices = custom_api.list_namespaced_custom_object(
                group="ray.io",
                version="v1",
                namespace=namespace,
                plural="rayservices"
            )
            
            matching_rayservice = None
            for rs in rayservices['items']:
                if (rs.get('metadata', {}).get('labels', {}).get('app.kubernetes.io/instance') == release_name):
                    matching_rayservice = rs
                    break
            
            if not matching_rayservice:
                print(f"No RayService found for release: {release_name}, waiting...")
                time.sleep(60)
                continue
            
            rayservice_name = matching_rayservice['metadata']['name']
            status = matching_rayservice.get('status', {})
            service_status = status.get('serviceStatus', 'Unknown')
            
            print(f"RayService {rayservice_name}: {service_status}")
            
            # Check if Ray Serve application is running and healthy
            if service_status.lower() == 'running':
                app_healthy = False
                while not (app_healthy := is_application_healthy(name=rayservice_name, namespace=namespace)):
                    print("Waiting for Ray Serve application to be Ready")
                    if (time.time() - start_time) > timeout:
                        break
                    time.sleep(60)
                    continue
                
                if app_healthy:
                    print(f"RayService {rayservice_name} in namespace {namespace} is Ready!")
                    return True

        except Exception as e:
            print(f"Error checking RayService: {e}")
        
        time.sleep(60)
    
    print(f"Timeout waiting for RayService to be Running and Healthy")
    return False

def find_k8s_service(service_name:str, 
                     namespace:str='kubeflow-user-example-com') -> str:

    target_service = ""
    try:
        target_service = v1.read_namespaced_service(
            name=service_name,
            namespace=namespace
        )
    except client.exceptions.ApiException as e:
        if e.status != 404:
            raise
    
    return target_service

def find_matching_helm_services(release_name, namespace='kubeflow-user-example-com'):
    """Find services managed by a specific Helm release"""
    helm_services = v1.list_namespaced_service(
        namespace=namespace
    )

    matching_services = []
    for service in helm_services.items:
        if (service.metadata.annotations and
            service.metadata.annotations.get('app.kubernetes.io/managed-by') == 'Helm' and
            service.metadata.annotations.get('app.kubernetes.io/instance') == release_name):
            matching_services.append(service)

    return matching_services

# Wait for Triton server to be ready
def wait_for_triton_server(release_name, namespace='kubeflow-user-example-com', timeout=1800):
    """Wait for Triton server pods to be running and ready"""
    print(f"Waiting for Triton server '{release_name}' to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            matching_pods = find_matching_helm_pods(release_name, namespace)
            
            if not matching_pods:
                print(f"No pods found in Hem release: {release_name} waiting...")
                time.sleep(60)
                continue
            
            all_ready = True
            for pod in matching_pods:
                status = pod.status.phase
                ready = all(condition.status == 'True' for condition in pod.status.conditions if condition.type == 'Ready')
                print(f"Pod {pod.metadata.name}: {status}, Ready: {ready}")
                
                if status != 'Running' or not ready:
                    all_ready = False
                    break
            
            if all_ready:
                print("Triton server is ready!")
                return True
                
        except Exception as e:
            print(f"Error checking pods: {e}")
        
        time.sleep(60)
    
    print(f"Timeout waiting for Triton server to be ready")
    return False