# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Set of helper methods to manage Kuberay clusters.

Based on https://github.com/ray-project/kuberay/blob/master/clients/python-client/python_client/utils/kuberay_cluster_utils.py
"""

import copy
import logging
import re
import time
from typing import Any, Optional, Tuple

from kubernetes import client, config
from kubernetes.client.rest import ApiException

from nemo_run.core.execution.kuberay import KubeRayExecutor

logger = logging.getLogger(__name__)

# Group, Version, Plural
GROUP = "ray.io"
VERSION = "v1alpha1"
PLURAL = "rayclusters"
KIND = "RayCluster"
RUNAI_SCHEDULER = "runai-scheduler"


def populate_meta(
    cluster: dict,
    name: str,
    k8s_namespace: str,
    labels: dict,
    ray_version: str,
) -> dict[str, Any]:
    """Populate the metadata and ray version of the cluster.

    Parameters:
    - cluster (dict): A dictionary representing a cluster.
    - name (str): The name of the cluster.
    - k8s_namespace (str): The namespace of the cluster.
    - labels (dict): A dictionary of labels to be applied to the cluster.
    - ray_version (str): The version of Ray to use in the cluster.

    Returns:
        dict: The updated cluster dictionary with metadata and ray version populated.
    """
    assert is_valid_name(name), f"Invalid cluster name: {name}."

    cluster["apiVersion"] = "{group}/{version}".format(group=GROUP, version=VERSION)
    cluster["kind"] = KIND
    cluster["metadata"] = {
        "name": name,
        "namespace": k8s_namespace,
        "labels": labels,
    }
    cluster["spec"] = {"rayVersion": ray_version}
    return cluster


def populate_ray_head(
    cluster: dict,
    ray_image: str,
    service_type: str,
    cpu_requests: str,
    memory_requests: str,
    cpu_limits: str,
    memory_limits: str,
    ray_start_params: dict,
    head_ports: list[dict[str, Any]],
    volume_mounts: list[dict[str, Any]],
    volumes: list[dict[str, Any]],
    spec_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Populate the ray head specs of the cluster
    Parameters:
    - cluster (dict): The dictionary representation of the cluster.
    - ray_image (str): The name of the ray image to use for the head node.
    - service_type (str): The type of service to run for the head node.
    - cpu_requests (str): The CPU resource requests for the head node.
    - memory_requests (str): The memory resource requests for the head node.
    - cpu_limits (str): The CPU resource limits for the head node.
    - memory_limits (str): The memory resource limits for the head node.
    - ray_start_params (dict): The parameters for starting the Ray cluster.

    Returns:
    - Tuple (dict, bool): The updated cluster, and a boolean indicating whether the update was successful.
    """
    # make sure metadata exists
    if "spec" in cluster.keys():
        if "headGroupSpec" not in cluster.keys():
            logger.info(f"setting the headGroupSpec for cluster {cluster['metadata']['name']}")
            cluster["spec"]["headGroupSpec"] = []
    else:
        logger.error("error creating ray head, the spec and/or metadata is not define")
        return cluster

    if "dashboard-host" not in ray_start_params:
        ray_start_params["dashboard-host"] = "0.0.0.0"

    # populate headGroupSpec
    cluster["spec"]["headGroupSpec"] = {
        "serviceType": service_type,
        "rayStartParams": ray_start_params,
        "template": {
            "spec": {
                "containers": [
                    {
                        "image": ray_image,
                        "name": "ray-head",
                        "ports": head_ports,
                        "lifecycle": {
                            "preStop": {"exec": {"command": ["/bin/sh", "-c", "ray stop"]}}
                        },
                        "resources": {
                            "requests": {
                                "cpu": cpu_requests,
                                "memory": memory_requests,
                            },
                            "limits": {"cpu": cpu_limits, "memory": memory_limits},
                        },
                        "volumeMounts": volume_mounts,
                    }
                ],
                "volumes": volumes,
                **spec_kwargs,
            }
        },
    }

    return cluster


def populate_worker_group(
    cluster: dict,
    group_name: str,
    ray_image: str,
    ray_command: Any,
    gpus_per_worker: Optional[int],
    cpu_requests: Optional[str],
    memory_requests: Optional[str],
    cpu_limits: Optional[str],
    memory_limits: Optional[str],
    replicas: int,
    min_replicas: int,
    max_replicas: int,
    ray_start_params: dict,
    volume_mounts: list[dict[str, Any]],
    volumes: list[dict[str, Any]],
    labels: dict[str, Any],
    annotations: dict[str, Any],
    spec_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Populate the worker group specification in the cluster dictionary.

    Parameters:
    - group_name (str): The name of the worker group.
    - ray_image (str): The image to use for the Ray worker containers.
    - ray_command (Any): The command to run in the Ray worker containers.
    - cpu_requests (str): The requested CPU resources for the worker containers.
    - memory_requests (str): The requested memory resources for the worker containers.
    - cpu_limits (str): The limit on CPU resources for the worker containers.
    - memory_limits (str): The limit on memory resources for the worker containers.
    - replicas (int): The desired number of replicas for the worker group.
    - min_replicas (int): The minimum number of replicas for the worker group.
    - max_replicas (int): The maximum number of replicas for the worker group.
    - ray_start_params (dict): The parameters to pass to the Ray worker start command.

    Returns:
    - Tuple[dict, bool]: A tuple of the worker group specification and a boolean indicating
        whether the worker group was successfully populated.
    """
    assert is_valid_name(group_name)
    assert max_replicas >= min_replicas

    if "workerGroupSpecs" not in cluster["spec"].keys():
        cluster["spec"]["workerGroupSpecs"] = []

    resource_requests = {}
    resource_limits = {}
    if cpu_requests:
        resource_requests["cpu"] = cpu_requests
    if memory_requests:
        resource_requests["memory"] = memory_requests
    if cpu_limits:
        resource_limits["cpu"] = cpu_limits
    if memory_limits:
        resource_limits["memory"] = memory_limits

    if gpus_per_worker:
        resource_requests["nvidia.com/gpu"] = gpus_per_worker
        resource_limits["nvidia.com/gpu"] = gpus_per_worker

    worker_group: dict[str, Any] = {
        "groupName": group_name,
        "maxReplicas": max_replicas,
        "minReplicas": min_replicas,
        "rayStartParams": ray_start_params,
        "replicas": replicas,
        "template": {
            "spec": {
                "schedulerName": RUNAI_SCHEDULER,
                "containers": [
                    {
                        "image": ray_image,
                        "command": ray_command,
                        "lifecycle": {
                            "preStop": {"exec": {"command": ["/bin/sh", "-c", "ray stop"]}}
                        },
                        "name": "ray-worker",
                        "resources": {
                            "requests": resource_requests,
                            "limits": resource_limits,
                        },
                        "volumeMounts": volume_mounts,
                    }
                ],
                "volumes": volumes,
                **spec_kwargs,
            }
        },
    }

    if labels or annotations:
        metadata = {}
        if labels:
            metadata["labels"] = labels
        if annotations:
            metadata["annotations"] = annotations

        worker_group["metadata"] = metadata

    cluster["spec"]["workerGroupSpecs"].append(worker_group)

    return cluster


def update_worker_group_replicas(
    cluster: dict,
    group_name: str,
    max_replicas: int,
    min_replicas: int,
    replicas: int,
) -> Tuple[dict, bool]:
    """Update the number of replicas for a worker group in the cluster.

    Parameters:
    - cluster (dict): The cluster to update.
    - group_name (str): The name of the worker group to update.
    - max_replicas (int): The maximum number of replicas for the worker group.
    - min_replicas (int): The minimum number of replicas for the worker group.
    - replicas (int): The desired number of replicas for the worker group.

    Returns:
    Tuple[dict, bool]: A tuple containing the updated cluster and a flag indicating whether the update was successful.
    """
    assert cluster["spec"]["workerGroupSpecs"]
    assert max_replicas >= min_replicas

    for i in range(len(cluster["spec"]["workerGroupSpecs"])):
        if cluster["spec"]["workerGroupSpecs"][i]["groupName"] == group_name:
            cluster["spec"]["workerGroupSpecs"][i]["maxReplicas"] = max_replicas
            cluster["spec"]["workerGroupSpecs"][i]["minReplicas"] = min_replicas
            cluster["spec"]["workerGroupSpecs"][i]["replicas"] = replicas
            return cluster, True

    return cluster, False


def update_worker_group_resources(
    cluster: dict,
    group_name: str,
    cpu_requests: str,
    memory_requests: str,
    cpu_limits: str,
    memory_limits: str,
    container_name="unspecified",
) -> Tuple[dict, bool]:
    """Update the resources for a worker group pods in the cluster.

    Parameters:
    - cluster (dict): The cluster to update.
    - group_name (str): The name of the worker group to update.
    - cpu_requests (str): CPU requests for the worker pods.
    - memory_requests (str): Memory requests for the worker pods.
    - cpu_limits (str): CPU limits for the worker pods.
    - memory_limits (str): Memory limits for the worker pods.

    Returns:
    Tuple[dict, bool]: A tuple containing the updated cluster and a flag indicating whether the update was successful.
    """
    assert cluster["spec"]["workerGroupSpecs"]

    worker_groups = cluster["spec"]["workerGroupSpecs"]

    def add_values(group_index: int, container_index: int):
        worker_groups[group_index]["template"]["spec"]["containers"][container_index]["resources"][
            "requests"
        ]["cpu"] = cpu_requests
        worker_groups[group_index]["template"]["spec"]["containers"][container_index]["resources"][
            "requests"
        ]["memory"] = memory_requests
        worker_groups[group_index]["template"]["spec"]["containers"][container_index]["resources"][
            "limits"
        ]["cpu"] = cpu_limits
        worker_groups[group_index]["template"]["spec"]["containers"][container_index]["resources"][
            "limits"
        ]["memory"] = memory_limits

    for group_index, worker_group in enumerate(worker_groups):
        if worker_group["groupName"] != group_name:
            continue

        containers = worker_group["template"]["spec"]["containers"]
        container_names = [container["name"] for container in containers]

        if len(containers) == 0:
            logger.error(
                f"error updating container resources, the worker group {group_name} has no containers"
            )
            return cluster, False

        if container_name == "unspecified":
            add_values(group_index, 0)
            return cluster, True
        elif container_name == "all_containers":
            for container_index in range(len(containers)):
                add_values(group_index, container_index)
            return cluster, True
        elif container_name in container_names:
            container_index = container_names.index(container_name)
            add_values(group_index, container_index)
            return cluster, True

    return cluster, False


def duplicate_worker_group(
    cluster: dict,
    group_name: str,
    new_group_name: str,
) -> Tuple[dict, bool]:
    """Duplicate a worker group in the cluster.

    Parameters:
    - cluster (dict): The cluster definition.
    - group_name (str): The name of the worker group to be duplicated.
    - new_group_name (str): The name for the duplicated worker group.

    Returns:
    Tuple[dict, bool]: A tuple containing the updated cluster definition and a boolean indicating the success of the operation.
    """
    assert is_valid_name(new_group_name)
    assert cluster["spec"]["workerGroupSpecs"]

    worker_groups = cluster["spec"]["workerGroupSpecs"]
    for _, worker_group in enumerate(worker_groups):
        if worker_group["groupName"] == group_name:
            duplicate_group = copy.deepcopy(worker_group)
            duplicate_group["groupName"] = new_group_name
            worker_groups.append(duplicate_group)
            return cluster, True

    logger.error(f"error duplicating worker group, no match was found for {group_name}")
    return cluster, False


def delete_worker_group(
    cluster: dict,
    group_name: str,
) -> Tuple[dict, bool]:
    """Deletes a worker group in the cluster.

    Parameters:
    - cluster (dict): The cluster definition.
    - group_name (str): The name of the worker group to be duplicated.

    Returns:
    Tuple[dict, bool]: A tuple containing the updated cluster definition and a boolean indicating the success of the operation.
    """
    assert cluster["spec"]["workerGroupSpecs"]

    worker_groups = cluster["spec"]["workerGroupSpecs"]
    first_or_none = next((x for x in worker_groups if x["groupName"] == group_name), None)
    if first_or_none:
        worker_groups.remove(first_or_none)
        return cluster, True

    logger.error(f"error removing worker group, no match was found for {group_name}")
    return cluster, False


def is_valid_name(name: str) -> bool:
    msg = "The name must be 63 characters or less, begin and end with an alphanumeric character, and contain only dashes, dots, and alphanumerics."
    if len(name) > 63 or not bool(re.match("^[a-z0-9]([-.]*[a-z0-9])+$", name)):
        logger.info(msg)
        return False
    return True


def is_valid_label(name: str) -> bool:
    msg = "The label name must be 63 characters or less, begin and end with an alphanumeric character, and contain only dashes, underscores, dots, and alphanumerics."
    if len(name) > 63 or not bool(re.match("^[a-z0-9]([-._]*[a-z0-9])+$", name)):
        logger.error(msg)
        return False
    return True


class KubeRayCluster:
    """
    RayClusterApi provides APIs to list, get, create, build, update, delete rayclusters.

    Methods:
    - list_ray_clusters(k8s_namespace: str = "default", async_req: bool = False) -> Any:
    - get_ray_cluster(name: str, k8s_namespace: str = "default") -> Any:
    - create_ray_cluster(body: Any, k8s_namespace: str = "default") -> Any:
    - delete_ray_cluster(name: str, k8s_namespace: str = "default") -> bool:
    - patch_ray_cluster(name: str, ray_patch: Any, k8s_namespace: str = "default") -> Any:
    """

    EXECUTOR_CLS = KubeRayExecutor

    # initial config to setup the kube client
    def __init__(self):
        # loading the config
        self.kube_config: Optional[Any] = config.load_kube_config()
        self.api = client.CustomObjectsApi()
        self.core_v1_api = client.CoreV1Api()

    def list_ray_clusters(
        self, k8s_namespace: str = "default", label_selector: str = "", async_req: bool = False
    ) -> Any:
        logger.info(
            f"Listing Ray clusters in namespace: {k8s_namespace}, label_selector: {label_selector}, async_req: {async_req}"
        )
        """List Ray clusters in a given namespace.

        Parameters:
        - k8s_namespace (str, optional): The namespace in which to list the Ray clusters. Defaults to "default".
        - async_req (bool, optional): Whether to make the request asynchronously. Defaults to False.

        Returns:
            Any: The custom resource for Ray clusters in the specified namespace, or None if not found.

        Raises:
            ApiException: If there was an error fetching the custom resource.
        """
        try:
            resource: Any = self.api.list_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                plural=PLURAL,
                namespace=k8s_namespace,
                label_selector=label_selector,
                async_req=async_req,
            )
            if "items" in resource:
                return resource
            return None
        except ApiException as e:
            if e.status == 404:
                logger.error("raycluster resource is not found. error = {}".format(e))
                return None
            else:
                logger.error("error fetching custom resource: {}".format(e))
                return None

    def get_ray_cluster(self, name: str, k8s_namespace: str = "default") -> Any:
        logger.info(f"Getting Ray cluster: {name} in namespace: {k8s_namespace}")
        """Get a specific Ray cluster in a given namespace.

        Parameters:
        - name (str): The name of the Ray cluster custom resource. Defaults to "".
        - k8s_namespace (str, optional): The namespace in which to retrieve the Ray cluster. Defaults to "default".

        Returns:
            Any: The custom resource for the specified Ray cluster, or None if not found.

        Raises:
            ApiException: If there was an error fetching the custom resource.
        """
        try:
            resource: Any = self.api.get_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                plural=PLURAL,
                name=name,
                namespace=k8s_namespace,
            )
            return resource
        except ApiException as e:
            if e.status == 404:
                logger.error("raycluster resource is not found. error = {}".format(e))
                return None
            else:
                logger.error("error fetching custom resource: {}".format(e))
                return None

    def get_ray_cluster_status(
        self,
        name: str,
        k8s_namespace: str = "default",
        timeout: int = 60,
        delay_between_attempts: int = 5,
    ) -> Any:
        logger.info(
            f"Getting Ray cluster status: {name} in namespace: {k8s_namespace}, timeout: {timeout}s, delay: {delay_between_attempts}s"
        )
        """Get a specific Ray cluster in a given namespace.

        Parameters:
        - name (str): The name of the Ray cluster custom resource. Defaults to "".
        - k8s_namespace (str, optional): The namespace in which to retrieve the Ray cluster. Defaults to "default".
        - timeout (int, optional): The duration in seconds after which we stop trying to get status if still not set. Defaults to 60 seconds.
        - delay_between_attempts (int, optional): The duration in seconds to wait between attempts to get status if not set. Defaults to 5 seconds.

        Returns:
            Any: The custom resource status for the specified Ray cluster, or None if not found.

        Raises:
            ApiException: If there was an error fetching the custom resource.
        """
        while timeout > 0:
            try:
                resource: Any = self.api.get_namespaced_custom_object_status(
                    group=GROUP,
                    version=VERSION,
                    plural=PLURAL,
                    name=name,
                    namespace=k8s_namespace,
                )
            except ApiException as e:
                if e.status == 404:
                    logger.error("raycluster resource is not found. error = {}".format(e))
                    return None
                else:
                    logger.error("error fetching custom resource: {}".format(e))
                    return None

            if "status" in resource and resource["status"]:
                return resource["status"]
            else:
                logger.info("raycluster {} status not set yet, waiting...".format(name))
                time.sleep(delay_between_attempts)
                timeout -= delay_between_attempts

        logger.info("raycluster {} status not set yet, timing out...".format(name))
        return None

    def wait_until_ray_cluster_running(
        self,
        name: str,
        executor: KubeRayExecutor,
        k8s_namespace: Optional[str] = None,
        timeout: int = 60,
        delay_between_attempts: int = 5,
    ) -> bool:
        namespace = k8s_namespace or executor.namespace
        logger.info(
            f"Waiting until Ray cluster: {name} in namespace: {namespace} is running, timeout: {timeout}s, delay: {delay_between_attempts}s"
        )
        """Get a specific Ray cluster in a given namespace.

        Parameters:
        - name (str): The name of the Ray cluster custom resource. Defaults to "".
        - k8s_namespace (str, optional): The namespace in which to retrieve the Ray cluster. Defaults to "default".
        - timeout (int, optional): The duration in seconds after which we stop trying to get status. Defaults to 60 seconds.
        - delay_between_attempts (int, optional): The duration in seconds to wait between attempts to get status if not set. Defaults to 5 seconds.

        Returns:
            Bool: True if the raycluster status is Running, False otherwise.

        """
        while timeout > 0:
            status = self.get_ray_cluster_status(
                name, k8s_namespace or executor.namespace, timeout, delay_between_attempts
            )
            if not status:
                logger.info(f"Ray cluster {name} status could not be retrieved")
                return False

            # TODO: once we add State to Status, we should check for that as well  <if status and status["state"] == "Running":>
            if status and status["head"] and status["head"]["serviceIP"]:
                logger.info(f"Ray cluster {name} is running")
                return True

            logger.info(
                "raycluster {} status is not running yet, current status is {}".format(
                    name, status["state"] if status and "state" in status else "unknown"
                )
            )
            time.sleep(delay_between_attempts)
            timeout -= delay_between_attempts

        logger.info("raycluster {} status is not running yet, timing out...".format(name))
        return False

    def create_ray_cluster(
        self, name: str, executor: KubeRayExecutor, k8s_namespace: Optional[str] = None
    ) -> Any:
        namespace = k8s_namespace or executor.namespace
        logger.info(f"Creating Ray cluster: {name} in namespace: {namespace}")
        """Create a new Ray cluster custom resource.

        Parameters:
        - body (Any): The data of the custom resource to create.
        - k8s_namespace (str, optional): The namespace in which to create the custom resource. Defaults to "default".

        Returns:
            Any: The created custom resource, or None if it already exists or there was an error.
        """
        try:
            resource: Any = self.api.create_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                plural=PLURAL,
                body=executor.get_cluster_body(name),
                namespace=k8s_namespace or executor.namespace,
            )
            return resource
        except ApiException as e:
            if e.status == 409:
                logger.error("raycluster resource already exists. error = {}".format(e.reason))
                return None
            else:
                logger.error("error creating custom resource: {}".format(e))
                return None

    def delete_ray_cluster(
        self,
        name: str,
        executor: KubeRayExecutor,
        k8s_namespace: Optional[str] = None,
        wait: bool = False,
        timeout: int = 300,
        poll_interval: int = 5,
    ) -> Optional[bool]:
        """Delete a Ray cluster custom resource and optionally wait for deletion to complete.

        Parameters:
        - name (str): The name of the Ray cluster custom resource to delete.
        - executor (KubeRayExecutor): The executor containing configuration details.
        - k8s_namespace (str, optional): The namespace in which the Ray cluster exists.
        - wait (bool, optional): Whether to wait for the cluster and all its pods to be fully deleted. Defaults to False.
        - timeout (int, optional): Maximum time to wait for deletion in seconds. Defaults to 300 seconds (5 minutes).
        - poll_interval (int, optional): Time between checks for deletion status in seconds. Defaults to 5 seconds.

        Returns:
            Optional[bool]: True if deletion was successful, None if already deleted or there was an error.
        """
        namespace = k8s_namespace or executor.namespace
        logger.info(f"Deleting Ray cluster: {name} in namespace: {namespace}")

        try:
            self.api.delete_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                plural=PLURAL,
                name=name,
                namespace=namespace,
            )

            if not wait:
                return True

            logger.info(f"Waiting for Ray cluster {name} and its pods to be fully deleted...")
            start_time = time.time()
            cluster_deleted = False

            # Wait for both the CR and all pods to be deleted
            while time.time() - start_time < timeout:
                # Check if CR still exists
                if not cluster_deleted:
                    try:
                        cluster = self.get_ray_cluster(name, namespace)
                        if not cluster:
                            logger.info(f"Ray cluster CR {name} has been deleted")
                            cluster_deleted = True
                    except ApiException as e:
                        if e.status == 404:
                            logger.info(f"Ray cluster CR {name} has been deleted")
                            cluster_deleted = True
                        else:
                            logger.error(f"Error checking Ray cluster status during deletion: {e}")

                # Once CR is deleted, check if any pods remain
                if cluster_deleted:
                    try:
                        # Check for any pods with the cluster label
                        pods = self.core_v1_api.list_namespaced_pod(
                            namespace=namespace, label_selector=f"ray.io/cluster={name}"
                        )

                        if not pods.items:
                            logger.info(f"All pods for Ray cluster {name} have been terminated")
                            return True

                        active_pods = [pod.metadata.name for pod in pods.items]
                        logger.info(
                            f"Waiting for {len(active_pods)} pods to terminate: {', '.join(active_pods[:3])}"
                            + (
                                f"... and {len(active_pods) - 3} more"
                                if len(active_pods) > 3
                                else ""
                            )
                        )

                    except ApiException as e:
                        logger.error(f"Error checking Ray cluster pods during deletion: {e}")

                # Sleep before next check
                time.sleep(poll_interval)

            # If we reach here, we've timed out
            logger.warning(
                f"Timed out waiting for Ray cluster {name} to be fully deleted after {timeout} seconds"
            )

            # Check final state
            try:
                cluster_exists = self.get_ray_cluster(name, namespace) is not None
                if cluster_exists:
                    logger.warning(f"Ray cluster CR {name} still exists after timeout")

                pods = self.core_v1_api.list_namespaced_pod(
                    namespace=namespace, label_selector=f"ray.io/cluster={name}"
                )
                if pods.items:
                    pod_names = [pod.metadata.name for pod in pods.items]
                    logger.warning(
                        f"Ray cluster {name} still has {len(pod_names)} pods: {', '.join(pod_names[:5])}"
                    )
            except Exception as e:
                logger.error(f"Error checking final state of Ray cluster {name}: {e}")

            return False

        except ApiException as e:
            if e.status == 404:
                logger.warning(f"Ray cluster {name} was already deleted")
                return None
            else:
                logger.error(f"Error deleting the Ray cluster {name}: {e}")
                return None

    def patch_ray_cluster(
        self,
        name: str,
        ray_patch: Any,
        executor: KubeRayExecutor,
        k8s_namespace: Optional[str] = None,
    ) -> Any:
        namespace = k8s_namespace or executor.namespace
        logger.info(f"Patching Ray cluster: {name} in namespace: {namespace}")
        """Patch an existing Ray cluster custom resource.

        Parameters:
        - name (str): The name of the Ray cluster custom resource to be patched.
        - ray_patch (Any): The patch data for the Ray cluster.
        - k8s_namespace (str, optional): The namespace in which the Ray cluster exists. Defaults to "default".

        Returns:
            bool: True if the patch was successful, False otherwise.
        """
        try:
            # we patch the existing raycluster with the new config
            self.api.patch_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                plural=PLURAL,
                name=name,
                body=ray_patch,
                namespace=namespace,
            )
        except ApiException as e:
            logger.error("raycluster `{}` failed to patch, with error: {}".format(name, e))
            return False
        else:
            logger.info("raycluster `%s` is patched successfully", name)

        return True

    def port_forward(
        self,
        name: str,
        port: int,
        target_port: int,
        k8s_namespace: str,
        wait: bool = False,
    ):
        """Port forward a Ray cluster service using kubectl in a daemon thread.

        When you want to stop the forwarding:
            forward_thread.stop_forwarding()  # Call this method to stop forwarding

        If wait=True, this function will block until interrupted (e.g., with Ctrl+C).

        Parameters:
        - name (str): The name of the Ray cluster custom resource.
        - port (int): The local port to use for forwarding.
        - target_port (int): The target port on the Ray cluster to forward to.
        - k8s_namespace (str, optional): The namespace in which the Ray cluster exists.
        - wait (bool, optional): If True, block indefinitely until interrupted. Defaults to False.

        Returns:
        - ForwardingThread: A thread object with stop_forwarding method.

        Raises:
        - RuntimeError: If the Ray head service cannot be found.
        - TimeoutError: If port forwarding fails to establish within the timeout period.
        """
        import queue
        import subprocess
        import threading
        import time

        # Get cluster details
        cluster = self.get_ray_cluster(name, k8s_namespace or "default")
        if not cluster:
            raise RuntimeError(f"Could not find Ray cluster {name}")

        namespace = cluster["metadata"]["namespace"]

        # Construct head service name - typically follows the pattern {cluster-name}-head-svc
        service_name = f"{name}-head-svc"

        # Verify the service exists
        try:
            self.core_v1_api.read_namespaced_service(name=service_name, namespace=namespace)
        except ApiException as e:
            if e.status == 404:
                raise RuntimeError(
                    f"Could not find Ray head service {service_name} in namespace {namespace}"
                )
            else:
                raise RuntimeError(f"Error getting Ray head service: {e}")

        # Use a queue for thread communication
        status_queue = queue.Queue()
        stop_event = threading.Event()

        class ForwardingThread(threading.Thread):
            def __init__(self, target, daemon):
                super().__init__(target=target, daemon=daemon)
                self._stop_event = stop_event

            def stop_forwarding(self):
                logger.info("Stopping port forwarding")
                self._stop_event.set()

        def forward_port_daemon():
            logger.info(
                f"Starting port forwarding from localhost:{port} to service {service_name}:{target_port} in namespace {namespace}"
            )

            connection_established = False
            max_retries = 3
            retry_count = 0

            while not stop_event.is_set():
                try:
                    # Use kubectl port-forward to the service instead of pod
                    cmd = [
                        "kubectl",
                        "port-forward",
                        f"service/{service_name}",  # Use service instead of pod
                        f"{port}:{target_port}",
                        "-n",
                        namespace,
                    ]

                    logger.info(f"Running command: {' '.join(cmd)}")

                    process = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )

                    # Signal success to the main thread after short wait to ensure it started
                    time.sleep(1)
                    if process.poll() is None:  # Process is still running
                        if not connection_established:
                            connection_established = True
                            status_queue.put(("success", None))
                            logger.info("Port forwarding connection established")

                        # Wait for the process to complete or be killed
                        while not stop_event.is_set() and process.poll() is None:
                            time.sleep(5)

                        # Kill the process if it's still running
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()  # Force kill if it doesn't terminate

                        # If we were stopped, exit the loop
                        if stop_event.is_set():
                            break

                        # If process exited with error, log it
                        if process.returncode != 0:
                            # Safe way to read stderr that handles None case
                            stderr_output = ""
                            if process.stderr:
                                stderr_output = process.stderr.read() or ""

                            logger.error(
                                f"Port forwarding process exited with code {process.returncode}: {stderr_output}"
                            )

                        # If we get here, the connection was closed unexpectedly
                        logger.info(
                            "Port forwarding connection closed, reconnecting in 5 seconds..."
                        )
                        time.sleep(5)
                        retry_count = 0  # Reset retry count for reconnection attempts

                    else:
                        # Process failed to start
                        stderr_output = ""
                        if process.stderr:
                            stderr_output = process.stderr.read() or ""

                        error_msg = f"Port forwarding process failed to start: {stderr_output}"
                        logger.error(error_msg)

                        retry_count += 1
                        if not connection_established and retry_count >= max_retries:
                            status_queue.put(("error", error_msg))
                            break

                        time.sleep(5)

                except Exception as e:
                    retry_count += 1
                    error_msg = f"Error in port forwarding: {e}"
                    logger.error(f"{error_msg}, retry {retry_count}/{max_retries}...")

                    if not connection_established and retry_count >= max_retries:
                        # Signal failure to the main thread if we couldn't establish the initial connection
                        status_queue.put(("error", error_msg))
                        break

                    if stop_event.is_set():
                        break

                    time.sleep(5)

        # Create and start the daemon thread
        forward_thread = ForwardingThread(target=forward_port_daemon, daemon=True)
        forward_thread.start()

        # Wait for port forwarding to establish or fail with a timeout
        try:
            status, error_msg = status_queue.get(timeout=30)  # 30 second timeout
            if status == "error":
                raise RuntimeError(f"Failed to establish port forwarding: {error_msg}")
        except queue.Empty:
            stop_event.set()  # Signal the thread to stop
            raise TimeoutError("Timed out waiting for port forwarding to establish")

        logger.info(f"Port forwarding daemon started for {name}:{target_port} -> localhost:{port}")

        # If wait option is set, block indefinitely until interrupted
        if wait:
            self._wait_for_forwarding_termination(forward_thread, stop_event)

        return forward_thread

    def _wait_for_forwarding_termination(self, forward_thread, stop_event):
        """Helper method to wait for port forwarding termination.

        Sets up signal handlers and blocks until interrupted or the stop_event is set.

        Parameters:
        - forward_thread: The thread running the port forwarding.
        - stop_event: The event used to signal the thread to stop.
        """
        import signal
        import time

        # Set up signal handler for graceful shutdown
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        original_sigterm_handler = signal.getsignal(signal.SIGTERM)

        def signal_handler(sig, frame):
            logger.info("Received signal to stop port forwarding")
            stop_event.set()

            # Restore original signal handlers
            signal.signal(signal.SIGINT, original_sigint_handler)
            signal.signal(signal.SIGTERM, original_sigterm_handler)

            # We don't need to manually call the original handler
            # The system will use it after we've restored it

        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signal

        try:
            logger.info("Port forwarding is active. Press Ctrl+C to stop...")
            # Main thread waits here indefinitely until interrupted
            while not stop_event.is_set():
                time.sleep(1)
        finally:
            # Ensure we stop the forwarding thread
            if not stop_event.is_set():
                stop_event.set()

            # Wait for the thread to finish
            forward_thread.join(timeout=5)
            logger.info("Port forwarding stopped")
