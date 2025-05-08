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
# Based on https://github.com/ray-project/kuberay/blob/master/clients/python-client/python_client/utils/kuberay_cluster_utils.py

import copy
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from nemo_run.core.execution.base import Executor

# Group, Version, Plural
GROUP = "ray.io"
VERSION = "v1alpha1"
PLURAL = "rayclusters"
KIND = "RayCluster"

logger = logging.getLogger(__name__)


@dataclass
class KubeRayWorkerGroup:
    """
    Configuration for a Ray worker group in a KubeRay cluster.
    """

    group_name: str
    replicas: int = 1
    min_replicas: Optional[int] = None  # Will be set in __post_init__
    max_replicas: Optional[int] = None  # Will be set in __post_init__
    gpus_per_worker: Optional[int] = None
    cpu_requests: Optional[str] = None
    memory_requests: Optional[str] = None
    cpu_limits: Optional[str] = None
    memory_limits: Optional[str] = None
    volume_mounts: list[dict[str, Any]] = field(default_factory=list)
    volumes: list[dict[str, Any]] = field(default_factory=list)
    labels: dict[str, Any] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Set min_replicas and max_replicas to match replicas if not set
        if self.min_replicas is None:
            self.min_replicas = self.replicas
        if self.max_replicas is None:
            self.max_replicas = self.replicas


@dataclass(kw_only=True)
class KubeRayExecutor(Executor):
    """
    Dataclass to configure a KubeRay Executor.

    This executor integrates with KubeRay to create and manage Ray clusters
    on Kubernetes using the KubeRay API.
    """

    namespace: str = "default"
    ray_version: str = "2.9.0"
    image: str = ""  # Will be set in __post_init__ if empty
    head_cpu: str = "1"
    head_memory: str = "2Gi"
    ray_start_params: Dict[str, Any] = field(default_factory=dict)
    worker_groups: List[KubeRayWorkerGroup] = field(default_factory=list)
    labels: Dict[str, Any] = field(default_factory=dict)
    service_type: str = "ClusterIP"
    head_ports: list[dict[str, Any]] = field(default_factory=list)
    volume_mounts: list[dict[str, Any]] = field(default_factory=list)
    volumes: list[dict[str, Any]] = field(default_factory=list)
    reuse_volumes_in_worker_groups: bool = False
    spec_kwargs: dict[str, Any] = field(default_factory=dict)
    lifecycle_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Set default image based on ray_version if not provided
        if not self.image:
            self.image = f"rayproject/ray:{self.ray_version}"

        if self.reuse_volumes_in_worker_groups:
            for worker_group in self.worker_groups:
                worker_group.volumes = copy.deepcopy(self.volumes)
                worker_group.volume_mounts = copy.deepcopy(self.volume_mounts)

    def get_cluster_body(self, name: str) -> dict[str, Any]:
        """
        Get the body for the Ray cluster custom resource.
        """
        cluster = {}
        cluster = populate_meta(
            cluster,
            name,
            k8s_namespace=self.namespace,
            labels=self.labels,
            ray_version=self.ray_version,
        )
        cluster = populate_ray_head(
            cluster,
            ray_image=self.image,
            service_type=self.service_type,
            cpu_requests=self.head_cpu,
            memory_requests=self.head_memory,
            cpu_limits=self.head_cpu,
            memory_limits=self.head_memory,
            ray_start_params=self.ray_start_params,
            head_ports=self.head_ports,
            volumes=self.volumes,
            volume_mounts=self.volume_mounts,
            spec_kwargs=self.spec_kwargs,
            lifecycle_kwargs=self.lifecycle_kwargs,
        )
        for worker_group in self.worker_groups:
            cluster = populate_worker_group(
                cluster,
                group_name=worker_group.group_name,
                ray_image=self.image,
                gpus_per_worker=worker_group.gpus_per_worker,
                cpu_requests=worker_group.cpu_requests,
                memory_requests=worker_group.memory_requests,
                cpu_limits=worker_group.cpu_limits,
                memory_limits=worker_group.memory_limits,
                replicas=worker_group.replicas,
                min_replicas=worker_group.min_replicas or worker_group.replicas,
                max_replicas=worker_group.max_replicas or worker_group.replicas,
                ray_start_params=self.ray_start_params,
                volume_mounts=worker_group.volume_mounts,
                volumes=worker_group.volumes,
                labels=worker_group.labels,
                annotations=worker_group.annotations,
                spec_kwargs=self.spec_kwargs,
                lifecycle_kwargs=self.lifecycle_kwargs,
            )
        return cluster


def is_valid_name(name: str) -> bool:
    msg = "The name must be 63 characters or less, begin and end with an alphanumeric character, and contain only dashes, dots, and alphanumerics."
    if len(name) > 63 or not bool(re.match("^[a-z0-9]([-.]*[a-z0-9])+$", name)):
        logger.info(msg)
        return False
    return True


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
    lifecycle_kwargs: dict[str, Any],
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
                            "preStop": {"exec": {"command": ["/bin/sh", "-c", "ray stop"]}},
                            **lifecycle_kwargs,
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
    lifecycle_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Populate the worker group specification in the cluster dictionary.

    Parameters:
    - group_name (str): The name of the worker group.
    - ray_image (str): The image to use for the Ray worker containers.
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
                "containers": [
                    {
                        "image": ray_image,
                        "lifecycle": {
                            "preStop": {"exec": {"command": ["/bin/sh", "-c", "ray stop"]}},
                            **lifecycle_kwargs,
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


def is_valid_label(name: str) -> bool:
    msg = "The label name must be 63 characters or less, begin and end with an alphanumeric character, and contain only dashes, underscores, dots, and alphanumerics."
    if len(name) > 63 or not bool(re.match("^[a-z0-9]([-._]*[a-z0-9])+$", name)):
        logger.error(msg)
        return False
    return True
