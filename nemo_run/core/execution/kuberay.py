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
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from kubernetes import client, watch
from kubernetes.client import CoreV1Api
from kubernetes.client.rest import ApiException

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
    ray_version: str = "2.43.0"
    image: str = ""  # Will be set in __post_init__ if empty
    head_cpu: str = "1"
    head_memory: str = "2Gi"
    ray_head_start_params: dict[str, Any] = field(default_factory=dict)
    ray_worker_start_params: dict[str, Any] = field(default_factory=dict)
    worker_groups: list[KubeRayWorkerGroup] = field(default_factory=list)
    labels: dict[str, Any] = field(default_factory=dict)
    service_type: str = "ClusterIP"
    head_ports: list[dict[str, Any]] = field(default_factory=list)
    volume_mounts: list[dict[str, Any]] = field(default_factory=list)
    volumes: list[dict[str, Any]] = field(default_factory=list)
    reuse_volumes_in_worker_groups: bool = True
    spec_kwargs: dict[str, Any] = field(default_factory=dict)
    container_kwargs: dict[str, Any] = field(default_factory=dict)
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
            ray_start_params=self.ray_head_start_params,
            head_ports=self.head_ports,
            env_vars=self.env_vars,
            volumes=self.volumes,
            volume_mounts=self.volume_mounts,
            spec_kwargs=self.spec_kwargs,
            lifecycle_kwargs=self.lifecycle_kwargs,
            container_kwargs=self.container_kwargs,
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
                ray_start_params=self.ray_worker_start_params,
                volume_mounts=worker_group.volume_mounts,
                volumes=worker_group.volumes,
                labels=worker_group.labels,
                annotations=worker_group.annotations,
                spec_kwargs=self.spec_kwargs,
                lifecycle_kwargs=self.lifecycle_kwargs,
                container_kwargs=self.container_kwargs,
                env_vars=self.env_vars,
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
    env_vars: dict[str, str],
    volume_mounts: list[dict[str, Any]],
    volumes: list[dict[str, Any]],
    spec_kwargs: dict[str, Any],
    lifecycle_kwargs: dict[str, Any],
    container_kwargs: dict[str, Any],
) -> dict[str, Any]:
    # make sure metadata exists
    if "spec" in cluster.keys():
        if "headGroupSpec" not in cluster.keys():
            logger.debug(f"setting the headGroupSpec for cluster {cluster['metadata']['name']}")
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
                        "env": [{"name": k, "value": v} for k, v in env_vars.items()],
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
                        **container_kwargs,
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
    container_kwargs: dict[str, Any],
    env_vars: dict[str, str],
) -> dict[str, Any]:
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
                        "name": "ray-worker",
                        "env": [{"name": k, "value": v} for k, v in env_vars.items()],
                        "lifecycle": {
                            "preStop": {"exec": {"command": ["/bin/sh", "-c", "ray stop"]}},
                            **lifecycle_kwargs,
                        },
                        "resources": {
                            "requests": resource_requests,
                            "limits": resource_limits,
                        },
                        "volumeMounts": volume_mounts,
                        **container_kwargs,
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
) -> tuple[dict, bool]:
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
) -> tuple[dict, bool]:
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
) -> tuple[dict, bool]:
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
) -> tuple[dict, bool]:
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


def sync_workdir_via_pod(
    *,
    pod_name: str,
    namespace: str,
    user_workspace_path: str,
    workdir: str,
    core_v1_api: CoreV1Api,
    volumes: list[dict[str, object]],
    volume_mounts: list[dict[str, object]],
    image: str = "alpine:3.19",
    cleanup: bool = False,
    cleanup_timeout: int = 5,
) -> None:
    """Spin up a throw-away Pod that mounts the same volumes as the Ray
    cluster and streams *workdir* into *workspace_path* inside the mount.

    The function blocks until the copy is complete and the Pod is removed.
    Requires that the *kubectl* binary is available in PATH and can access
    the same cluster context as the Kubernetes Python client.
    """
    # Pod manifest
    pod_body = client.V1Pod(
        metadata=client.V1ObjectMeta(name=pod_name, namespace=namespace),
        spec=client.V1PodSpec(
            restart_policy="Never",
            containers=[
                client.V1Container(
                    name="mover",
                    image=image,
                    command=["sh", "-c", "sleep infinity"],
                    volume_mounts=volume_mounts,
                    lifecycle={
                        "postStart": {
                            "exec": {
                                "command": [
                                    "sh",
                                    "-c",
                                    # Install rsync on first container start if missing (Alpine)
                                    "command -v rsync >/dev/null 2>&1 || apk add --no-cache rsync",
                                ]
                            }
                        }
                    },
                )
            ],
            volumes=volumes,
        ),
    )

    # Create Pod (idempotent – reuse if already exists)
    logger.info(
        f"Creating data-mover pod '{pod_name}' in namespace '{namespace}' (or re-using if present)"
    )
    try:
        core_v1_api.create_namespaced_pod(namespace=namespace, body=pod_body)
    except ApiException as e:
        if e.status == 409:  # AlreadyExists
            logger.info(f"Data-mover pod '{pod_name}' already exists – will reuse it")
        else:
            raise

    # Wait until pod is Running
    w = watch.Watch()
    for event in w.stream(
        core_v1_api.list_namespaced_pod,
        namespace=namespace,
        field_selector=f"metadata.name={pod_name}",
        timeout_seconds=120,
    ):
        pod_obj: client.V1Pod = event.get("object")  # type: ignore[assignment]
        phase = pod_obj.status.phase if pod_obj.status else None
        if phase == "Running":
            w.stop()
            break
    else:
        raise RuntimeError("Data-mover pod did not reach Running state in time")

    # Ensure workspace dir exists
    subprocess.check_call(
        [
            "kubectl",
            "exec",
            "-n",
            namespace,
            pod_name,
            "--",
            "mkdir",
            "-p",
            user_workspace_path,
        ]
    )

    # Use rsync over kubectl exec
    rsync_cmd: list[str] = [
        "rsync",
        "-az",
        "--delete",
    ]

    # Respect .gitignore rules if present in the workdir
    if os.path.isfile(os.path.join(workdir, ".gitignore")):
        rsync_cmd.extend(["--filter=:- .gitignore"])

    # Tell rsync to reach the remote side via kubectl exec
    rsync_cmd.extend(
        [
            "-e",
            f"kubectl exec -i -n {namespace} {pod_name}",
            "--",  # Marks end-of-options for rsync – mandatory when the dest starts with "--:"
            f"{os.path.abspath(workdir).rstrip(os.sep)}/",
            f"--:{user_workspace_path.rstrip('/')}/",
        ]
    )

    # Emit the full command for easier troubleshooting
    logger.debug("Running rsync command: %s", " ".join(rsync_cmd))

    subprocess.check_call(rsync_cmd)
    logger.info(f"Workdir synced to PVC at {user_workspace_path} via data-mover pod.")

    if cleanup:
        core_v1_api.delete_namespaced_pod(
            name=pod_name, namespace=namespace, body=client.V1DeleteOptions()
        )

        # Wait for termination
        timeout = time.time() + cleanup_timeout
        while time.time() < timeout:
            try:
                core_v1_api.read_namespaced_pod(name=pod_name, namespace=namespace)
            except ApiException as e:
                if e.status == 404:
                    break
            time.sleep(2)
