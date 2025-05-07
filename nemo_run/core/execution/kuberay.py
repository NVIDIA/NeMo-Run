import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List

from nemo_run.core.execution.base import Executor
from nemo_run.run.ray.kuberay import (
    populate_meta,
    populate_ray_head,
    populate_worker_group,
)


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
    ray_command: Optional[Any] = None
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
        )
        for worker_group in self.worker_groups:
            cluster = populate_worker_group(
                cluster,
                group_name=worker_group.group_name,
                ray_image=self.image,
                ray_command=worker_group.ray_command,
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
            )
        return cluster
