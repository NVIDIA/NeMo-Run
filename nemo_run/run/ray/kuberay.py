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
# Based on https://github.com/ray-project/kuberay/blob/master/clients/python-client/python_client/kuberay_cluster_api.py

import getpass
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from nemo_run.core.execution.kuberay import GROUP, PLURAL, VERSION, KubeRayExecutor

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class KubeRayCluster:
    """Lightweight helper around the KubeRay operator RayCluster CRD lifecycle.

    The class mirrors :class:`SlurmRayCluster` and :class:`KubeRayJob` – it's
    a lightweight dataclass holding just two identifying fields (*name* and
    *executor*).  All Kubernetes clients are instantiated lazily in
    :py:meth:`__post_init__`.
    """

    # ------------------------------------------------------------------
    # Class-level constants / type bindings
    # ------------------------------------------------------------------
    EXECUTOR_CLS = KubeRayExecutor

    # ------------------------------------------------------------------
    # Primary identifiers (mirrors SlurmRayCluster API)
    # ------------------------------------------------------------------
    name: str
    executor: KubeRayExecutor

    # ------------------------------------------------------------------
    # Dataclass lifecycle hooks
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:  # noqa: D401 – simple verb is fine
        """Initialise Kubernetes API clients once the instance is created."""
        # Load local kube-config once; the function returns *None* so we don't store it.
        config.load_kube_config()

        # The dedicated clients are what we interact with throughout the class
        # – separating CoreV1 for pods/services from CustomObjects for CRDs.
        self.api = client.CustomObjectsApi()
        self.core_v1_api = client.CoreV1Api()
        self.user = get_user()
        self.cluster_name = f"{self.user}-{self.name}-raycluster"

    def _get(
        self,
        name: Optional[str] = None,
        k8s_namespace: Optional[str] = None,
    ) -> Any:
        # Return the RayCluster custom object, if present.

        name = name or self.cluster_name
        namespace = k8s_namespace or self.executor.namespace or "default"

        logger.debug(f"Getting Ray cluster '{name}' in namespace '{namespace}'")

        try:
            resource: Any = self.api.get_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                plural=PLURAL,
                name=name,
                namespace=namespace,
            )
            return resource
        except ApiException as e:
            if e.status == 404:
                logger.error(f"Ray cluster '{name}' not found in namespace '{namespace}': {e}")
                return None
            else:
                logger.error(f"Error fetching Ray cluster '{name}' in namespace '{namespace}': {e}")
                return None

    def status(
        self,
        timeout: int = 60,
        delay_between_attempts: int = 5,
        display: bool = False,
    ) -> Any:
        """Return the ``status`` stanza of the RayCluster CR (blocking).

        Polls until the CR contains a *status* field or *timeout* is reached.
        """

        namespace = self.executor.namespace or "default"
        name = self.cluster_name

        logger.debug(
            f"Getting Ray cluster status for '{name}' in namespace '{namespace}', "
            f"timeout: {timeout}s, delay: {delay_between_attempts}s"
        )

        remaining = timeout
        while remaining > 0:
            try:
                resource: Any = self.api.get_namespaced_custom_object_status(
                    group=GROUP,
                    version=VERSION,
                    plural=PLURAL,
                    name=name,
                    namespace=namespace,
                )
            except ApiException as e:
                if e.status == 404:
                    logger.debug(
                        f"Ray cluster '{name}' status fetch failed: resource not found: {e}"
                    )
                    return None
                logger.error(
                    f"Error fetching status for Ray cluster '{name}' in namespace '{namespace}': {e}"
                )
                return None

            if resource.get("status"):
                status_dict = resource["status"]
                if display:
                    self._display_banner(name, status_dict)
                return status_dict

            logger.debug(f"Ray cluster '{name}' status not set yet, waiting...")
            time.sleep(delay_between_attempts)
            remaining -= delay_between_attempts

        logger.debug(f"Ray cluster '{name}' status not set yet, timing out...")
        return None

    def wait_until_running(
        self,
        timeout: int = 600,
        delay_between_attempts: int = 5,
    ) -> bool:
        """Block until the Ray head service has a reachable IP **and** the head pod is running.

        The previous implementation returned as soon as the operator had
        populated ``status.head.serviceIP`` in the RayCluster CR.  This is a
        good proxy for readiness of the *service* object but does **not**
        guarantee that the underlying *pod* has actually reached the
        ``Running``/``Ready`` state.

        We now additionally query the Kubernetes API for the head pod and
        ensure that it is both *Running* **and** *Ready* before returning
        success.  The head pod is identified via the same labels that the
        KubeRay operator applies to every pod:

        • ``ray.io/cluster=<cluster-name>``
        • ``ray.io/node-type=head``
        """

        namespace = self.executor.namespace or "default"
        name = self.cluster_name

        logger.info(
            f"Waiting until Ray cluster '{name}' is running in namespace '{namespace}', "
            f"timeout: {timeout}s, delay: {delay_between_attempts}s"
        )

        def _head_pod_is_ready() -> bool:
            """Return *True* if the head pod exists and is Running/Ready."""
            try:
                pods = self.core_v1_api.list_namespaced_pod(
                    namespace=namespace, label_selector=f"ray.io/cluster={name}"
                )
            except ApiException as e:
                logger.debug(f"Error listing pods for Ray cluster '{name}': {e}")
                return False

            for pod in pods.items:
                labels = pod.metadata.labels or {}
                # Newer KubeRay versions set `ray.io/node-type=head`; fall back to
                # a heuristic on the pod name otherwise.
                is_head = labels.get("ray.io/node-type") == "head" or "-head" in pod.metadata.name
                if not is_head:
                    continue

                if pod.status.phase != "Running":
                    return False

                # Ensure the Ready condition is *True* (best-effort)
                if pod.status.conditions:
                    for cond in pod.status.conditions:
                        if cond.type == "Ready":
                            return cond.status == "True"
                # If no conditions, fall back to phase only
                return True

            # No head pod found
            return False

        remaining = timeout
        while remaining > 0:
            poll_window = min(delay_between_attempts, remaining)

            status = self.status(display=False)
            if not status:
                logger.info(f"Ray cluster '{name}' status could not be retrieved")
                return False

            svc_ip_ready = bool(status.get("head", {}).get("serviceIP"))
            pod_ready = False
            if svc_ip_ready:
                pod_ready = _head_pod_is_ready()

            if svc_ip_ready and pod_ready:
                logger.info(f"Ray cluster '{name}' is running and head pod is ready")
                return True

            logger.debug(
                f"Ray cluster '{name}' not ready yet – svc_ip_ready={svc_ip_ready}, pod_ready={pod_ready}"
            )

            remaining -= poll_window

        logger.debug(f"Ray cluster '{name}' status is not running yet, timing out...")
        return False

    def create(
        self,
        pre_ray_start_commands: Optional[list[str]] = None,
        dryrun: bool = False,
    ) -> Any:
        """Create the RayCluster CR (idempotent)."""

        namespace = self.executor.namespace or "default"
        name = self.cluster_name

        logger.info(f"Creating Ray cluster '{name}' in namespace '{namespace}'")

        # Ensure lifecycle_kwargs dict exists (older executor versions may omit it)
        if not hasattr(self.executor, "lifecycle_kwargs") or self.executor.lifecycle_kwargs is None:
            self.executor.lifecycle_kwargs = {}

        if pre_ray_start_commands:
            k8s_pre_ray_start_commands = "\n".join(pre_ray_start_commands)
            self.executor.lifecycle_kwargs["postStart"] = {
                "exec": {"command": ["/bin/sh", "-c", k8s_pre_ray_start_commands]}
            }

        body = self.executor.get_cluster_body(name)

        if dryrun:
            print(yaml.dump(body))
            return body

        try:
            resource: Any = self.api.create_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                plural=PLURAL,
                body=body,
                namespace=namespace,
            )
            return resource
        except ApiException as e:
            if e.status == 409:
                logger.error(f"Ray cluster '{name}' already exists: {e.reason}")
                return None
            logger.error(f"Error creating Ray cluster '{name}' in namespace '{namespace}': {e}")
            return None

    def delete(
        self,
        wait: bool = False,
        timeout: int = 300,
        poll_interval: int = 5,
    ) -> Optional[bool]:
        """Delete the RayCluster CR and, optionally, wait for full teardown.

        Parameters
        ----------
        wait : bool, default False
            When *True*, block until the RayCluster CR and all its pods have
            disappeared.  A best-effort poll is performed every
            *poll_interval* seconds up to *timeout* seconds.
        timeout : int, default 300
            Maximum time in seconds to wait for deletion when *wait* is
            enabled.
        poll_interval : int, default 5
            Interval between successive status checks.

        Returns
        -------
        bool | None
            • *True*  – deletion confirmed.
            • *False* – timed out while waiting.
            • *None*  – cluster already absent before the call.
        """
        namespace = self.executor.namespace or "default"
        name = self.cluster_name

        logger.info(f"Deleting Ray cluster '{name}' in namespace '{namespace}'")

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

            logger.debug(f"Waiting for Ray cluster '{name}' and its pods to be fully deleted...")
            start_time = time.time()
            cluster_deleted = False

            # Wait for both the CR and all pods to be deleted
            while time.time() - start_time < timeout:
                # Check if CR still exists
                if not cluster_deleted:
                    try:
                        cluster = self._get(name=name, k8s_namespace=namespace)
                        if not cluster:
                            logger.info(f"Ray cluster CR '{name}' has been deleted")
                            cluster_deleted = True
                    except ApiException as e:
                        if e.status == 404:
                            logger.info(f"Ray cluster CR '{name}' has been deleted")
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
                            logger.info(f"All pods for Ray cluster '{name}' have been terminated")
                            return True

                        active_pods = [pod.metadata.name for pod in pods.items]
                        logger.debug(
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
                f"Timed out waiting for Ray cluster '{name}' to be fully deleted after {timeout} seconds"
            )

            # Check final state
            try:
                cluster_exists = self._get(name=name, k8s_namespace=namespace) is not None
                if cluster_exists:
                    logger.warning(f"Ray cluster CR '{name}' still exists after timeout")

                pods = self.core_v1_api.list_namespaced_pod(
                    namespace=namespace, label_selector=f"ray.io/cluster={name}"
                )
                if pods.items:
                    pod_names = [pod.metadata.name for pod in pods.items]
                    logger.warning(
                        f"Ray cluster '{name}' still has {len(pod_names)} pods: {', '.join(pod_names[:5])}"
                    )
            except Exception as e:
                logger.error(f"Error checking final state of Ray cluster '{name}': {e}")

            return False

        except ApiException as e:
            if e.status == 404:
                logger.warning(f"Ray cluster '{name}' was already deleted")
                return None
            else:
                logger.error(f"Error deleting Ray cluster '{name}': {e}")
                return None

    def patch(
        self,
        ray_patch: Any,
    ) -> Any:
        """Patch the RayCluster custom resource with a user-supplied body.

        The patch is applied using the Kubernetes *merge* strategy, mirroring
        ``kubectl patch --type=merge``.

        Parameters
        ----------
        ray_patch : Any
            A JSON-serialisable object representing the patch to apply.

        Returns
        -------
        bool
            *True* on success, *False* if the API call raised an exception.
        """
        namespace = self.executor.namespace or "default"
        name = self.cluster_name
        logger.info(f"Patching Ray cluster '{name}' in namespace '{namespace}'")
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
            logger.error(f"Failed to patch Ray cluster '{name}': {e}")
            return False
        else:
            logger.info(f"Ray cluster '{name}' patched successfully")

        return True

    def port_forward(
        self,
        port: int,
        target_port: int,
        wait: bool = False,
    ):
        """Expose the Ray head service locally via *kubectl port-forward*.

        Parameters
        ----------
        port : int
            Local port on which to listen.
        target_port : int
            Port number of the Ray head service inside the cluster.
        wait : bool, default False
            If *True*, block until the user terminates the process (SIGINT or
            SIGTERM).  Otherwise a daemon thread is returned immediately.

        Returns
        -------
        threading.Thread
            The daemon thread encapsulating the port-forwarding subprocess.
        """
        import queue
        import subprocess
        import threading
        import time

        name = self.cluster_name
        executor = self.executor

        # Get cluster details
        cluster = self._get(name=name, k8s_namespace=executor.namespace or "default")
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
                logger.debug("Stopping port forwarding")
                self._stop_event.set()

        def forward_port_daemon():
            logger.debug(
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

                    logger.debug(f"Running command: {' '.join(cmd)}")

                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,  # avoid dead-lock on unread STDOUT
                        stderr=subprocess.PIPE,
                        text=True,
                    )

                    # Signal success to the main thread after short wait to ensure it started
                    time.sleep(1)
                    if process.poll() is None:  # Process is still running
                        if not connection_established:
                            connection_established = True
                            status_queue.put(("success", None))
                            logger.debug("Port forwarding connection established")

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
                            stderr_output = ""
                            if process.stderr:
                                stderr_output = process.stderr.read() or ""

                            logger.debug(
                                f"Port forwarding process exited with code {process.returncode}: {stderr_output}"
                            )

                        # If we get here, the connection was closed unexpectedly
                        logger.debug(
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
        import signal
        import time

        # Set up signal handler for graceful shutdown
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        original_sigterm_handler = signal.getsignal(signal.SIGTERM)

        def signal_handler(sig, frame):
            logger.info("Received signal to stop port forwarding.")
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
            logger.info("Waiting for port forwarding thread to finish for 5 seconds...")
            forward_thread.join(timeout=5)
            logger.info("Port forwarding stopped")

    # Helper to print banner
    def _display_banner(self, name: str, status_dict: Any) -> None:
        namespace = self.executor.namespace or "default"

        pvc_locations = ", ".join(
            [vm.get("mountPath", "N/A") for vm in self.executor.volume_mounts]
        )

        logger.info(
            f"""
Ray Cluster Status (KubeRay)
============================

Namespace: {namespace}
Name:      {name}
State:     {status_dict.get("state", "UNKNOWN") if isinstance(status_dict, dict) else "UNKNOWN"}
Head IP:   {status_dict.get("head", {}).get("serviceIP") if isinstance(status_dict, dict) else "N/A"}
Persistent file paths: {pvc_locations}

Useful Commands
---------------

• Inspect cluster:
  kubectl get rayclusters {name} -n {namespace}

• Delete cluster:
  kubectl delete rayclusters {name} -n {namespace}

• Exec into Ray head pod:
  kubectl exec -it -n {namespace} $(kubectl get pods -n {namespace} \\
    -l ray.io/cluster={name},ray.io/node-type=head \\
    -o jsonpath='{{.items[0].metadata.name}}') -- /bin/bash

• View Ray dashboard:
  kubectl port-forward -n {namespace} service/{name}-head-svc 8265:8265

• List all pods:
  kubectl get pods -n {namespace} -l ray.io/cluster={name}
"""
        )


@dataclass(kw_only=True)
class KubeRayJob:
    """Helper object for interacting with a KubeRay RayJob.

    Parameters
    ----------
    name : str
        Name of the RayJob custom resource.
    namespace : str
        Kubernetes namespace in which the job was created.
    """

    name: str
    executor: KubeRayExecutor

    def __post_init__(self):
        config.load_kube_config()

        # Lazily create K8s API clients if not supplied
        self.api = client.CustomObjectsApi()
        self.core_v1_api = client.CoreV1Api()
        # Ensure backward-compat: if cluster is None we still work (stand-alone usage)
        self.user = get_user()
        self.job_name = f"{self.user}-{self.name}-rayjob"

    # ------------------------------------------------------------------
    # Public helpers mirroring SlurmRayJob API for downstream symmetry.
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Delete the RayJob custom resource (equivalent to job cancellation)."""
        logger.debug(
            f"Cancelling RayJob '{self.job_name}' in namespace '{self.executor.namespace}'"
        )
        try:
            self.api.delete_namespaced_custom_object(
                group="ray.io",
                version="v1",
                plural="rayjobs",
                name=self.job_name,
                namespace=self.executor.namespace,
            )
            logger.debug(f"RayJob '{self.job_name}' cancellation requested (CR deleted)")
        except ApiException as e:
            if e.status == 404:
                logger.warning(f"RayJob '{self.job_name}' not found – maybe already deleted")
            else:
                logger.error(f"Failed to cancel RayJob '{self.job_name}': {e}")

    def logs(self, follow: bool = False, lines: int = 100, timeout: int | None = None) -> None:
        """Stream or show logs from the RayJob submitter pod.

        This simply shells out to ``kubectl logs -l job-name=<rayjob>`` which
        is how the Ray docs recommend fetching RayJob logs.
        """

        cmd = [
            "kubectl",
            "logs",
            "-l",
            f"job-name={self.job_name}",
            "-n",
            self.executor.namespace,
        ]

        if follow:
            cmd.append("-f")
        else:
            cmd.extend(["--tail", str(lines)])

        logger.info(
            f"Running: {' '.join(cmd)} (streaming={'yes' if follow else 'no'}, tail={lines})"
        )

        try:
            if follow:
                subprocess.run(cmd, check=False, timeout=timeout)
            else:
                output = subprocess.check_output(cmd, text=True, timeout=timeout)
                print(output)
        except FileNotFoundError:
            logger.error("kubectl not found in PATH – cannot fetch logs")
        except subprocess.CalledProcessError as e:
            logger.error(f"kubectl logs returned non-zero exit status {e.returncode}")
        except subprocess.TimeoutExpired:
            logger.error(f"kubectl logs timed out after {timeout} seconds")

    def status(self, display: bool = True) -> Dict[str, Any]:
        """Return current RayJob status as a lightweight dict and pretty-print it."""

        try:
            resource = self.api.get_namespaced_custom_object(
                group="ray.io",
                version="v1",
                plural="rayjobs",
                name=self.job_name,
                namespace=self.executor.namespace,
            )
        except ApiException as e:
            logger.error(f"Failed to fetch status for RayJob '{self.job_name}': {e}")
            return {"jobStatus": "ERROR", "jobDeploymentStatus": "ERROR"}

        status = resource.get("status", {}) if isinstance(resource, dict) else {}
        job_status = status.get("jobStatus", "UNKNOWN")
        deployment_status = status.get("jobDeploymentStatus", "UNKNOWN")

        if display:
            # Derive related resource names
            data_mover_pod = f"{self.job_name}-data-mover"
            ray_cluster_name = f"{self.job_name}-raycluster"
            pvc_locations = ", ".join(
                [vm.get("mountPath", "N/A") for vm in self.executor.volume_mounts]
            )

            # Construct workdir paths based on standard patterns
            # Note: These are estimates based on the naming conventions in the code
            user_workspace_base = f"{self.executor.volume_mounts[0]['mountPath']}/{self.user}/code"

            logger.info(
                f"""
Ray Job Status (KubeRay)
========================

Namespace:         {self.executor.namespace}
Job Name:          {self.job_name}
Job Status:        {job_status}
Deployment Status: {deployment_status}
Persistent file paths: {pvc_locations}

Related Resources
-----------------

• Ray Cluster:     {ray_cluster_name}
• Data Mover Pod:  {data_mover_pod}
  (syncs local workdir to PVC)

Workdir Locations
-----------------

• Local code synced to: {user_workspace_base}/<workdir-name>
• Container workdir:    {user_workspace_base}/<workdir-name>

Useful Commands
---------------

• View logs:
  kubectl logs -l job-name={self.job_name} -n {self.executor.namespace} -f

• Exec into Ray head pod:
  kubectl exec -it -n {self.executor.namespace} $(kubectl get pods -n {self.executor.namespace} \\
    -l ray.io/cluster={ray_cluster_name},ray.io/node-type=head \\
    -o jsonpath='{{.items[0].metadata.name}}') -- /bin/bash

• Exec into data mover pod:
  kubectl exec -it {data_mover_pod} -n {self.executor.namespace} -- /bin/bash

• Check Ray cluster status:
  kubectl get raycluster {ray_cluster_name} -n {self.executor.namespace}
"""
            )

        return {"jobStatus": job_status, "jobDeploymentStatus": deployment_status}

    # ------------------------------------------------------------------
    # Convenience: tail logs asynchronously while waiting for completion,
    # then optionally delete the RayJob CR once finished.
    # ------------------------------------------------------------------

    def follow_logs_until_completion(
        self,
        poll_interval: int = 10,
        delete_on_finish: bool = True,
    ) -> None:
        """Stream job logs in real-time and clean up when the RayJob ends.

        This helper starts a background thread running ``kubectl logs -f``
        while the main thread polls the RayJob status every *poll_interval*
        seconds.  As soon as the job transitions to a terminal state
        (SUCCEEDED/FAILED or Deployment Complete/Failed) the log thread is
        joined and – if *delete_on_finish* is *True* – the RayJob CR is
        deleted.
        """

        # ------------------------------------------------------------------
        # 1) Poll until the RayJob is actually running – only then start logs
        # ------------------------------------------------------------------

        RUNNING_DEPLOY_STATUS = "Running"

        while True:
            st = self.status(display=False)
            if st.get("jobDeploymentStatus") == RUNNING_DEPLOY_STATUS:
                break

            # If job already finished/failed before reaching Running, bail out
            if st.get("jobDeploymentStatus") in {"Complete", "Failed"}:
                if delete_on_finish:
                    self.stop()
                return
            time.sleep(poll_interval)

        # ------------------------------------------------------------------
        # 2) Start log streaming in a daemon thread
        # ------------------------------------------------------------------

        def _tail():
            try:
                self.logs(follow=True)
            except Exception as e:  # pragma: no cover – logging only
                logger.error(f"Log tailing thread encountered an error: {e}")

        import threading

        log_thread = threading.Thread(target=_tail, daemon=True)
        log_thread.start()

        # ------------------------------------------------------------------
        # 3) Poll until RayJob ends, then cleanup
        # ------------------------------------------------------------------

        TERMINAL_JOB_STATUSES = {"SUCCEEDED", "FAILED"}
        TERMINAL_DEPLOY_STATUSES = {"Complete", "Failed"}

        try:
            while True:
                status = self.status(display=False)
                if (
                    status.get("jobStatus") in TERMINAL_JOB_STATUSES
                    or status.get("jobDeploymentStatus") in TERMINAL_DEPLOY_STATUSES
                ):
                    break
                time.sleep(poll_interval)
        finally:
            log_thread.join(timeout=5)

            if delete_on_finish:
                try:
                    self.stop()
                except Exception as e:  # pragma: no cover
                    logger.debug(f"Ignoring error during job cleanup: {e}")

    def start(
        self,
        command: str,
        workdir: str | None = None,
        runtime_env_yaml: str | None = None,
        pre_ray_start_commands: Optional[list[str]] = None,
        dryrun: bool = False,
    ):
        """Create a RayJob CR via the KubeRay operator and return a live helper.

        This is a front-door convenience wrapper around
        :py:meth:`KubeRayCluster.schedule_ray_job` so users can directly do::

            KubeRayJob.start(
                name="my-job",
                executor=my_kuberay_executor,
                command="python train.py",
                workdir="./src",
            )
        """
        # We directly replicate the logic previously living in
        # `KubeRayCluster.schedule_ray_job` so that callers interact solely with
        # *job* helpers, keeping cluster classes focused on cluster lifecycle
        # only.

        # ------------------------------------------------------------------
        # 1.  Handle optional *workdir* sync (data-mover pod).
        # ------------------------------------------------------------------
        from nemo_run.core.execution.kuberay import sync_workdir_via_pod

        executor = self.executor
        namespace = executor.namespace

        # Ensure lifecycle_kwargs dict exists on executor
        if not hasattr(executor, "lifecycle_kwargs") or executor.lifecycle_kwargs is None:
            executor.lifecycle_kwargs = {}

        if pre_ray_start_commands:
            k8s_pre_cmds = "\n".join(pre_ray_start_commands)
            executor.lifecycle_kwargs["postStart"] = {
                "exec": {"command": ["/bin/sh", "-c", k8s_pre_cmds]}
            }

        user_workspace_path = None

        if workdir:
            if not executor.volumes or not executor.volume_mounts:
                raise ValueError(
                    "`workdir` specified but executor has no volumes/volume_mounts to mount it."
                )

            user_workspace_path = os.path.join(
                executor.volume_mounts[0]["mountPath"], self.user, "code", Path(workdir).name
            )
            # Add user-based scoping to pod name and workspace path
            pod_name = f"{self.job_name}-data-mover"

            if not dryrun:
                sync_workdir_via_pod(
                    pod_name=pod_name,
                    namespace=namespace,
                    user_workspace_path=user_workspace_path,
                    workdir=workdir,
                    core_v1_api=self.core_v1_api,
                    volumes=executor.volumes,
                    volume_mounts=executor.volume_mounts,
                )
                logger.info(f"Synced workdir {workdir} to {user_workspace_path}")

        # In-place patch of executor.lifecycle_kwargs with *postStart* if needed
        if pre_ray_start_commands:
            executor.lifecycle_kwargs["postStart"] = {
                "exec": {"command": ["/bin/sh", "-c", "\n".join(pre_ray_start_commands)]}
            }

        # ------------------------------------------------------------------
        # 2.  Build RayCluster spec (via executor).
        # ------------------------------------------------------------------
        cluster_name = f"{self.job_name}-raycluster"
        ray_cluster_body = executor.get_cluster_body(cluster_name)
        ray_cluster_spec = ray_cluster_body.get("spec", {})

        # Ensure consistent workingDir inside all Ray containers so that relative
        # paths in `ray job submit` resolve as expected.
        container_workdir = "/workspace"
        if workdir:
            container_workdir = os.path.join(
                executor.volume_mounts[0]["mountPath"], Path(workdir).name
            )

        def _apply_workdir(pod_template: dict):
            try:
                for c in pod_template["spec"]["containers"]:
                    c["workingDir"] = container_workdir
            except Exception:
                pass  # ignore malformed specs

        if "headGroupSpec" in ray_cluster_spec:
            _apply_workdir(ray_cluster_spec["headGroupSpec"]["template"])

        for w in ray_cluster_spec.get("workerGroupSpecs", []):
            _apply_workdir(w["template"])  # type: ignore[arg-type]

        # ------------------------------------------------------------------
        # 3.  Assemble RayJob CRD manifest
        # ------------------------------------------------------------------
        if runtime_env_yaml and os.path.isfile(Path(runtime_env_yaml)):
            with open(runtime_env_yaml, "r") as f:
                runtime_env_yaml = f.read()

        rayjob_body = {
            "apiVersion": "ray.io/v1",
            "kind": "RayJob",
            "metadata": {
                "name": self.job_name,
                "namespace": namespace,
            },
            "spec": {
                "entrypoint": command,
                "shutdownAfterJobFinishes": True,
                "rayClusterSpec": ray_cluster_spec,
                "runtimeEnvYAML": runtime_env_yaml,
            },
        }

        if dryrun:
            print(yaml.dump(rayjob_body))
            return rayjob_body

        # Create the RayJob CR via Kubernetes API
        try:
            self.api.create_namespaced_custom_object(
                group="ray.io",
                version="v1",
                plural="rayjobs",
                body=rayjob_body,
                namespace=namespace,
            )
            self.status()
        except ApiException as e:
            if e.status == 409:
                raise RuntimeError(f"RayJob '{self.job_name}' already exists: {e.reason}")
            raise RuntimeError(f"Error creating RayJob '{self.job_name}': {e}")


def get_user():
    # Get user for scoping if not provided
    try:
        user = getpass.getuser()
    except Exception:
        # Fallback to environment variables if getpass fails
        user = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"
    # Clean user name for use in pod name (k8s resource naming rules)
    user = re.sub(r"[^a-z0-9\-]", "-", user.lower())
    return user
