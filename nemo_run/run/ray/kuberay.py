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

import logging
import time
from typing import Any, Optional

import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from nemo_run.core.execution.kuberay import GROUP, PLURAL, VERSION, KubeRayExecutor

logger = logging.getLogger(__name__)


class KubeRayCluster:
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
        logger.info(f"Getting Ray cluster '{name}' in namespace '{k8s_namespace}'")

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
                logger.error(f"Ray cluster '{name}' not found in namespace '{k8s_namespace}': {e}")
                return None
            else:
                logger.error(
                    f"Error fetching Ray cluster '{name}' in namespace '{k8s_namespace}': {e}"
                )
                return None

    def get_ray_cluster_status(
        self,
        name: str,
        k8s_namespace: str = "default",
        timeout: int = 60,
        delay_between_attempts: int = 5,
    ) -> Any:
        logger.info(
            f"Getting Ray cluster status for '{name}' in namespace '{k8s_namespace}', timeout: {timeout}s, delay: {delay_between_attempts}s"
        )

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
                    logger.error(
                        f"Ray cluster '{name}' status fetch failed: resource not found: {e}"
                    )
                    return None
                else:
                    logger.error(
                        f"Error fetching status for Ray cluster '{name}' in namespace '{k8s_namespace}': {e}"
                    )
                    return None

            if "status" in resource and resource["status"]:
                return resource["status"]
            else:
                logger.info(f"Ray cluster '{name}' status not set yet, waiting...")
                time.sleep(delay_between_attempts)
                timeout -= delay_between_attempts

        logger.info(f"Ray cluster '{name}' status not set yet, timing out...")
        return None

    def wait_until_ray_cluster_running(
        self,
        name: str,
        executor: KubeRayExecutor,
        timeout: int = 60,
        delay_between_attempts: int = 5,
        k8s_namespace: Optional[str] = None,
    ) -> bool:
        namespace = k8s_namespace or executor.namespace
        logger.info(
            f"Waiting until Ray cluster '{name}' is running in namespace '{namespace}', timeout: {timeout}s, delay: {delay_between_attempts}s"
        )

        while timeout > 0:
            status = self.get_ray_cluster_status(
                name, k8s_namespace or executor.namespace, timeout, delay_between_attempts
            )
            if not status:
                logger.info(f"Ray cluster '{name}' status could not be retrieved")
                return False

            # TODO: once we add State to Status, we should check for that as well  <if status and status["state"] == "Running":>
            if status and status["head"] and status["head"]["serviceIP"]:
                logger.info(f"Ray cluster '{name}' is running")
                return True

            logger.info(
                f"Ray cluster '{name}' status is not running yet, current status: {status.get('state', 'unknown')}"
            )
            time.sleep(delay_between_attempts)
            timeout -= delay_between_attempts

        logger.info(f"Ray cluster '{name}' status is not running yet, timing out...")
        return False

    def create_ray_cluster(
        self,
        name: str,
        executor: KubeRayExecutor,
        pre_ray_start_commands: Optional[list[str]] = None,
        dryrun: bool = False,
        k8s_namespace: Optional[str] = None,
    ) -> Any:
        namespace = k8s_namespace or executor.namespace
        logger.info(f"Creating Ray cluster '{name}' in namespace '{namespace}'")

        if pre_ray_start_commands:
            k8s_pre_ray_start_commands = "\n".join(pre_ray_start_commands)
            executor.lifecycle_kwargs["postStart"] = {
                "exec": {
                    "command": [
                        "/bin/sh",
                        "-c",
                        k8s_pre_ray_start_commands,
                    ]
                }
            }

        body = executor.get_cluster_body(name)

        if dryrun:
            print(yaml.dump(body))
            return

        try:
            resource: Any = self.api.create_namespaced_custom_object(
                group=GROUP,
                version=VERSION,
                plural=PLURAL,
                body=body,
                namespace=k8s_namespace or executor.namespace,
            )
            return resource
        except ApiException as e:
            if e.status == 409:
                logger.error(f"Ray cluster '{name}' already exists: {e.reason}")
                return None
            else:
                logger.error(f"Error creating Ray cluster '{name}' in namespace '{namespace}': {e}")
                return None

    def schedule_ray_job(
        self,
        name: str,
        executor: KubeRayExecutor,
        command: str,
        workdir: str,
    ):
        raise NotImplementedError("KubeRay does not support scheduling jobs")

    def delete_ray_cluster(
        self,
        name: str,
        executor: KubeRayExecutor,
        k8s_namespace: Optional[str] = None,
        wait: bool = False,
        timeout: int = 300,
        poll_interval: int = 5,
    ) -> Optional[bool]:
        namespace = k8s_namespace or executor.namespace
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

            logger.info(f"Waiting for Ray cluster '{name}' and its pods to be fully deleted...")
            start_time = time.time()
            cluster_deleted = False

            # Wait for both the CR and all pods to be deleted
            while time.time() - start_time < timeout:
                # Check if CR still exists
                if not cluster_deleted:
                    try:
                        cluster = self.get_ray_cluster(name, namespace)
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
                f"Timed out waiting for Ray cluster '{name}' to be fully deleted after {timeout} seconds"
            )

            # Check final state
            try:
                cluster_exists = self.get_ray_cluster(name, namespace) is not None
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

    def patch_ray_cluster(
        self,
        name: str,
        ray_patch: Any,
        executor: KubeRayExecutor,
        k8s_namespace: Optional[str] = None,
    ) -> Any:
        namespace = k8s_namespace or executor.namespace
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
        name: str,
        port: int,
        target_port: int,
        executor: KubeRayExecutor,
        wait: bool = False,
    ):
        import queue
        import subprocess
        import threading
        import time

        # Get cluster details
        cluster = self.get_ray_cluster(name, executor.namespace or "default")
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
