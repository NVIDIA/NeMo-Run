#!/usr/bin/env python3
"""
Unified script for NeMo-Run KubeRay RBAC management.

This script can:
1. Test Kubernetes permissions required for KubeRay operations
2. Apply RBAC manifests to grant necessary permissions
3. Verify CRD installations

Usage:
  python kuberay_rbac.py test -n namespace
  python kuberay_rbac.py apply -n namespace -u username
"""

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

from rich.console import Console


console = Console()


def print_info(message: str):
    console.print(f"[INFO] {message}", style="green")


def print_error(message: str):
    console.print(f"[ERROR] {message}", style="red")


def print_warning(message: str):
    console.print(f"[WARNING] {message}", style="yellow")


def print_debug(message: str):
    console.print(f"[DEBUG] {message}", style="blue")


def get_default_namespace() -> str:
    """Get the default namespace from kubectl config."""
    try:
        result = subprocess.run(
            [
                "kubectl",
                "config",
                "view",
                "--minify",
                "-o",
                "jsonpath={.contexts[0].context.namespace}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    # Fallback to "default" if we can't get it from config
    return "default"


@dataclass
class PermissionCheck:
    """Represents a single permission check."""

    verb: str
    resource: str
    namespace: str = "default"
    api_group: str = ""
    description: str = ""
    required_for: str = ""


class KubeRayRBAC:
    """Unified class for KubeRay RBAC management."""

    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self.failed_checks: List[PermissionCheck] = []
        self.passed_checks: List[PermissionCheck] = []

    def check_prerequisites(self) -> bool:
        """Check if kubectl is available and cluster is accessible."""
        # Check if kubectl is available
        try:
            subprocess.run(["kubectl", "version", "--client"], capture_output=True, check=True)
        except Exception:
            print_error("kubectl is not installed or not in PATH")
            return False

        # Check if we can connect to a cluster
        try:
            subprocess.run(["kubectl", "get", "pods"], capture_output=True, check=True)
        except Exception:
            print_error("Cannot connect to Kubernetes cluster. Please check your kubeconfig.")
            return False

        return True

    def get_required_permissions(self) -> List[PermissionCheck]:
        """Define all permissions required by the KubeRay implementation."""
        permissions = []

        # RayCluster CRD permissions
        for verb in ["create", "get", "list", "watch", "update", "patch", "delete"]:
            permissions.append(
                PermissionCheck(
                    verb=verb,
                    resource="rayclusters",
                    api_group="ray.io",
                    namespace=self.namespace,
                    description=f"{verb.capitalize()} RayCluster custom resources",
                    required_for="KubeRayCluster lifecycle management",
                )
            )

        # RayCluster status subresource
        permissions.append(
            PermissionCheck(
                verb="get",
                resource="rayclusters/status",
                api_group="ray.io",
                namespace=self.namespace,
                description="Get RayCluster status",
                required_for="KubeRayCluster.status() method",
            )
        )

        # RayJob CRD permissions
        for verb in ["create", "get", "list", "watch", "delete"]:
            permissions.append(
                PermissionCheck(
                    verb=verb,
                    resource="rayjobs",
                    api_group="ray.io",
                    namespace=self.namespace,
                    description=f"{verb.capitalize()} RayJob custom resources",
                    required_for="KubeRayJob lifecycle management",
                )
            )

        # Pod permissions
        pod_verbs = ["get", "list", "watch", "create", "delete"]
        for verb in pod_verbs:
            permissions.append(
                PermissionCheck(
                    verb=verb,
                    resource="pods",
                    namespace=self.namespace,
                    description=f"{verb.capitalize()} pods",
                    required_for="Pod management, data-mover operations",
                )
            )

        # Pod exec permission
        permissions.append(
            PermissionCheck(
                verb="create",
                resource="pods/exec",
                namespace=self.namespace,
                description="Execute commands in pods",
                required_for="Data sync, debugging, exec into Ray pods",
            )
        )

        # Pod logs permission
        permissions.append(
            PermissionCheck(
                verb="get",
                resource="pods/log",
                namespace=self.namespace,
                description="Read pod logs",
                required_for="KubeRayJob.logs() method",
            )
        )

        # Service permissions
        service_verbs = ["get", "list", "watch"]
        for verb in service_verbs:
            permissions.append(
                PermissionCheck(
                    verb=verb,
                    resource="services",
                    namespace=self.namespace,
                    description=f"{verb.capitalize()} services",
                    required_for="Service discovery, port-forwarding",
                )
            )

        # Port-forward permission
        permissions.append(
            PermissionCheck(
                verb="create",
                resource="pods/portforward",
                namespace=self.namespace,
                description="Create port-forward to pods",
                required_for="KubeRayCluster.port_forward() method",
            )
        )

        # PersistentVolumeClaim permissions
        for verb in ["get", "list"]:
            permissions.append(
                PermissionCheck(
                    verb=verb,
                    resource="persistentvolumeclaims",
                    namespace=self.namespace,
                    description=f"{verb.capitalize()} PersistentVolumeClaims",
                    required_for="Volume management for workdir sync",
                )
            )

        return permissions

    def check_permission(self, perm: PermissionCheck) -> bool:
        """Check a single permission using kubectl auth can-i."""
        cmd = ["kubectl", "auth", "can-i", perm.verb, perm.resource]

        if perm.namespace:
            cmd.extend(["-n", perm.namespace])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            return result.returncode == 0
        except Exception as e:
            print_error(f"Error checking permission: {e}")
            return False

    def check_crds_installed(self) -> bool:
        """Check if KubeRay CRDs are installed."""
        print_info("Checking if KubeRay CRDs are installed...")
        all_installed = True

        for crd in ["rayclusters.ray.io", "rayjobs.ray.io"]:
            try:
                result = subprocess.run(
                    ["kubectl", "get", "crd", crd], capture_output=True, check=False
                )
                if result.returncode == 0:
                    console.print(f"  ✓ {crd} is installed", style="green")
                else:
                    console.print(f"  ✗ {crd} is NOT installed", style="red")
                    all_installed = False
            except Exception as e:
                console.print(f"  ✗ Error checking {crd}: {e}", style="red")
                all_installed = False

        if not all_installed:
            print_warning(
                "Install KubeRay operator: https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html"
            )

        return all_installed

    def test_permissions(self, check_crd: bool = False) -> Tuple[int, int]:
        """Test all required permissions."""
        if not self.check_prerequisites():
            return 0, 0

        if check_crd:
            self.check_crds_installed()

        permissions = self.get_required_permissions()
        total = len(permissions)

        console.print(
            f"\nChecking {total} Kubernetes permissions for namespace '{self.namespace}'...\n"
        )
        console.print("-" * 80)

        for i, perm in enumerate(permissions, 1):
            resource_str = f"{perm.api_group}/{perm.resource}" if perm.api_group else perm.resource
            check_str = f"{perm.verb} {resource_str}"

            has_permission = self.check_permission(perm)

            if has_permission:
                self.passed_checks.append(perm)
                status = "[green]✓ PASS[/green]"
            else:
                self.failed_checks.append(perm)
                status = "[red]✗ FAIL[/red]"

            console.print(f"{status} [{i:3d}/{total}] {check_str:35s} | {perm.description}")

            if not has_permission and perm.required_for:
                console.print(f"            Required for: {perm.required_for}")

        console.print("-" * 80)
        self.print_summary()

        return len(self.passed_checks), len(self.failed_checks)

    def print_summary(self):
        """Print a summary of the permission checks."""
        total = len(self.passed_checks) + len(self.failed_checks)
        passed = len(self.passed_checks)
        failed = len(self.failed_checks)

        console.print("\nSummary:")
        console.print(f"  Total checks: {total}")
        console.print(f"  Passed: {passed}", style="green")
        console.print(f"  Failed: {failed}", style="red")

        if self.failed_checks:
            console.print("\nMissing Permissions:", style="red")

            # Group by required_for
            grouped: Dict[str, List[PermissionCheck]] = {}
            for perm in self.failed_checks:
                key = perm.required_for or "General"
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(perm)

            for feature, perms in grouped.items():
                console.print(f"\n  {feature}:")
                for perm in perms:
                    resource_str = (
                        f"{perm.api_group}/{perm.resource}" if perm.api_group else perm.resource
                    )
                    console.print(f"    - {perm.verb} {resource_str}")

            console.print("\nTo grant these permissions:", style="yellow")
            console.print("  1. Use 'kuberay_rbac.py apply' to create RBAC resources")
            console.print("  2. Contact your Kubernetes administrator")
            console.print("  3. Ensure the KubeRay CRDs are installed in the cluster")
        else:
            console.print(
                "\nAll permission checks passed! You have the required permissions to use KubeRay.",
                style="green",
            )

    def generate_rbac_manifest(self) -> str:
        """Generate RBAC manifest YAML."""
        return """# RBAC manifest for NeMo-Run KubeRay operations
# This creates a Role with all necessary permissions and binds it to a user

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: nemo-run-kuberay-role
  namespace: NAMESPACE_NAME
rules:
# RayCluster CRD permissions
- apiGroups:
  - "ray.io"
  resources:
  - "rayclusters"
  verbs:
  - "create"
  - "get"
  - "list"
  - "watch"
  - "update"
  - "patch"
  - "delete"

# RayCluster status subresource
- apiGroups:
  - "ray.io"
  resources:
  - "rayclusters/status"
  verbs:
  - "get"

# RayJob CRD permissions
- apiGroups:
  - "ray.io"
  resources:
  - "rayjobs"
  verbs:
  - "create"
  - "get"
  - "list"
  - "watch"
  - "delete"

# Pod permissions
- apiGroups:
  - ""
  resources:
  - "pods"
  verbs:
  - "get"
  - "list"
  - "watch"
  - "create"
  - "delete"

# Pod exec permission
- apiGroups:
  - ""
  resources:
  - "pods/exec"
  verbs:
  - "create"

# Pod logs permission
- apiGroups:
  - ""
  resources:
  - "pods/log"
  verbs:
  - "get"

# Pod port-forward permission
- apiGroups:
  - ""
  resources:
  - "pods/portforward"
  verbs:
  - "create"

# Service permissions
- apiGroups:
  - ""
  resources:
  - "services"
  verbs:
  - "get"
  - "list"
  - "watch"

# PersistentVolumeClaim permissions
- apiGroups:
  - ""
  resources:
  - "persistentvolumeclaims"
  verbs:
  - "get"
  - "list"

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: nemo-run-kuberay-rolebinding
  namespace: NAMESPACE_NAME
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: nemo-run-kuberay-role
subjects:
- kind: User
  name: USERNAME
  apiGroup: rbac.authorization.k8s.io"""

    def apply_rbac(self, username: str, dry_run: bool = False) -> bool:
        """Apply RBAC manifest to the cluster."""
        if not self.check_prerequisites():
            return False

        print_info("Applying KubeRay RBAC permissions...")
        print_info(f"Namespace: {self.namespace}")
        print_info(f"Username: {username}")

        # Check if namespace exists
        try:
            result = subprocess.run(
                ["kubectl", "get", "namespace", self.namespace], capture_output=True, check=False
            )
            if result.returncode != 0:
                print_warning(f"Namespace '{self.namespace}' does not exist. Creating it...")
                if not dry_run:
                    subprocess.run(["kubectl", "create", "namespace", self.namespace], check=True)
        except Exception as e:
            print_error(f"Error checking/creating namespace: {e}")
            return False

        # Generate manifest with substitutions
        manifest_content = self.generate_rbac_manifest()
        manifest_content = manifest_content.replace("NAMESPACE_NAME", self.namespace)
        manifest_content = manifest_content.replace("USERNAME", username)

        if dry_run:
            console.print("\nGenerated RBAC manifest:")
            console.print("-" * 40)
            console.print(manifest_content)
            console.print("-" * 40)
            return True

        # Apply the manifest
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                f.write(manifest_content)
                temp_file = f.name

            print_info("Applying RBAC manifest...")
            subprocess.run(["kubectl", "apply", "-f", temp_file], check=True)

            # Clean up temp file
            os.unlink(temp_file)

            print_info("RBAC permissions successfully applied!")

        except Exception as e:
            print_error(f"Failed to apply RBAC manifest: {e}")
            return False

        return True

    def test_user_permissions(self, username: str):
        """Test a few key permissions for a specific user."""
        tests = [
            ("create", "rayclusters", "Can create RayCluster resources"),
            ("create", "rayjobs", "Can create RayJob resources"),
            ("create", "pods/exec", "Can execute commands in pods"),
        ]

        for verb, resource, description in tests:
            try:
                result = subprocess.run(
                    [
                        "kubectl",
                        "auth",
                        "can-i",
                        verb,
                        resource,
                        f"--as={username}",
                        "-n",
                        self.namespace,
                    ],
                    capture_output=True,
                    check=False,
                )
                if result.returncode == 0:
                    console.print(f"  ✓ {description}", style="green")
                else:
                    console.print(f"  ✗ {description}", style="red")
            except Exception as e:
                console.print(f"  ✗ Error testing {resource}: {e}", style="red")


def show_help():
    """Display help message using rich console."""
    help_text = """
[bold]usage:[/bold] kuberay_rbac.py [-h] {test,apply} ...

Unified script for NeMo-Run KubeRay RBAC management

[bold]positional arguments:[/bold]
  {test,apply}  Available commands
    test        Test Kubernetes permissions
    apply       Apply RBAC manifest

[bold]options:[/bold]
  -h, --help    show this help message and exit

[green]FEATURES:[/green]
  • Test permissions: Verify current user has all required Kubernetes permissions
  • Apply RBAC: Create and apply RBAC manifests for users
  • CRD verification: Check if KubeRay Custom Resource Definitions are installed
  • Color-coded output: Easy-to-read status indicators
  • Dry-run support: Generate manifests without applying them

[green]PERMISSIONS MANAGED:[/green]
  The script manages [yellow]27 Kubernetes permissions[/yellow] across:

  [blue]RayCluster CRD Operations:[/blue]
    • Create, read, update, patch, delete RayCluster resources
    • Get RayCluster status

  [blue]RayJob CRD Operations:[/blue]
    • Create, read, list, watch, delete RayJob resources

  [blue]Core Kubernetes Resources:[/blue]
    • Pods: create, get, list, watch, delete
    • Pod exec: execute commands in pods (for debugging/data sync)
    • Pod logs: read pod logs (for job monitoring)
    • Pod port-forward: create port-forwards (for Ray dashboard access)
    • Services: get, list, watch (for service discovery)
    • PersistentVolumeClaims: get, list (for volume discovery)

[green]EXAMPLES:[/green]
  [yellow]# Test permissions in default namespace[/yellow]
  python kuberay_rbac.py test

  [yellow]# Test permissions in specific namespace with CRD check[/yellow]
  python kuberay_rbac.py test -n my-namespace --check-crd

  [yellow]# Apply RBAC for a user in default namespace[/yellow]
  python kuberay_rbac.py apply -u myusername

  [yellow]# Apply RBAC in specific namespace[/yellow]
  python kuberay_rbac.py apply -n my-namespace -u myusername

  [yellow]# Generate RBAC manifest without applying (dry-run)[/yellow]
  python kuberay_rbac.py apply -n my-namespace -u myusername --dry-run

[green]SAMPLE OUTPUT (test command):[/green]
  [green][INFO][/green] Checking if KubeRay CRDs are installed...
    [green]✓[/green] rayclusters.ray.io is installed
    [green]✓[/green] rayjobs.ray.io is installed

  Checking 27 Kubernetes permissions for namespace 'my-namespace'...
  --------------------------------------------------------------------------------
  [green]✓ PASS[/green] [  1/27] create ray.io/rayclusters   | Create RayCluster custom resources
  [green]✓ PASS[/green] [  2/27] get ray.io/rayclusters      | Get RayCluster custom resources
  [red]✗ FAIL[/red] [  3/27] list ray.io/rayclusters     | List RayCluster custom resources
              Required for: KubeRayCluster lifecycle management
  ...

  Summary:
    Total checks: 27
    [green]Passed: 23[/green]
    [red]Failed: 4[/red]

[green]SAMPLE OUTPUT (apply command):[/green]
  [green][INFO][/green] Applying KubeRay RBAC permissions...
  [green][INFO][/green] Namespace: my-namespace
  [green][INFO][/green] Username: myuser
  [green][INFO][/green] Applying RBAC manifest...
  role.rbac.authorization.k8s.io/nemo-run-kuberay-role created
  rolebinding.rbac.authorization.k8s.io/nemo-run-kuberay-rolebinding created
  [green][INFO][/green] RBAC permissions successfully applied!
  [green][INFO][/green] Testing key permissions for user 'myuser'...
    [green]✓[/green] Can create RayCluster resources
    [green]✓[/green] Can create RayJob resources
    [green]✓[/green] Can execute commands in pods

[green]RBAC RESOURCES CREATED:[/green]
  1. [blue]Role:[/blue] 'nemo-run-kuberay-role' - Contains all required permissions
  2. [blue]RoleBinding:[/blue] 'nemo-run-kuberay-rolebinding' - Binds the role to the specified user

[green]TROUBLESHOOTING:[/green]
  [blue]kubectl not found:[/blue]
    [red][ERROR][/red] kubectl is not installed or not in PATH
    → Install kubectl and ensure it's in your PATH

  [blue]Cannot connect to cluster:[/blue]
    [red][ERROR][/red] Cannot connect to Kubernetes cluster. Please check your kubeconfig.
    → Verify your kubeconfig with [yellow]'kubectl get pods'[/yellow]

  [blue]CRDs not installed:[/blue]
    [yellow][WARNING][/yellow] Install KubeRay operator: https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html
    → Install the KubeRay operator before using NeMo-Run

  [blue]Permission denied when applying RBAC:[/blue]
    → You need cluster-admin or sufficient permissions to create Roles and RoleBindings

[green]REQUIREMENTS:[/green]
  • Python 3.10+
  • kubectl installed and configured
  • Active connection to a Kubernetes cluster
    """
    console.print(help_text)


def main():
    """Main entry point with argparse subcommands."""
    parser = argparse.ArgumentParser(
        description="Unified script for NeMo-Run KubeRay RBAC management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )

    # Add custom help argument
    parser.add_argument("-h", "--help", action="store_true", help="Show this help message and exit")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Get default namespace for help text
    default_ns = get_default_namespace()

    # Test subcommand
    test_parser = subparsers.add_parser("test", help="Test Kubernetes permissions")
    test_parser.add_argument(
        "-n",
        "--namespace",
        default=default_ns,
        help=f"Kubernetes namespace to check permissions for (default: {default_ns})",
    )
    test_parser.add_argument(
        "--check-crd", action="store_true", help="Also check if KubeRay CRDs are installed"
    )

    # Apply subcommand
    apply_parser = subparsers.add_parser("apply", help="Apply RBAC manifest")
    apply_parser.add_argument(
        "-n",
        "--namespace",
        default=default_ns,
        help=f"Kubernetes namespace to apply RBAC in (default: {default_ns})",
    )
    apply_parser.add_argument(
        "-u", "--username", required=True, help="Username to grant permissions to"
    )
    apply_parser.add_argument(
        "--dry-run", action="store_true", help="Generate manifest without applying it"
    )

    args = parser.parse_args()

    # Handle custom help
    if args.help:
        show_help()
        sys.exit(0)

    if not args.command:
        show_help()
        sys.exit(1)

    rbac_manager = KubeRayRBAC(namespace=args.namespace)

    if args.command == "test":
        passed, failed = rbac_manager.test_permissions(check_crd=args.check_crd)
        sys.exit(0 if failed == 0 else 1)

    elif args.command == "apply":
        success = rbac_manager.apply_rbac(username=args.username, dry_run=args.dry_run)
        if success and not args.dry_run:
            print_info("\nTo verify the permissions, run:")
            console.print(f"  python kuberay_rbac.py test -n {args.namespace}")
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
