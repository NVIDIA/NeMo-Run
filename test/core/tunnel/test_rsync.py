"""Tests for the rsync module."""

import unittest
from unittest.mock import Mock, call, patch

from fabric import Connection

from nemo_run.core.tunnel.rsync import rsync


class TestRsync(unittest.TestCase):
    """Test cases for the rsync function."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_connection = Mock(spec=Connection)
        self.mock_connection.user = "testuser"
        self.mock_connection.host = "testhost"
        self.mock_connection.port = 22
        self.mock_connection.connect_kwargs = {}

        # Create a mock for the local command result
        self.mock_result = Mock()
        self.mock_result.command = "rsync command"

        # Set up the connection's local method to return our mock result
        self.mock_connection.local.return_value = self.mock_result

        # Create source and target paths
        self.source = "/local/path"
        self.target = "/remote/path"

    def test_basic_rsync(self):
        """Test basic rsync with minimal parameters."""
        rsync(self.mock_connection, self.source, self.target)

        # Check that mkdir was called
        self.mock_connection.run.assert_called_once_with(f"mkdir -p {self.target}", hide=True)

        # Check that local command was called with correct parameters
        self.mock_connection.local.assert_called_once()
        cmd = self.mock_connection.local.call_args[0][0]

        # Verify command components
        self.assertIn("-p 22", cmd)
        self.assertIn("-pthrvz", cmd)
        self.assertIn(f"{self.source}", cmd)
        self.assertIn(f"testuser@testhost:{self.target}", cmd)

    def test_rsync_with_exclude_string(self):
        """Test rsync with a single exclude string."""
        exclude_pattern = "*.log"
        rsync(self.mock_connection, self.source, self.target, exclude=exclude_pattern)

        cmd = self.mock_connection.local.call_args[0][0]
        self.assertIn(f'--exclude "{exclude_pattern}"', cmd)

    def test_rsync_with_exclude_list(self):
        """Test rsync with a list of exclude patterns."""
        exclude_patterns = ["*.log", "*.tmp", ".git/"]
        rsync(self.mock_connection, self.source, self.target, exclude=exclude_patterns)

        cmd = self.mock_connection.local.call_args[0][0]
        for pattern in exclude_patterns:
            self.assertIn(f'--exclude "{pattern}"', cmd)

    def test_rsync_with_exclude_generator(self):
        """Test rsync with a generator of exclude patterns."""
        # Using a generator expression instead of a list
        exclude_patterns = ["*.log", "*.tmp", ".git/"]
        rsync(self.mock_connection, self.source, self.target, exclude=exclude_patterns)

        cmd = self.mock_connection.local.call_args[0][0]
        for pattern in ["*.log", "*.tmp", ".git/"]:
            self.assertIn(f'--exclude "{pattern}"', cmd)

    def test_rsync_with_delete(self):
        """Test rsync with delete flag enabled."""
        rsync(self.mock_connection, self.source, self.target, delete=True)

        cmd = self.mock_connection.local.call_args[0][0]
        self.assertIn("--delete", cmd)

    def test_rsync_without_delete(self):
        """Test rsync with delete flag disabled."""
        rsync(self.mock_connection, self.source, self.target, delete=False)

        cmd = self.mock_connection.local.call_args[0][0]
        self.assertNotIn("--delete", cmd)

    def test_rsync_with_custom_ssh_opts(self):
        """Test rsync with custom SSH options."""
        ssh_opts = "-o Compression=yes"
        rsync(self.mock_connection, self.source, self.target, ssh_opts=ssh_opts)

        cmd = self.mock_connection.local.call_args[0][0]
        self.assertIn(ssh_opts, cmd)

    def test_rsync_with_custom_rsync_opts(self):
        """Test rsync with custom rsync options."""
        rsync_opts = "--checksum"
        rsync(self.mock_connection, self.source, self.target, rsync_opts=rsync_opts)

        cmd = self.mock_connection.local.call_args[0][0]
        self.assertIn(rsync_opts, cmd)

    def test_rsync_with_ssh_keys(self):
        """Test rsync with SSH key files."""
        self.mock_connection.connect_kwargs = {"key_filename": "/path/to/key.pem"}
        rsync(self.mock_connection, self.source, self.target)

        cmd = self.mock_connection.local.call_args[0][0]
        self.assertIn("-i /path/to/key.pem", cmd)

    def test_rsync_with_multiple_ssh_keys(self):
        """Test rsync with multiple SSH key files."""
        self.mock_connection.connect_kwargs = {
            "key_filename": ["/path/to/key1.pem", "/path/to/key2.pem"]
        }
        rsync(self.mock_connection, self.source, self.target)

        cmd = self.mock_connection.local.call_args[0][0]
        self.assertIn("-i /path/to/key1.pem -i /path/to/key2.pem", cmd)

    def test_rsync_with_ipv6_host(self):
        """Test rsync with IPv6 host."""
        self.mock_connection.host = "2001:db8::1"
        rsync(self.mock_connection, self.source, self.target)

        cmd = self.mock_connection.local.call_args[0][0]
        self.assertIn(f"[testuser@{self.mock_connection.host}]", cmd)

    def test_rsync_with_ipv4_host(self):
        """Test rsync with IPv4 host."""
        self.mock_connection.host = "192.168.1.1"
        rsync(self.mock_connection, self.source, self.target)

        cmd = self.mock_connection.local.call_args[0][0]
        self.assertIn(f"testuser@{self.mock_connection.host}", cmd)
        self.assertNotIn(f"[testuser@{self.mock_connection.host}]", cmd)

    def test_rsync_with_show_output(self):
        """Test rsync with output shown."""
        rsync(self.mock_connection, self.source, self.target, hide_output=False)

        self.mock_connection.run.assert_called_once_with(f"mkdir -p {self.target}", hide=False)
        self.mock_connection.local.assert_called_once()
        self.assertEqual(self.mock_connection.local.call_args[1]["hide"], False)

    @patch("nemo_run.core.tunnel.rsync.logger")
    def test_rsync_success_logging(self, mock_logger):
        """Test that successful rsync execution is logged."""
        rsync(self.mock_connection, self.source, self.target)

        # Verify info logs
        mock_logger.info.assert_has_calls(
            [
                call(f"rsyncing {self.source} to {self.target} ..."),
                call(f"Successfully ran `{self.mock_result.command}`"),
            ]
        )

    def test_rsync_failure(self):
        """Test that rsync failure raises an exception."""
        # Make local command return False to simulate failure
        self.mock_connection.local.return_value = False

        with self.assertRaises(RuntimeError) as context:
            rsync(self.mock_connection, self.source, self.target)

        self.assertEqual("rsync failed", str(context.exception))


if __name__ == "__main__":
    unittest.main()
