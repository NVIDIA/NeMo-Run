import filecmp
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemo_run.core.packaging.base import Packager
from nemo_run.core.packaging.hybrid import HybridPackager
from test.conftest import MockContext


@pytest.fixture
def mock_subpackager_one(tmp_path) -> Packager:
    """
    Creates a mocked Packager that packages a single file named file1.txt.
    """
    mock_packager = MagicMock(spec=Packager)
    # Prepare a small file to tar
    file_path = tmp_path / "file1.txt"
    file_path.write_text("Content from packager one")

    tar_path = str(tmp_path / "packager_one.tar.gz")
    subprocess.run(["tar", "-czf", tar_path, "-C", str(tmp_path), "file1.txt"], check=True)

    # Make the package() call return the path to this tar
    mock_packager.package.return_value = tar_path
    return mock_packager


@pytest.fixture
def mock_subpackager_two(tmp_path) -> Packager:
    """
    Creates a mocked Packager that packages a single file named file2.txt.
    """
    mock_packager = MagicMock(spec=Packager)
    # Prepare a small file to tar
    file_path = tmp_path / "file2.txt"
    file_path.write_text("Content from packager two")

    tar_path = str(tmp_path / "packager_two.tar.gz")
    subprocess.run(["tar", "-czf", tar_path, "-C", str(tmp_path), "file2.txt"], check=True)

    mock_packager.package.return_value = tar_path
    return mock_packager


@patch("nemo_run.core.packaging.hybrid.Context", MockContext)
def test_hybrid_packager(mock_subpackager_one, mock_subpackager_two, tmp_path):
    hybrid = HybridPackager(
        sub_packagers={
            "1": mock_subpackager_one,
            "2": mock_subpackager_two,
        }
    )
    with tempfile.TemporaryDirectory() as job_dir:
        output_tar = hybrid.package(Path(tmp_path), job_dir, "hybrid_test")

        assert os.path.exists(output_tar)

        # Extract the resulting tar to verify contents
        extract_dir = os.path.join(job_dir, "hybrid_extracted")
        os.makedirs(extract_dir, exist_ok=True)
        subprocess.run(["tar", "-xzf", output_tar, "-C", extract_dir], check=True)

        # Compare subfolder "1" for file1.txt
        cmp = filecmp.dircmp(
            os.path.dirname(mock_subpackager_one.package.return_value),
            os.path.join(extract_dir, "1"),
        )
        assert not cmp.diff_files

        # Compare subfolder "2" for file2.txt
        cmp = filecmp.dircmp(
            os.path.dirname(mock_subpackager_two.package.return_value),
            os.path.join(extract_dir, "2"),
        )
        assert not cmp.diff_files