import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from invoke.context import Context

from nemo_run.core.packaging.base import Packager


@dataclass(kw_only=True)
class HybridPackager(Packager):
    """
    A packager that combines multiple other packagers into one final archive.
    Each subpackager is mapped to a target directory name, which will become
    the top-level folder under which that packager’s content is placed.
    """

    sub_packagers: Dict[str, Packager] = field(default_factory=dict)

    def package(self, path: Path, job_dir: str, name: str) -> str:
        final_tar_gz = os.path.join(job_dir, f"{name}.tar.gz")
        if os.path.exists(final_tar_gz):
            return final_tar_gz

        # Create an empty tar to append packaged files from each sub-packager
        tmp_tar = final_tar_gz + ".tmp"
        ctx = Context()
        ctx.run(f"tar -cf {tmp_tar} --files-from /dev/null")

        # For each subpackager, run its .package() method and extract to a subfolder
        for folder_name, packager in self.sub_packagers.items():
            subarchive_path = packager.package(path, job_dir, f"{name}_{folder_name}")

            # Create a temp folder, extract subarchive content into it,
            # then add that folder to the final tar under the desired subpath
            tmp_extract_dir = os.path.join(job_dir, f"__extract_{folder_name}")
            os.makedirs(tmp_extract_dir, exist_ok=True)

            ctx.run(f"tar -xf {subarchive_path} -C {tmp_extract_dir}")
            ctx.run(f"tar -rf {tmp_tar} -C {tmp_extract_dir} . --transform='s,^,{folder_name}/,'")
            ctx.run(f"rm -rf {tmp_extract_dir}")
            ctx.run(f"rm {subarchive_path}")

        # Finally, compress the combined tar
        ctx.run(f"gzip -c {tmp_tar} > {final_tar_gz}")
        ctx.run(f"rm {tmp_tar}")

        return final_tar_gz