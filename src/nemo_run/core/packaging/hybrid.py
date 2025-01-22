import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

from invoke.context import Context

from nemo_run.core.packaging.base import Packager


@dataclass(kw_only=True)
class HybridPackager(Packager):
    """
    A packager that combines multiple other packagers into one final archive.
    Each subpackager is mapped to a target directory name, which will become
    the top-level folder under which that packager’s content is placed.
    """

    sub_packagers: Dict[str, Union[Packager, List[Packager]]] = field(default_factory=dict)

    def package(self, path: Path, job_dir: str, name: str) -> str:
        final_tar_gz = os.path.join(job_dir, f"{name}.tar.gz")
        if os.path.exists(final_tar_gz):
            return final_tar_gz

        # Create an empty tar to append packaged files from each sub-packager
        tmp_tar = final_tar_gz + ".tmp"
        ctx = Context()
        ctx.run(f"tar -cf {tmp_tar} --files-from /dev/null")

        # Defer deletion of temporary files until all subpackagers have been processed
        subarchive_list = set([])
        tmp_extract_dir_list = set([])

        # For each subpackager, run its .package() method and extract to a subfolder
        for folder_name, packagers in self.sub_packagers.items():
            if not isinstance(packagers, list):
                packagers = [packagers]

            for packager in packagers:
                subarchive_path = packager.package(path, job_dir, f"{name}_{folder_name}")
                subarchive_list.add(subarchive_path)

                # Create a temp folder, extract subarchive content into it,
                # then add that folder to the final tar under the desired subpath
                tmp_extract_dir = os.path.join(job_dir, f"__extract_{folder_name}")
                tmp_extract_dir_list.add(tmp_extract_dir)
                os.makedirs(tmp_extract_dir, exist_ok=True)

                ctx.run(f"tar -xf {subarchive_path} -C {tmp_extract_dir}")

                # If a folder name is provided, add the content under that folder
                if folder_name != '':
                    ctx.run(f"tar -rf {tmp_tar} -C {tmp_extract_dir} . --transform='s,^,{folder_name}/,'")
                else:
                    # Otherwise, add the content directly to the root of the tar
                    ctx.run(f"tar -rf {tmp_tar} -C {tmp_extract_dir} .")

        for tmp_extract_dir in tmp_extract_dir_list:
            ctx.run(f"rm -rf {tmp_extract_dir}")

        for subarchive_path in subarchive_list:
            ctx.run(f"rm {subarchive_path}")

        # Finally, compress the combined tar
        ctx.run(f"gzip -c {tmp_tar} > {final_tar_gz}")
        ctx.run(f"rm {tmp_tar}")

        return final_tar_gz
