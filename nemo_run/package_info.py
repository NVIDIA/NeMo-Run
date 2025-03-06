# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from packaging.version import Version

__version__ = "0.2.0rc0.dev0"

MAJOR = Version(__version__).major
MINOR = Version(__version__).minor
PATCH = 3082
if pre := Version(__version__).pre:
    PRE_RELEASE = "".join(map(str, pre))
else:
    PRE_RELEASE = ""

DEV = Version(__version__).dev

__package_name__ = "nemo_curator"
__contact_names__ = "NVIDIA"
__contact_emails__ = "nemo-toolkit@nvidia.com"
__homepage__ = "https://docs.nvidia.com/nemo-framework/user-guide/latest/datacuration/index.html"
__repository_url__ = "https://github.com/NVIDIA/NeMo-Curator"
__download_url__ = "https://github.com/NVIDIA/NeMo-Curator/releases"
__description__ = (
    "NeMo Curator - Scalable Data Preprocessing Tool for Training Large Language Models"
)
__license__ = "Apache2"
__keywords__ = "deep learning, machine learning, gpu, NLP, NeMo, nvidia, pytorch, torch, language, preprocessing, LLM, large language model"
