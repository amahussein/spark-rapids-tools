# Copyright (c) 2025, NVIDIA CORPORATION.
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

"""Implementation of a wrapper to process the results of the core-tool eventlog analysis."""

from dataclasses import dataclass
from typing import override

from spark_rapids_pytools.rapids.qualification import QualificationAsLocal


@dataclass
class QualLocalV2(QualificationAsLocal):
    """
    Implementation of a wrapper to process the results of the core-tool eventlog analysis.
    This class is specifically designed to handle per-app core-tool results.
    """
    description: str = 'This is the localQualification processes per-app core-tool results'

    @override
    def _process_output(self) -> None:
        """
        Process the core-tool output.
        """
        return None
