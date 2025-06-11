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

"""Hold the context for a single application to be processed by the tools engine"""

from dataclasses import dataclass
from typing import Optional, List

import pandas as pd

@dataclass
class AppContext:
    """
    doc string
    """
    app_name: str
    app_id: str
    attempt_id: int
    eventlog_path: str
    result_path: str

    def get_id(self) -> str:
        """
        Represents a unique ID for the current application object.
        When supporting multiple attempt, this method would return a composite key of app_id and
        attempt_id
        :return: the UUID for the application
        """
        return self.app_id

    def load_core_data(self, file_label: str, cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        doc string
        :param file_label:
        :param cols:
        :return:
        """
        # here we should use a loader that will be working by label
        # for example we have raw_metrics, we have qual_metrics, we have turning..etc
        return None
