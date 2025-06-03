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

"""Implementation of handlers that manage a single Spark app result."""

from dataclasses import dataclass
from functools import cached_property

from spark_rapids_tools.enums import AppCoreStatusEnum


@dataclass
class BaseAppDescriptor:
    """
    Holds the base information loaded from the core-tool output. This is typically
    same information extracted from the app_status in the core tool output.
    """
    eventlog_path: str
    status: AppCoreStatusEnum


    @cached_property
    def is_processable(self) -> bool:
        """
        We only process apps that were successfully analyzed by the core-tool.
        :return: bool to indicate true if the app should be processed in the user tool
        """
        return self.status == AppCoreStatusEnum.SUCCESS

    def __post_init__(self):
        pass


@dataclass
class AppDescriptor(BaseAppDescriptor):
    """
    Represents a successful app results that was processed by the core-tool.
    An instant of that class is created and it will be processed by the heuristics and QualX and
    the report generator.
    """
    app_id: str
    app_name: str
    attempt_id: int

    @cached_property
    def uuid(self) -> str:
        """
        Create a unique identifier for the app. This is used to identify the app in report output
        folders.
        In addition, it should be used to for joining between results from different paths.
        For now, we typically use the app_id as a unique_id.
        However, in the future, this representation could change when we support multiple attempts.
        In the latter case, the attempt_id should become part of the app identification.
        :return: a unique identifier for app.
        """
        return f"{self.app_id}"

    def __post_init__(self):
        super().__post_init__()
