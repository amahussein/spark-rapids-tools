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

"""Implementation of handlers that manage and process the output of the core-tool module."""

from dataclasses import dataclass, field
from functools import cached_property
from logging import Logger
from typing import List

import pandas as pd

from spark_rapids_pytools.common.utilities import ToolLogging
from spark_rapids_tools import CspPathT
from spark_rapids_tools.tools.core.app_handler import AppDescriptor, BaseAppDescriptor


@dataclass
class QualCoreHandler(object):
    """
    Handler that processes the output of the core-tool module.

    :param result_path: the path to the core-tool output
    :param app_descriptors: list of apps that were analyzed by the core-tool and are ready to be
           processed by the python wrapper.
    :param app_raw_status: list of all eventlogs that were analyzed by the core-tool.
           This include both failed/skipped and successfull apps
    :param logger: logging object used for logging that section.
    """
    result_path: CspPathT
    app_descriptors: List[AppDescriptor] = field(default_factory=list, init=False)
    app_raw_status: List[BaseAppDescriptor] = field(default=list, init=False)
    logger: Logger = field(default=None, init=False)
    # need to define a list of apps that will be loaded from the csv files

    @cached_property
    def is_missing_core_results(self) -> bool:
        """
        Return true if the core-tool results are missing
        :return: true if the core-results are missing
        """
        return not self.core_result_path.exists()

    @cached_property
    def core_result_path(self) -> CspPathT:
        """
        Return the path to the core-tool output
        :return: the CSPAth to the core-tool output
        """
        return self.result_path.create_sub_path('qual_core_output')

    @cached_property
    def total_analyzed_apps(self) -> int:
        """
        Return the number of apps that were analyzed by the core-tool. This is typically the number
        of eventlogs that were processed by the core-tool.
        :return: the number of apps that were analyzed by the core-tool
        """
        return len(self.app_raw_status)

    def has_no_processable_apps(self) -> bool:
        """
        API call to check if the core-tool did not produce any processable rows to be analyzed.
        when this is true, the user-tools should understand that the apps are empty. However, it is
        possible that the core-tool processed apps but they were all skipped (i.e., not processable)
        :return: true, if the processable apps are no
        """
        return self.is_missing_core_results and not self.app_descriptors


    def __post_init__(self):
        self.logger = ToolLogging.get_and_setup_logger(f'rapids.tools.{self.__class__.__name__}')
        self._load_apps()

    def _load_apps(self) -> None:
        """
        Load the apps that were analyzed by the core-tool.
        Steps:
        1. load the QualOutputTable descriptor and use it to map labels to CSV files.
        2. load the rows from app_status.csv -> this will give all the analyzed eventlogs
        3. load the rows from app_summary.csv -> combine with appstatus to create the
           app_descriptor object
        :return: None
        """
        if self.is_missing_core_results:
            # results are empty. there is nothing to be loaded.
            return None
        summary_csv = self.core_result_path.create_sub_path('apps_summary.csv')
        status_csv = self.core_result_path.create_sub_path('status.csv')
        # load the summary and status, then create the app lists.
        # If an app is successful, it should be added to the app_descriptors list.
        # Otherwise, only keep it in the app_raw_status
        # TODO: fix the code blow
        with status_csv.open_input_stream() as f:
            self.app_raw_status = [BaseAppDescriptor(**row) for row in pd.read_csv(f).to_dict('records')]
        with summary_csv.open_input_stream() as f:
            # we should get the eventlog path and status from the BaseApp
            self.app_descriptors = [AppDescriptor(**row) for row in pd.read_csv(f).to_dict('records')]
        return None
