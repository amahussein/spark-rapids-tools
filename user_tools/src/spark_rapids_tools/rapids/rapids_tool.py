# Copyright (c) 2023, NVIDIA CORPORATION.
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

"""Abstract class representing wrapper around the RAPIDS acceleration tools."""

import os
import sys
from dataclasses import dataclass, field
from logging import Logger
from typing import Any, Callable, Type

from spark_rapids_tools.cloud_api.sp_types import CloudPlatform, get_platform, PlatformBase, ClusterBase
from spark_rapids_tools.common.prop_manager import YAMLPropertiesContainer
from spark_rapids_tools.common.sys_storage import LocalFS
from spark_rapids_tools.common.utilities import resource_path, gen_random_string, ToolLogging


@dataclass
class ToolContext(YAMLPropertiesContainer):
    """
    A container that holds properties and characteristics of a given execution.
    """
    name: str = None
    platform_cls: Type[PlatformBase] = None
    region: str = None
    logger: Logger = field(default=None, init=False)
    platform: PlatformBase = field(default=None, init=False)

    def __connect_to_platform(self):
        self.logger.info('Start connecting to the platform')
        self.platform = self.platform_cls({'region': self.region})

    def _init_fields(self):
        self.logger = ToolLogging.get_and_setup_logger(f'rapids.tools.{self.name}.ctxt')
        self.__connect_to_platform()
        self.props['localCtx'] = {}
        self.props['remoteCtx'] = {}
        self.props['wrapperCtx'] = {}

    def loginfo(self, msg: str):
        self.logger.info(msg)

    def logdebug(self, msg: str):
        self.logger.debug(msg)

    def logwarn(self, msg: str):
        self.logger.warning(msg)

    def set_ctxt(self, key: str, val: Any):
        self.props['wrapperCtx'][key] = val

    def get_ctxt(self, key: str):
        return self.props['wrapperCtx'].get(key)

    def set_remote(self, key: str, val: Any):
        self.props['remoteCtx'][key] = val

    def set_local(self, key: str, val: Any):
        self.props['localCtx'][key] = val

    def get_local(self, key: str):
        return self.props['localCtx'].get(key)

    def get_remote(self, key: str):
        return self.props['remoteCtx'].get(key)

    def set_local_workdir(self, parent: str):
        relative_path = self.get_value('platform', 'workDir')
        local_work_dir = os.path.join(parent, relative_path)
        self.logdebug(f'creating dependency folder {local_work_dir}')
        # first delete the folder if it exists
        LocalFS.remove_dir(local_work_dir, fail_on_error=False)
        # now create the new folder
        LocalFS.make_dirs(local_work_dir, exist_ok=False)
        self.set_local('depFolder', local_work_dir)
        output_folder = os.path.join(local_work_dir, self.get_value('platform', 'outputDir'))
        self.set_local('toolOutputFolder', output_folder)
        self.logdebug(f'setting local output folder of the tool to {self.get_local("toolOutputFolder")}')

    def get_remote_output_dir(self) -> str:
        remote_work_dir = self.get_remote('depFolder')
        return os.path.join(remote_work_dir, self.get_value('platform', 'outputDir'))

    def get_local_output_dir(self) -> str:
        local_folder = self.get_wrapper_local_output()
        if self.get_value_silent('toolOutput', 'subFolder') is None:
            return local_folder
        return os.path.join(local_folder, self.get_value('toolOutput', 'subFolder'))

    def get_wrapper_local_output(self) -> str:
        local_folder = os.path.join(self.get_local_work_dir(), self.get_value('platform', 'outputDir'))
        return local_folder

    def set_remote_workdir(self, parent: str):
        static_prefix = os.path.join(parent, self.get_value('platform', 'workDir'))
        remote_work_dir = f'{static_prefix}_{gen_random_string(12)}'
        self.set_remote('depFolder', remote_work_dir)

    def get_local_work_dir(self) -> str:
        return self.get_local('depFolder')

    def get_remote_work_dir(self) -> str:
        return self.get_remote('depFolder')

    def get_default_jar_name(self) -> str:
        jar_version = self.get_value('sparkRapids', 'version')
        default_jar_name = self.get_value('sparkRapids', 'jarFile')
        return default_jar_name.format(jar_version)

    def get_rapids_jar_url(self) -> str:
        jar_version = self.get_value('sparkRapids', 'version')
        rapids_url = self.get_value('sparkRapids', 'repoUrl').format(jar_version, jar_version)
        return rapids_url

    def get_tool_main_class(self) -> str:
        return self.get_value('sparkRapids', 'mainClass')


@dataclass
class RapidsTool(object):
    """
    A generic class that represents a RAPIDS plugin tool.
    :param platform_type: the type of platform associated with the current execution.
    :param cluster: name of the cluster on which the application will be running
    :param region: name of region
    :param output_folder: location to store the output of the execution
    :param config_path: location of the configuration file of the current tool
    :param wrapper_options: dictionary containing options specific to the wrapper tool execution.
    :param rapids_options: dictionary containing the options to be passed as CLI arguments to the RAPIDS Accelerator.
    :param name: the name of the tool
    :param ctxt: context manager for the current tool execution.
    :param logger: the logger instant associated to the current tool.
    """
    platform_type: CloudPlatform
    cluster: str
    region: str
    output_folder: str
    config_path: str = None
    wrapper_options: dict = field(default_factory=dict)
    rapids_options: dict = field(default_factory=dict)
    name: str = field(default=None, init=False)
    ctxt: ToolContext = field(default=None, init=False)
    logger: Logger = field(default=None, init=False)

    def pretty_name(self):
        return self.name.capitalize()

    def get_exec_cluster(self) -> ClusterBase:
        return self.ctxt.get_ctxt('exec_cluster')

    def phase_banner(func_cb: Callable):   # pylint: disable=no-self-argument
        """Phases of each tool announces the beginning and end"""
        def wrapper(self, *args, **kwargs):
            func_name = func_cb.__name__  # pylint: disable=no-member
            try:
                self.logger.info(f'*** {self.pretty_name()} starting  phase {func_name} ***')
                func_cb(self, *args, **kwargs)     # pylint: disable=not-callable
                self.logger.info(f'*** {self.pretty_name()} succeeded phase {func_name} ***')
            except Exception as exception:    # pylint: disable=broad-except
                self.logger.error(f'*** {self.pretty_name()} raised an error in phase {func_name} *** {exception}')
                sys.exit(1)
        return wrapper

    def __post_init__(self):
        # when debug is set to true set it in the environment.
        self.logger = ToolLogging.get_and_setup_logger(f'rapids.tools.{self.name}')

    def _check_environment(self) -> None:
        self.ctxt.platform.setup_and_validate_env()

    def _process_output_args(self):
        self.logger.debug('Processing Output Arguments')
        workdir = os.path.join(self.output_folder, 'wrapper-output')
        self.ctxt.set_local_workdir(workdir)

    def _process_rapids_args(self):
        pass

    def _process_custom_args(self):
        pass

    @phase_banner
    def _process_arguments(self):
        self._process_output_args()
        # 1- process any arguments to be passed to the RAPIDS tool
        self._process_rapids_args()
        # 2- we need to process the arguments of the CLI
        self._process_custom_args()

    @phase_banner
    def _init_tool(self):
        self._init_ctxt()
        self._check_environment()

    def _init_ctxt(self):
        if self.config_path is None:
            self.config_path = resource_path(f'{self.name}-conf.yaml')

        self.ctxt = ToolContext(platform_cls=get_platform(self.platform_type),
                                region=self.region,
                                prop_arg=self.config_path,
                                name=self.name)

    def _run_rapids_tool(self):
        pass

    @phase_banner
    def _execute(self):
        """
        Phase representing actual execution of the wrapper command.
        """
        self._run_rapids_tool()

    def _process_output(self):
        pass

    def _download_output(self):
        pass

    def _write_summary(self):
        pass

    @phase_banner
    def _collect_result(self):
        """
        Following a successful run, collect and process data as needed
        :return:
        """
        self._download_output()
        self._process_output()
        self._write_summary()

    @phase_banner
    def _connect_to_execution_cluster(self):
        """
        Connecting to execution cluster
        :return:
        """
        exec_cluster = self.ctxt.platform.connect_cluster_by_name(self.cluster)
        if not exec_cluster.is_cluster_running():
            raise RuntimeError(
                f'Cluster {exec_cluster.name} is not running. '
                f'Make sure that the execution cluster is in RUNNING state, then re-try.')
        self.ctxt.set_ctxt('exec_cluster', exec_cluster)

    def launch(self):
        self._init_tool()
        self._process_arguments()
        self._connect_to_execution_cluster()
        self._execute()
        self._collect_result()

    def _report_results_are_empty(self) -> None:
        print(f'The {self.pretty_name()} tool did not generate any output. Nothing to display.')


@dataclass
class RapidsJarTool(RapidsTool):
    def _process_rapids_args(self):
        self.logger.info('Hello World from Jar tool')
