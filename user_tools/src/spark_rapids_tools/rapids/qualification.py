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

"""Implementation class representing wrapper around the RAPIDS acceleration Qualification tool."""

from dataclasses import dataclass

from spark_rapids_tools.rapids.rapids_tool import RapidsJarTool


@dataclass
class Qualification(RapidsJarTool):
    """
    Wrapper layer around Qualification Tool.
    """
    name = 'qualification'

    def _process_rapids_args(self):
        """
        Qualification tool processes extra arguments:
        1. filter out applications.
        2. gpu-device type to be used for the cost estimation.
        3. gpu_per_machine: number of gpu installed on a worker node.
        4. cuda version
        """
        super()._process_rapids_args()
        self.logger.info('Qualification tool processing the arguments')

    def _process_custom_args(self):
        """
        Qualification tool processes extra arguments:
        1. filter out applications.
        2. gpu-device type to be used for the cost estimation.
        3. gpu_per_machine: number of gpu installed on a worker node.
        4. cuda version
        """
        def process_filter_opt(arg_val: str):
            available_filters = self.ctxt.get_value('sparkRapids', 'cli', 'defaults', 'filters',
                                                    'definedFilters')

            selected_filter = self.ctxt.get_value('sparkRapids', 'cli', 'defaults', 'filters',
                                                  'defaultFilter')
            if arg_val is not None:
                processed_filter = arg_val.lower().strip()
                if processed_filter in available_filters:
                    # correct argument
                    selected_filter = processed_filter
                else:
                    # revert to default filter
                    self.logger.warning(
                        'Invalid argument filter_apps=%s.\n\t'
                        'Accepted options are: [%s].\n\t'
                        'Falling-back to default filter: %s',
                        processed_filter, ' | '.join(available_filters), selected_filter)
            self.ctxt.set_ctxt('filter_apps', selected_filter)

        gpu_device = self.ctxt.get_value('sparkRapids', 'gpu', 'device')
        gpu_device_arg = self.wrapper_options.get('gpu_device')
        if gpu_device_arg is not None:
            gpu_device = gpu_device_arg
        gpu_per_machine = int(self.ctxt.get_value('sparkRapids', 'gpu', 'workersPerNode'))
        gpu_per_machine_arg = self.wrapper_options.get('gpu_per_machine')
        if gpu_per_machine_arg is not None:
            gpu_per_machine = gpu_per_machine_arg
        cuda = self.ctxt.get_value('sparkRapids', 'gpu', 'cudaVersion')
        cuda_arg = self.wrapper_options.get('cuda')
        if cuda_arg is not None:
            cuda = cuda_arg
        self.ctxt.set_ctxt('gpu_per_machine', gpu_per_machine)
        self.ctxt.set_ctxt('gpu_device', gpu_device)
        self.ctxt.set_ctxt('cuda', cuda)
        process_filter_opt(self.wrapper_options.get('filter_apps'))
        self.logger.debug('%s custom arguments = %s', self.pretty_name(), self.ctxt.props['wrapperCtx'])
