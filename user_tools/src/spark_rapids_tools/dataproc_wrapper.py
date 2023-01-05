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

"""Wrapper class to run tools associated with RAPIDS Accelerator for Apache Spark plugin on Dataproc."""

import fire

from spark_rapids_tools.cloud_api.sp_types import CloudPlatform
from spark_rapids_tools.rapids.bootstrap import Bootstrap
from spark_rapids_tools.rapids.profiling import Profiling
from spark_rapids_tools.rapids.qualification import Qualification
from spark_rapids_tools.common.utilities import ToolLogging


class DataprocWrapper:
    """
    A wrapper script to run Rapids tools (Qualification, Profiling, and Bootstrap) tools on
    Gcloud Dataproc.
    Disclaimer:
      Estimates provided by the tools are based on the currently supported "SparkPlan" or
      "Executor Nodes" used in the application. It currently does not handle all the expressions
      or datatypes used.
      The pricing estimate does not take into considerations:
      1- Sustained Use discounts
      2- Cost of on-demand VMs

    Run one of the following commands:
    :qualification args
    :profiling args
    :bootstrap args
    spark_rapids_dataproc qualification *args

    For more details on each command: run qualification --help
    """
    def __init__(self):
        self.__platform_type = CloudPlatform.DATAPROC

    def __get_platform_type(self):
        return self.__platform_type

    def profiling(self,
                  cluster: str,
                  region: str,
                  tools_jar: str = None,
                  eventlogs: str = None,
                  output_folder: str = '.',
                  gpu_cluster_props: str = None,
                  gpu_cluster_region: str = None,
                  gpu_cluster_zone: str = None,
                  debug: bool = False,
                  **rapids_options) -> None:
        """
        The Profiling tool analyzes both CPU or GPU generated event logs and generates information
        which can be used for debugging and profiling Apache Spark applications.

        The output information contains the Spark version, executor details, properties, etc. It also
        uses heuristics based techniques to recommend Spark configurations for users to run Spark on RAPIDS.

        :param cluster: Name of the dataproc cluster.
               Note that the cluster has to: 1- be running; and 2- support Spark3.x+.
        :param region: Compute region (e.g. us-central1) for the cluster.
        :param tools_jar: Path to a bundled jar including Rapids tool.
                          The path is a local filesystem, or gstorage url.
        :param eventlogs: Event log filenames(comma separated) or gcloud storage directories
            containing event logs.
            eg: gs://<BUCKET>/eventlog1,gs://<BUCKET1>/eventlog2
            If not specified, the wrapper will pull the default SHS directory from the cluster
            properties, which is equivalent to gs://$temp_bucket/$uuid/spark-job-history or the
            PHS log directory if any.
        :param output_folder: Base output directory.
            The final output will go into a subdirectory called wrapper-output.
            It will overwrite any existing directory with the same name.
        :param gpu_cluster_props: Path to a file containing configurations of the GPU cluster
            on which the Spark applications ran on.
            The path is a local filesystem, or gstorage url.
            This option does not require the cluster to be live. When missing, the configurations
            are pulled from the live cluster on which the Qualification tool is submitted.
        :param gpu_cluster_region: The region where the GPU cluster belongs to. Note that this parameter requires
            'gpu_cluster_props' to be defined.
            When missing, the region is set to the value passed in the 'region' argument.
        :param gpu_cluster_zone: The zone where the GPU cluster belongs to. Note that this parameter requires
            'gpu_cluster_props' to be defined.
            When missing, the zone is set to the same zone as the 'cluster' on which the Profiling tool is
            submitted.
        :param debug: True or False to enable verbosity to the wrapper script.
        :param rapids_options: A list of valid Profiling tool options.
            Note that the wrapper ignores the following flags
            [“auto-tuner“, “worker-info“, “compare“, “combined“, “output-directory“].
            For more details on Profiling tool options, please visit
            https://nvidia.github.io/spark-rapids/docs/spark-profiling-tool.html#profiling-tool-options.
        """
        if debug:
            # when debug is set to true set it in the environment.
            ToolLogging.enable_debug_mode()
        wrapper_prof_options = {
            'migration_clusters_props': {
                'gpu_cluster_props_path': gpu_cluster_props,
                'gpu_cluster_region': gpu_cluster_region,
                'gpu_cluster_zone': gpu_cluster_zone
            },
            'eventlogs': eventlogs,
            'tools_jar': tools_jar,
        }
        rapids_tool = Profiling(platform_type=self.__get_platform_type(),
                                cluster=cluster,
                                region=region,
                                output_folder=output_folder,
                                wrapper_options=wrapper_prof_options,
                                rapids_options=rapids_options)
        rapids_tool.launch()

    def qualification(self,
                      cluster: str,
                      region: str = None,
                      output_folder: str = '.',
                      eventlogs: str = None,
                      tools_jar: str = None,
                      cpu_cluster_props: str = None,
                      cpu_cluster_region: str = None,
                      cpu_cluster_zone: str = None,
                      gpu_cluster_props: str = None,
                      gpu_cluster_region: str = None,
                      gpu_cluster_zone: str = None,
                      # wrapper options
                      filter_apps: str = 'savings',
                      gpu_device: str = 'T4',
                      gpu_per_machine: int = 2,
                      cuda: str = '11.5',
                      debug: bool = False,
                      **rapids_options) -> None:
        """

        :param cluster:
        :param region:
        :param output_folder:
        :param eventlogs:
        :param tools_jar:
        :param cpu_cluster_props:
        :param cpu_cluster_region:
        :param cpu_cluster_zone:
        :param gpu_cluster_props:
        :param gpu_cluster_region:
        :param gpu_cluster_zone:
        :param filter_apps:
        :param gpu_device:
        :param gpu_per_machine:
        :param cuda:
        :param debug:
        :param rapids_options:
        :return:
        """
        if debug:
            # when debug is set to true set it in the environment.
            ToolLogging.enable_debug_mode()
        wrapper_qual_options = {
            'migration_clusters_props': {
                'cpu_cluster_props_path': cpu_cluster_props,
                'cpu_cluster_region': cpu_cluster_region,
                'cpu_cluster_zone': cpu_cluster_zone,
                'gpu_cluster_props_path': gpu_cluster_props,
                'gpu_cluster_region': gpu_cluster_region,
                'gpu_cluster_zone': gpu_cluster_zone
            },
            'eventlogs': eventlogs,
            'filter_apps': filter_apps,
            'tools_jar': tools_jar,
            'gpu_device': gpu_device,
            'gpu_per_machine': gpu_per_machine,
            'cuda': cuda
        }
        rapids_tool = Qualification(platform_type=self.__get_platform_type(),
                                    cluster=cluster,
                                    region=region,
                                    output_folder=output_folder,
                                    wrapper_options=wrapper_qual_options,
                                    rapids_options=rapids_options)
        rapids_tool.launch()

    def bootstrap(self,
                  cluster: str,
                  region: str,
                  output_folder: str = '.',
                  dry_run: bool = False,
                  debug: bool = False) -> None:
        """
        The bootstrap tool analyzes the CPU and GPU configuration of the EMR cluster
        and updates the Spark default configuration on the cluster's master nodes.

        :param cluster: Name of the dataproc cluster
        :param region: Compute region (e.g. us-central1) for the cluster.
        :param dry_run: True or False to update the Spark config settings on Dataproc master node.
        :param output_folder: Base output directory. The final recommendations will be logged in the
               subdirectory 'wrapper-output/rapids_user_tools_bootstrap'.
               Note that this argument only accepts local filesystem.
        :param debug: True or False to enable verbosity to the wrapper script.
        """
        if debug:
            # when debug is set to true set it in the environment.
            ToolLogging.enable_debug_mode()
        wrapper_boot_options = {
            'dry_run': dry_run
        }
        rapids_tool = Bootstrap(platform_type=self.__get_platform_type(),
                                cluster=cluster,
                                region=region,
                                output_folder=output_folder,
                                wrapper_options=wrapper_boot_options)
        rapids_tool.launch()


def main():
    fire.Fire(DataprocWrapper)


if __name__ == '__main__':
    main()
