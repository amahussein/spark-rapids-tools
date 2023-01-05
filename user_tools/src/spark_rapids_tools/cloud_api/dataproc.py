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

"""Implementation specific to Dataproc"""

import json
import os
import re
from dataclasses import dataclass, field

from spark_rapids_tools.cloud_api.sp_types import PlatformBase, CloudPlatform, CMDDriverBase, ClusterState, \
    ClusterBase, SparkNodeType, ClusterNode, GpuDevice, GpuHWInfo, SysInfo
from spark_rapids_tools.common.prop_manager import YAMLPropertiesContainer, JSONPropertiesContainer


@dataclass
class DataprocPlatform(PlatformBase):
    """
    Represents the interface and utilities required by Dataproc.
    Prerequisites:
    - install gcloud command lines (gcloud, gsutil)
    - configure the gcloud CLI.
    - dataproc has staging temporary storage. we can retrieve that from the cluster properties.
    """
    type_id = CloudPlatform.DATAPROC

    def _create_cli_instance(self):
        return DataprocCMDDriver(cloud_ctxt=self.ctxt)

    def _construct_cluster_from_props(self,
                                      cluster: str,
                                      props: str = None):
        return DataprocCluster(self).set_connection(cluster_id=cluster, props=props)


@dataclass
class DataprocCMDDriver(CMDDriverBase):
    """Represents the command interface that will be used by Dataproc"""

    system_prerequisites = ['gcloud', 'gsutil']
    # configuration defaults = os.getenv('CLOUDSDK_DATAPROC_REGION')

    def get_and_set_env_vars(self):
        """For that driver, try to get all the available system environment for the system."""
        super().get_and_set_env_vars()
        # get the region if not set
        if self.get_region() is None:
            env_region = os.getenv('CLOUDSDK_DATAPROC_REGION')
            if env_region is not None:
                self.env_vars.update({'region': env_region})

    def pull_cluster_props_by_args(self, args: dict) -> str:
        cluster_name = args.get('cluster')
        # region is already set in the instance
        region_name = self.get_region()
        describe_cmd = f'gcloud dataproc clusters describe {cluster_name} --region={region_name}'
        return self.run_sys_cmd(describe_cmd)

    def _build_ssh_cmd_prefix_for_node(self, node: ClusterNode) -> str:
        return f'gcloud compute ssh {node.name} --zone={node.zone}'

    def _construct_ssh_cmd_with_prefix(self, prefix: str, remote_cmd: str) -> str:
        return f'{prefix} --command={remote_cmd}'

    def _build_platform_describe_node_instance(self, node: ClusterNode) -> list:
        cmd_params = ['gcloud compute machine-types describe',
                      f'{node.instance_type}',
                      f'--zone={node.zone}']
        return cmd_params

    def _build_platform_list_cluster(self,
                                     cluster,
                                     query_args: dict = None) -> list:
        filter_args = [f'clusterName = {cluster.name}']
        cmd_params = ['gcloud dataproc clusters list',
                      f"--region='{self.get_region()}'"]
        if query_args is not None:
            if 'state' in query_args:
                state_param = query_args.get('state')
                filter_args.append(f'status.state = {state_param}')
        filter_arg = ' AND '.join(filter_args)
        cmd_params.append(f"--filter='{filter_arg}'")
        return cmd_params


@dataclass
class DataprocNode(ClusterNode):
    """Implementation of Dataproc cluster node."""
    zone: str = field(default=None, init=False)

    @classmethod
    def __decode_machine_type_uri(cls, uri) -> (str, str):
        uri_parts = uri.split('/')[-4:]
        if uri_parts[0] != 'zones' or uri_parts[2] != 'machineTypes':
            raise RuntimeError(
                f'Unable to parse machine type from machine type URI: {uri}. '
                'Failed while processing CPU info')
        zone_val = uri_parts[1]
        machine_type_val = uri_parts[3]
        return zone_val, machine_type_val

    def _pull_and_set_mc_props(self, cli=None):
        instance_description = cli.exec_platform_describe_node_instance(self)
        self.mc_props = YAMLPropertiesContainer(prop_arg=instance_description, file_load=False)

    def _set_fields_from_props(self):
        worker_machine_type_uri = self.props.get_value('machineTypeUri')
        zone, machine_type = DataprocNode.__decode_machine_type_uri(worker_machine_type_uri)
        self.instance_type = machine_type
        self.zone = zone

    def _fetch_gpu_device(self, cli=None) -> GpuDevice or None:
        raw_gpu_device = cli.ssh_cmd_node(node=self,
                                          ssh_cmd='\"nvidia-smi --query-gpu=gpu_name --format=csv,noheader\"')
        all_lines = raw_gpu_device.splitlines()
        supported_gpus = DataprocPlatform.list_supported_gpus()

        for line in all_lines:
            processed_line = line.upper()
            for gpu_device in supported_gpus:
                if processed_line.find(GpuDevice.tostring(gpu_device)):
                    return gpu_device
        return None

    def _pull_gpu_hw_info(self, cli=None) -> GpuHWInfo:
        # get the GPU info
        # pull the GPU memory
        gpu_raw_memory = cli.ssh_cmd_node(node=self,
                                          ssh_cmd='\"nvidia-smi --query-gpu=memory.total --format=csv,noheader\"')
        # sometimes the output of the command may include SSH warning messages.
        # match only lines in with expression in the following format (15109 MiB)
        match_arr = re.findall(r'(\d+)\s+(MiB)', gpu_raw_memory, flags=re.MULTILINE)
        num_gpus = len(match_arr)
        gpu_mem = 0
        if num_gpus == 0:
            raise RuntimeError(f'Failed while pulling GPU memory information. '
                               f'Unrecognized GPU memory output format: {gpu_raw_memory}')
        for (mem_size, _) in match_arr:
            gpu_mem = max(int(mem_size), gpu_mem)

        gpu_device = self._fetch_gpu_device(cli)
        if gpu_device is None:
            gpu_device = GpuDevice.get_default_gpu()
        return GpuHWInfo(gpu_mem=gpu_mem,
                         num_gpus=num_gpus,
                         gpu_device=gpu_device)

    def _pull_sys_info(self, cli=None) -> SysInfo:
        # get the CPU info
        num_cpus = self.mc_props.get_value('guestCpus')
        cpu_mem = self.mc_props.get_value('memoryMb')
        return SysInfo(num_cpus=num_cpus, cpu_mem=cpu_mem)


@dataclass
class DataprocCluster(ClusterBase):
    """
    Represents an instance of running cluster on Dataproc.
    """
    def _init_nodes(self):
        # assume that only one master node
        master_nodes_from_conf = self.props.get_value('config', 'masterConfig', 'instanceNames')
        worker_nodes_from_conf = self.props.get_value('config', 'workerConfig', 'instanceNames')
        # create workers array
        worker_nodes: list = []
        raw_worker_prop = json.dumps(self.props.get_value('config', 'workerConfig'))

        for worker_node in worker_nodes_from_conf:
            worker_props = {
                'name': worker_node,
                'props': JSONPropertiesContainer(prop_arg=raw_worker_prop, file_load=False)
            }
            worker = DataprocNode.create_worker_node().set_fields_from_dict(worker_props)
            # TODO for optimization, we should set HW props for 1 worker
            worker.fetch_and_set_hw_info(self.cli)
            worker_nodes.append(worker)
        raw_master_props = json.dumps(self.props.get_value('config', 'masterConfig'))
        master_props = {
            'name': master_nodes_from_conf[0],
            'props': JSONPropertiesContainer(prop_arg=raw_master_props, file_load=False)
        }
        master_node = DataprocNode.create_master_node().set_fields_from_dict(master_props)
        master_node.fetch_and_set_hw_info(self.cli)
        self.nodes = {
            SparkNodeType.WORKER: worker_nodes,
            SparkNodeType.MASTER: master_node
        }

    def _set_fields_from_props(self):
        # set the zone
        zoneuri = self.props.get_value('config', 'gceClusterConfig', 'zoneUri')
        self.zone = zoneuri[zoneuri.rindex('/') + 1:]
        # set uuid
        self.uuid = self.props.get_value('clusterUuid')
        # set status
        self.state = ClusterState.fromstring(self.props.get_value('status', 'state'))

    def _init_connection(self, cluster_id: str = None,
                         props: str = None) -> dict:
        name = cluster_id
        if props is None:
            # we need to pull the properties from the platform
            props = self.cli.pull_cluster_props_by_args(args={'cluster': name, 'region': self.region})
        cluster_props = YAMLPropertiesContainer(props, file_load=False)
        cluster_args = {
            'name': name,
            'props': cluster_props
        }
        return cluster_args
