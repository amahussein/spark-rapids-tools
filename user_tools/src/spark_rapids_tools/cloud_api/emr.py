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

"""Implementation specific to EMR"""

import json
import os
from dataclasses import field, dataclass

from spark_rapids_tools.cloud_api.sp_types import PlatformBase, ClusterBase, CMDDriverBase, CloudPlatform, \
    ClusterState, SparkNodeType, ClusterNode, GpuHWInfo, SysInfo, GpuDevice
from spark_rapids_tools.common.prop_manager import JSONPropertiesContainer
from spark_rapids_tools.common.utilities import find_full_rapids_tools_env_key, get_rapids_tools_env


@dataclass
class EMRPlatform(PlatformBase):
    """
    Represents the interface and utilities required by AWS EMR.
    Prerequisites:
    - install gcloud command lines (gcloud, gsutil)
    - configure the aws
        - this may be done by region
    - aws has no staging available in the cluster properties.
    - gsutil is used to move data from/to storage
    """
    type_id = CloudPlatform.EMR

    @classmethod
    def get_spark_node_type_fromstring(cls, value) -> SparkNodeType:
        if value.upper() in ['TASK', 'CORE']:
            return SparkNodeType.WORKER
        return SparkNodeType.fromstring(value)

    def _create_cli_instance(self):
        return EMRCMDDriver(timeout=0, cloud_ctxt=self.ctxt)

    def _construct_cluster_from_props(self,
                                      cluster: str,
                                      props: str = None):
        return EMRCluster(self).set_connection(cluster_id=cluster, props=props)


@dataclass
class EMRCMDDriver(CMDDriverBase):
    """Represents the command interface that will be used by EMR"""
    system_prerequisites = ['aws']
    # configuration defaults = AWS_REGION; AWS_DEFAULT_REGION

    def get_and_set_env_vars(self):
        """For that driver, try to get all the available system environment for the system."""
        super().get_and_set_env_vars()
        # TODO: verify that the AWS CLI is configured.
        # get the region
        if self.env_vars.get('region') is None:
            env_region = os.getenv('AWS_REGION', os.getenv('AWS_DEFAULT_REGION'))
            if env_region is not None:
                self.env_vars.update({'region': env_region})
        self.env_vars.update({
            'output': os.getenv('AWS_DEFAULT_OUTPUT', 'json'),
            'profile': os.getenv('AWS_PROFILE', 'DEFAULT')
        })
        # For EMR we need the key_pair file name for the connection to clusters
        # TODO: Check the keypair has extension pem file and they are set correctly.
        emr_key_name = get_rapids_tools_env('EMR_KEY_NAME')
        emr_pem_path = get_rapids_tools_env('EMR_PEM_PATH')
        self.env_vars.update({
            'keyPairName': emr_key_name,
            'keyPemPATH': emr_pem_path
        })

    def validate_env(self):
        super().validate_env()
        incorrect_envs = []
        # check that private key file path is correct
        emr_pem_path = self.env_vars.get('keyPemPATH')
        if emr_pem_path is not None:
            if not os.path.exists(emr_pem_path):
                incorrect_envs.append(f'Private key file path [{emr_pem_path}] does not exist')
                # check valid extension
                if not (emr_pem_path.endswith('.pem') or emr_pem_path.endswith('ppk')):
                    incorrect_envs.append(f'Private key file path [{emr_pem_path}] should be ppk or pem format')
        else:
            incorrect_envs.append(
                f'Private key file path is not set. '
                f'Set {find_full_rapids_tools_env_key("EMR_KEY_NAME")}.')
        # check that private key is set
        if self.env_vars.get('keyPairName') is None:
            incorrect_envs.append(
                f'Private key name is not set correctly. '
                f'Set {find_full_rapids_tools_env_key("EMR_PEM_PATH")}.')
        if len(incorrect_envs) > 0:
            exc_msg = '; '.join(incorrect_envs)
            raise RuntimeError(f'Invalid environment {exc_msg}')

    def pull_cluster_props_by_args(self, args: dict) -> str:
        aws_cluster_id = args.get('Id')
        cluster_name = args.get('cluster')
        if args.get('Id') is None:
            # use cluster name to get the cluster values
            # we need to get the cluster_id from the list command first.
            list_cmd_res = self.exec_platform_list_cluster_by_name(cluster_name)
            error_msg = f'Could not find EMR cluster {cluster_name} by name'
            if not list_cmd_res:
                raise RuntimeError(error_msg)
            # listed_cluster is json formatted string of array, but we need only the first entry
            # to read the clusterID
            cluster_headers: list = json.loads(list_cmd_res)
            if len(cluster_headers) == 0:
                raise RuntimeError(error_msg)
            existing_cluster = cluster_headers[0]
            aws_cluster_id = existing_cluster['Id']
        self.logger.debug('Cluster %s has an Id %s', cluster_name, aws_cluster_id)
        cluster_described = self.exec_platform_describe_cluster_by_id(aws_cluster_id)
        self.logger.debug('Cluster %s description = %s', cluster_name, cluster_described)
        if cluster_described is not None:
            cluster_json = json.loads(cluster_described)
            if cluster_json.get('Cluster') is not None:
                return json.dumps(cluster_json.get('Cluster'))
        return cluster_described

    def _build_ssh_cmd_prefix_for_node(self, node: ClusterNode) -> str:
        # get the pem file
        pem_file_path = self.env_vars.get('keyPemPATH')
        prefix_args = ['ssh',
                       '-o StrictHostKeyChecking=no',
                       f'-i {pem_file_path}',
                       f'hadoop@{node.name}']
        return ' '.join(prefix_args)

    def _build_platform_describe_node_instance(self, node: ClusterNode) -> list:
        cmd_params = ['aws ec2 describe-instance-types',
                      '--region', f'{self.get_region()}',
                      '--instance-types', f'{node.instance_type}']
        return cmd_params

    def _build_platform_list_cluster(self,
                                     cluster,
                                     query_args: dict = None) -> list:
        # aws emr list-instances --cluster-id j-2DDF0Q87QOXON
        cmd_params = ['aws emr list-instances',
                      '--cluster-id',
                      f'{cluster.uuid}']
        if query_args is not None:
            for q_key in query_args:
                cmd_params.append(f'--{q_key}')
                cmd_params.append(f'{query_args.get(q_key)}')
        return cmd_params

    def exec_platform_list_cluster_by_name(self,
                                           cluster_name: str):
        list_cmd = f"aws emr list-clusters --query 'Clusters[?Name==`{cluster_name}`]'"
        return self.run_sys_cmd(list_cmd)

    def exec_platform_describe_cluster_by_id(self,
                                             cluster_id: str):
        describe_cmd = f'aws emr describe-cluster --cluster-id {cluster_id}'
        return self.run_sys_cmd(describe_cmd)


@dataclass
class EMRNode(ClusterNode):
    """
    Represents EMR cluster Node.
    We assume that all nodes are running on EC2 instances.
    """
    emr_grp_type: str = field(default=None, init=False)
    ec2_id: str = field(default=None, init=False)
    instance_id: str = field(default=None, init=False)
    group_id: str = field(default=None, init=False)

    def _pull_and_set_mc_props(self, cli=None):
        instance_description = cli.exec_platform_describe_node_instance(self)
        mc_description = json.loads(instance_description)['InstanceTypes'][0]
        self.mc_props = JSONPropertiesContainer(prop_arg=json.dumps(mc_description), file_load=False)

    def _set_fields_from_props(self):
        self.name = self.props.get_value('PublicDnsName')
        self.ec2_id = self.props.get_value('Ec2InstanceId')
        self.instance_id = self.props.get_value('Id')
        self.group_id = self.props.get_value('InstanceGroupId')

    def _pull_sys_info(self, cli=None) -> SysInfo:
        cpu_mem = self.mc_props.get_value('MemoryInfo', 'SizeInMiB')
        # TODO: should we use DefaultVCpus or DefaultCores
        num_cpus = self.mc_props.get_value('VCpuInfo', 'DefaultCores')
        return SysInfo(num_cpus=num_cpus, cpu_mem=cpu_mem)

    def _pull_gpu_hw_info(self, cli=None) -> GpuHWInfo or None:
        raw_gpus = self.mc_props.get_value_silent('GpuInfo')
        if raw_gpus is None:
            return None
        # TODO: we assume all gpus of the same type
        raw_gpu_arr = raw_gpus.get('Gpus')
        if raw_gpu_arr is None:
            return None
        raw_gpu = raw_gpu_arr[0]
        gpu_device = GpuDevice.fromstring(raw_gpu['Name'])
        gpu_cnt = raw_gpu['Count']
        gpu_mem = raw_gpu['MemoryInfo']['SizeInMiB']
        return GpuHWInfo(num_gpus=gpu_cnt,
                         gpu_device=gpu_device,
                         gpu_mem=gpu_mem)


@dataclass
class EMRCluster(ClusterBase):
    """
    Represents an instance of running cluster on EMR.
    """

    def _init_connection(self, cluster_id: str = None,
                         props: str = None) -> dict:
        name = cluster_id
        if props is None:
            # we need to pull the properties from the platform
            props = self.cli.pull_cluster_props_by_args(args={'cluster': name, 'region': self.region})
        cluster_props = JSONPropertiesContainer(props, file_load=False)
        cluster_args = {
            'name': name,
            'props': cluster_props
        }
        return cluster_args

    def _init_nodes(self):
        def process_cluster_group_list(inst_groups: list) -> list:
            processed_instances = []
            for inst_grp in inst_groups:
                grp_id = inst_grp['Id']
                grp_type = inst_grp['InstanceGroupType']
                instance_type = inst_grp['InstanceType']
                count = inst_grp['RequestedInstanceCount']
                # we need to get the public dns name of all the nodes which can be found in
                # the output of the instances command
                query_args = {'instance-group-id': grp_id}
                raw_instance_list = self.cli.exec_platform_list_cluster_instances(self, query_args=query_args)
                instances_list = json.loads(raw_instance_list).get('Instances')
                curr_instance = {
                    'grp_id': grp_id,
                    'emr_grp_type': grp_type,
                    'spark_grp_type': EMRPlatform.get_spark_node_type_fromstring(grp_type),
                    'instance_type': instance_type,
                    'count': count,
                    'instances_list': instances_list
                }
                processed_instances.append(curr_instance)
            return processed_instances

        # get instance_groups from the cluster props.
        instance_groups = self.props.get_value('InstanceGroups')
        inst_groups_list = process_cluster_group_list(instance_groups)
        worker_nodes: list = []
        master_nodes: list = []
        for curr_group in inst_groups_list:
            for inst in curr_group['instances_list']:
                node_props = {
                    'instance_type': inst['InstanceType'],
                    'emr_grp_type': curr_group['emr_grp_type'],
                    'props': JSONPropertiesContainer(prop_arg=json.dumps(inst), file_load=False)
                }
                c_node = EMRNode.create_node(curr_group['spark_grp_type']).set_fields_from_dict(node_props)
                c_node.fetch_and_set_hw_info(self.cli)
                if c_node.node_type == SparkNodeType.WORKER:
                    worker_nodes.append(c_node)
                else:
                    master_nodes.append(c_node)
        self.nodes = {
            SparkNodeType.WORKER: worker_nodes,
            SparkNodeType.MASTER: master_nodes[0]
        }

    def _set_fields_from_props(self):
        self.uuid = self.props.get_value('Id')
        self.state = ClusterState.fromstring(self.props.get_value('Status', 'State'))

    def is_cluster_running(self) -> bool:
        acceptable_init_states = [
            ClusterState.RUNNING,
            ClusterState.STARTING,
            ClusterState.BOOTSTRAPPING,
            ClusterState.WAITING
        ]
        return self.state in acceptable_init_states
