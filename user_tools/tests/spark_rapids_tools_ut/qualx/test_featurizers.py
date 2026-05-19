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

"""Test qualx default featurizer module"""
from unittest.mock import patch

import pandas as pd

from spark_rapids_tools.tools.qualx.featurizers.default import extract_raw_features
from ..conftest import SparkRapidsToolsUT


def _make_minimal_tables(ds_tbl: pd.DataFrame) -> dict:
    """Build a minimal set of tables for testing extract_raw_features."""
    app_id = 'app-12345'
    sql_id = 0
    app_name = 'test_app'

    byte_features = [
        'diskBytesSpilled', 'memoryBytesSpilled', 'input_bytesRead', 'output_bytesWritten',
        'sr_localBytesRead', 'sr_remoteBytesRead', 'sr_remoteBytesReadToDisk',
        'sr_totalBytesRead', 'sw_bytesWritten',
    ]
    time_features = [
        'executorCPUTime', 'executorDeserializeCPUTime', 'executorDeserializeTime',
        'executorRunTime', 'jvmGCTime', 'sr_fetchWaitTime', 'sw_writeTime',
    ]

    jsa_data = {
        'appId': [app_id], 'appName': [app_name], 'sqlID': [sql_id],
        'numTasks_sum': [10], 'duration_sum': [1000],
        'duration_min': [100], 'duration_max': [200],
    }
    for f in byte_features + time_features:
        jsa_data[f + '_sum'] = [100]
    jsa_df = pd.DataFrame(jsa_data)

    app_df = pd.DataFrame({
        'appId': [app_id], 'sqlID': [sql_id], 'startTime': [1000], 'appDuration': [5000],
        'Duration': [1000], 'description': ['test'], 'sparkRuntime': ['Spark'],
        'sparkVersion': ['3.0'], 'pluginEnabled': [False], 'resourceProfileId': [0],
        'numExecutors': [2], 'executorCores': [4], 'maxMem': [1000], 'maxOnHeapMem': [1000],
        'maxOffHeapMem': [0], 'executorMemory': [1000], 'numGpusPerExecutor': [0],
        'executorOffHeap': [0], 'taskCpu': [1.0], 'taskGpu': [0.0], 'appName': [app_name],
    })

    ops_df = pd.DataFrame({
        'appId': [app_id], 'sqlID': [sql_id], 'metricType': ['timing'],
        'nodeName': ['Filter'], 'nodeID': [1], 'name': ['duration'], 'total': [100], 'max': [100],
    })

    failed_tasks_df = pd.DataFrame(
        columns=['appName', 'appId', 'sqlID', 'failed_tasks']
    ).astype({'sqlID': int, 'failed_tasks': int})

    return {
        'app_tbl': app_df,
        'ops_tbl': ops_df,
        'job_stage_agg_tbl': jsa_df,
        'wholestage_tbl': pd.DataFrame(),
        'ds_tbl': ds_tbl,
        'failed_tasks_tbl': failed_tasks_df,
        'spark_props_tbl': pd.DataFrame(),
    }


class TestDefaultFeaturizer(SparkRapidsToolsUT):
    """Test class for qualx default featurizer module"""

    def test_extract_raw_features_empty_ds_tbl(self):
        """extract_raw_features should not raise KeyError when ds_tbl has no appId column.

        Regression test for https://github.com/NVIDIA/spark-rapids-tools/issues/2050
        """
        toc = pd.DataFrame({
            'appId': ['app-12345'],
            'ds_name': ['test_app'],
            'table_name': ['dummy'],
            'filepath': ['/dummy/path'],
        })
        tables = _make_minimal_tables(ds_tbl=pd.DataFrame())

        with patch(
            'spark_rapids_tools.tools.qualx.featurizers.default.load_csv_files',
            return_value=tables,
        ):
            result = extract_raw_features(toc, node_level_supp=None, qualtool_filter=None)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        # data source columns should be absent when ds_tbl is empty
        for col in ('scan_bw', 'scan_time', 'decode_time', 'data_size'):
            assert col not in result.columns

    def test_extract_raw_features_with_ds_tbl(self):
        """extract_raw_features should include data source features when ds_tbl is populated."""
        app_id = 'app-12345'
        sql_id = 0
        toc = pd.DataFrame({
            'appId': [app_id],
            'ds_name': ['test_app'],
            'table_name': ['dummy'],
            'filepath': ['/dummy/path'],
        })
        ds_tbl = pd.DataFrame({
            'appId': [app_id], 'sqlID': [sql_id],
            'data_size': [1000], 'scan_time': [100], 'decode_time': [50],
        })
        tables = _make_minimal_tables(ds_tbl=ds_tbl)

        with patch(
            'spark_rapids_tools.tools.qualx.featurizers.default.load_csv_files',
            return_value=tables,
        ):
            result = extract_raw_features(toc, node_level_supp=None, qualtool_filter=None)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        # data source columns should be present when ds_tbl has data
        for col in ('scan_bw', 'scan_time', 'decode_time', 'data_size'):
            assert col in result.columns
        assert result['scan_bw'].iloc[0] == 10.0  # 1000 / 100
