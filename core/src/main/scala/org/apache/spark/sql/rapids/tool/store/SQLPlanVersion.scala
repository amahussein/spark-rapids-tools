/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.rapids.tool.store

import scala.collection.breakOut

import com.nvidia.spark.rapids.tool.planparser.{DataWritingCommandExecParser, ReadParser}

import org.apache.spark.sql.execution.SparkPlanInfo
import org.apache.spark.sql.execution.ui.SparkPlanGraph
import org.apache.spark.sql.rapids.tool.AppBase
import org.apache.spark.sql.rapids.tool.util.ToolsPlanGraph


/**
 * Represents a version of the SQLPlan holding the specific information related to that SQL plan.
 * In an AQE event, Spark sends the new update with the updated SparkPlanInfo and
 * PhysicalPlanDescription.
 * @param sqlId sql plan ID. This is redundant because the SQLPlan Model has the same information.
 *              However, storing it here makes the code easier as we don't have to build
 *              (sqlId, SQLPLanVersion) tuples.
 * @param version the version number of this plan. It starts at 0 and increments by 1 for each
 *                SparkListenerSQLAdaptiveExecutionUpdate.
 * @param planInfo The instance of SparkPlanInfo for this version of the plan.
 * @param physicalPlanDescription The string representation of the physical plan for this version.
 * @param isFinal Flag to indicate if this plan is the final plan for the SQLPlan
 */
class SQLPlanVersion(
    val sqlId: Long,
    val version: Int,
    val planInfo: SparkPlanInfo,
    val physicalPlanDescription: String,
    var isFinal: Boolean = true) {

  // Used to cache the Spark graph for that plan to avoid creating a plan.
  // This graph can be used and then cleaned up at the end of the execution.
  // This has to be accessed through the getPlanGraph() which synchronizes on this object to avoid
  // races between threads.
  private var sparkGraph: Option[SparkPlanGraph] = None

  private def getPlanGraph(): SparkPlanGraph = {
    this.synchronized {
      if (sparkGraph.isEmpty) {
        sparkGraph = Some(ToolsPlanGraph(planInfo))
      }
      sparkGraph.get
    }
  }

  /**
   * Builds the list of write records for this plan.
   * This works by looping on all the nodes and filtering write execs.
   * @return the list of write records for this plan if any.
   */
  private def initWriteOperationRecords(): Iterable[WriteOperationRecord] = {
    getPlanGraph().allNodes
      // pick only nodes that are DataWritingCommandExec
      .filter(node => DataWritingCommandExecParser.isWritingCmdExec(node.name.stripSuffix("$")))
      .map { n =>
        // extract the meta data and convert it to store record.
        val opMeta = DataWritingCommandExecParser.getWriteOpMetaFromNode(n)
        WriteOperationRecord(sqlId, version, n.id, operationMeta = opMeta)
      }
  }

  // Captures the write operations for this plan. This is lazy because we do not need
  // to construct this until we need it.
  lazy val writeRecords: Iterable[WriteOperationRecord] = {
    initWriteOperationRecords()
  }

  // Converts the writeRecords into a write formats.
  def getWriteDataFormats: Set[String] = {
    writeRecords.map(_.operationMeta.dataFormat())(breakOut)
  }

  /**
   * Reset any data structure that has been used to free memory.
   */
  def cleanUpPlan(): Unit = {
    this.synchronized {
      sparkGraph = None
    }
  }

  def resetFinalFlag(): Unit = {
    // This flag depends on the AQE events sequence.
    // It does not set that field using the substring of the physicalPlanDescription
    // (isFinalPlan=true).
    // The consequences of this is that for incomplete eventlogs, the last PlanInfo to be precessed
    // is considered final.
    isFinal = false
  }

  /**
   * Starting with the SparkPlanInfo for this version, recursively get all the SparkPlanInfo within
   * the children that define a ReadSchema.
   * @return Sequence of SparkPlanInfo that have a ReadSchema attached to it.
   */
  def getPlansWithSchema: Seq[SparkPlanInfo] = {
    SQLPlanVersion.getPlansWithSchemaRecursive(planInfo)
  }

  /**
   * This is used to extract the metadata of ReadV1 nodes in Spark Plan Info
   * @param planGraph planGraph Optional SparkPlanGraph to use. If not provided, it will be created.
   * @return all the read datasources V1 recursively that are read by this plan including.
   */
  def getReadDSV1(planGraph: Option[SparkPlanGraph] = None): Iterable[DataSourceRecord] = {
    val graph = planGraph.getOrElse(getPlanGraph())
    getPlansWithSchema.flatMap { plan =>
      val meta = plan.metadata
      // TODO: Improve the extraction of ReaSchema using RegEx (ReadSchema):\s(.*?)(\.\.\.|,\s|$)
      val readSchema =
        ReadParser.formatSchemaStr(meta.getOrElse(ReadParser.METAFIELD_TAG_READ_SCHEMA, ""))
      val scanNodes = graph.allNodes.filter(ReadParser.isScanNode).filter(node => {
        // Get ReadSchema of each Node and sanitize it for comparison
        val trimmedNode = AppBase.trimSchema(ReadParser.parseReadNode(node).schema)
        readSchema.contains(trimmedNode)
      })
      if (scanNodes.nonEmpty) {
        Some(DataSourceRecord(
          sqlId,
          version,
          scanNodes.head.id,
          ReadParser.extractTagFromV1ReadMeta(ReadParser.METAFIELD_TAG_FORMAT, meta),
          ReadParser.extractTagFromV1ReadMeta(ReadParser.METAFIELD_TAG_LOCATION, meta),
          ReadParser.extractTagFromV1ReadMeta(ReadParser.METAFIELD_TAG_PUSHED_FILTERS, meta),
          readSchema,
          ReadParser.extractTagFromV1ReadMeta(ReadParser.METAFIELD_TAG_DATA_FILTERS, meta),
          ReadParser.extractTagFromV1ReadMeta(ReadParser.METAFIELD_TAG_PARTITION_FILTERS, meta),
          fromFinalPlan = isFinal))
      } else {
        None
      }
    }
  }

  /**
   * Get all the DataSources that are read by this plan (V2).
   * @param planGraph Optional SparkPlanGraph to use. If not provided, it will be created.
   * @return List of DataSourceRecord for all the V2 DataSources read by this plan.
   */
  def getReadDSV2(planGraph: Option[SparkPlanGraph] = None): Iterable[DataSourceRecord] = {
    val graph = planGraph.getOrElse(getPlanGraph())
    graph.allNodes.filter(ReadParser.isDataSourceV2Node).map { node =>
      val res = ReadParser.parseReadNode(node)
      DataSourceRecord(
        sqlId,
        version,
        node.id,
        res.format,
        res.location,
        res.pushedFilters,
        res.schema,
        res.dataFilters,
        res.partitionFilters,
        fromFinalPlan = isFinal)
    }
  }

  /**
   * Get all the DataSources that are read by this plan (V1 and V1).
   * @return Iterable of DataSourceRecord
   */
  def getAllReadDS: Iterable[DataSourceRecord] = {
    val planGraph = Option(getPlanGraph())
    getReadDSV1(planGraph) ++ getReadDSV2(planGraph)
  }
}

object SQLPlanVersion {
  /**
   * Recursive call to get all the SparkPlanInfo that have a schema attached to it.
   * This is mainly used for V1 ReadSchema
   * @param planInfo The SparkPlanInfo to start the search from
   * @return A list of SparkPlanInfo that have a schema attached to it.
   */
  private def getPlansWithSchemaRecursive(planInfo: SparkPlanInfo): Seq[SparkPlanInfo] = {
    val childRes = planInfo.children.flatMap(getPlansWithSchemaRecursive)
    if (planInfo.metadata != null &&
      planInfo.metadata.contains(ReadParser.METAFIELD_TAG_READ_SCHEMA)) {
      childRes :+ planInfo
    } else {
      childRes
    }
  }
}
