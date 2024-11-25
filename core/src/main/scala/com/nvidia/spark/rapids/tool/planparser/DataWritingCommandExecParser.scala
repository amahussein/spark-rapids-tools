/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.tool.planparser

import com.nvidia.spark.rapids.tool.qualification.PluginTypeChecker

import org.apache.spark.sql.execution.ui.SparkPlanGraphNode

case class DataWritingCommandExecParser(
    node: SparkPlanGraphNode,
    checker: PluginTypeChecker,
    sqlID: Long) extends ExecParser {

  // hardcode because InsertIntoHadoopFsRelationCommand uses this same exec
  // and InsertIntoHadoopFsRelationCommand doesn't have an entry in the
  // supported execs file
  val fullExecName: String = DataWritingCommandExecParser.dataWriteCMD
  val execNameRef = ExecRef.getOrCreate(fullExecName)

  override def parse: ExecInfo = {
    // At this point we are sure that the wrapper is defined
    val wStub = DataWritingCommandExecParser.getWriteCMDWrapper(node).get
    val writeSupported = checker.isWriteFormatSupported(wStub.dataFormat)
    val duration = None
    val speedupFactor = checker.getSpeedupFactor(wStub.mappedExec)
    val finalSpeedup = if (writeSupported) speedupFactor else 1
    // We do not want to parse the node description to avoid mistakenly marking the node as RDD/UDF
    ExecInfo.createExecNoNode(
      sqlID = sqlID,
      exec = s"${node.name.trim} ${wStub.dataFormat.toLowerCase.trim}",
      expr = s"Format: ${wStub.dataFormat.toLowerCase.trim}",
      speedupFactor = finalSpeedup,
      duration = duration,
      nodeId = node.id,
      opType = OpTypes.WriteExec,
      isSupported = writeSupported,
      execRef = execNameRef
    )
  }
}

// A case class used to hold information of the parsed Sql-node.
// The purpose is to be able to reuse the code between different components.
case class DataWritingCmdWrapper(
    execName: String,
    mappedExec: String,
    dataFormat: String)

object DataWritingCommandExecParser {
  val dataWriteCMD = "DataWritingCommandExec"
  private val defaultPhysicalCMD = dataWriteCMD
  // made public for testing
  val insertIntoHadoopCMD = "InsertIntoHadoopFsRelationCommand"
  val appendDataExecV1 = "AppendDataExecV1"
  val overwriteByExprExecV1 = "OverwriteByExpressionExecV1"
  val atomicReplaceTableExec = "AtomicReplaceTableAsSelect"
  val atomicCreateTableExec = "AtomicCreateTableAsSelect"
  // Note: List of writeExecs that represent a physical command.
  // hardcode because InsertIntoHadoopFsRelationCommand uses this same exec
  // and InsertIntoHadoopFsRelationCommand doesn't have an entry in the
  // supported execs file Set(defaultPhysicalCMD)


  // A set of the logical commands that will be mapped to the physical write command
  // which has an entry in the speedupSheet
  // - InsertIntoHadoopFsRelationCommand is a logical command that does not have an entry in the
  //   supported execs file
  // - SaveIntoDataSourceCommand is generated by DeltaLake Table writes
  // - InsertIntoHiveTable is generated by Hive catalogue implementation
  // - OverwriteByExprExecV1 is a spark write Op but it has special handling for DeltaLake
  // - AppendDataExecV1 is a spark write Op but it has special handling for DeltaLake
  // - AtomicReplaceTableAsSelectExec and AtomicCreateTableAsSelectExec are spark write Op but
  //   they have special handling for DeltaLake
  private val logicalWriteCommands = Set(
    dataWriteCMD,
    insertIntoHadoopCMD,
    HiveParseHelper.INSERT_INTO_HIVE_LABEL,
    appendDataExecV1,
    overwriteByExprExecV1,
    atomicReplaceTableExec,
    atomicCreateTableExec
  )

  // Note: Defines a list of the execs that include formatted data.
  // This will be used to extract the format and then check whether the
  // format is supported or not. Set(dataWriteCMD, insertIntoHadoopCMD)

  // For now, we map the SaveIntoDataSourceCommand to defaultPhysicalCMD because we do not
  // have speedup entry for the deltaLake write operation
  private val logicalToPhysicalCmdMap = Map(
    insertIntoHadoopCMD -> defaultPhysicalCMD,
    HiveParseHelper.INSERT_INTO_HIVE_LABEL-> defaultPhysicalCMD,
    appendDataExecV1 -> appendDataExecV1,
    overwriteByExprExecV1 -> overwriteByExprExecV1,
    atomicReplaceTableExec -> "AtomicReplaceTableAsSelectExec",
    atomicCreateTableExec -> "AtomicCreateTableAsSelectExec"
  )

  // Map to hold the relation between writeExecCmd and the format.
  // This used for expressions that do not show the format as part of the description.
  private val specialWriteFormatMap = Map[String, String](
    // if appendDataExecV1 is not deltaLakeProvider, then we want to mark it as unsupported
    appendDataExecV1 -> "unknown",
    // if overwriteByExprExecV1 is not deltaLakeProvider, then we want to mark it as unsupported
    overwriteByExprExecV1 -> "unknown",
    // if atomicReplaceTableExec is not deltaLakeProvider, then we want to mark it as unsupported
    atomicReplaceTableExec -> "unknown",
    // if atomicCreateTableExec is not deltaLakeProvider, then we want to mark it as unsupported
    atomicCreateTableExec -> "unknown"
  )

  // Checks whether a node is a write CMD Exec
  def isWritingCmdExec(nodeName: String): Boolean = {
    logicalWriteCommands.exists(nodeName.contains(_)) || DeltaLakeHelper.accepts(nodeName)
  }

  def getPhysicalExecName(opName: String): String = {
    logicalToPhysicalCmdMap.getOrElse(opName, defaultPhysicalCMD)
  }

  def getWriteCMDWrapper(node: SparkPlanGraphNode): Option[DataWritingCmdWrapper] = {
    val processedNodeName = node.name.trim
    logicalWriteCommands.find(processedNodeName.contains(_)) match {
      case None => None
      case Some(wCmd) =>
        // get the dataformat from the map if it exists.
        // Otherwise, fallback to the string parser
        val dataFormat = specialWriteFormatMap.get(wCmd) match {
          case Some(f) => f
          case None =>
            if (HiveParseHelper.isHiveTableInsertNode(node.name)) {
              // Use Hive Utils to extract the format from insertIntoHiveTable based on the SerDe
              // class.
              HiveParseHelper.getWriteFormat(node)
            } else {
              // USe the default parser to extract the write-format
              getWriteFormatString(node.desc)
            }
        }
        val physicalCmd = getPhysicalExecName(wCmd)
        Some(DataWritingCmdWrapper(wCmd, physicalCmd, dataFormat))
    }
  }

  def parseNode(node: SparkPlanGraphNode,
      checker: PluginTypeChecker,
      sqlID: Long): ExecInfo = {
    if (DeltaLakeHelper.acceptsWriteOp(node)) {
      DeltaLakeHelper.parseNode(node, checker, sqlID)
    } else {
      DataWritingCommandExecParser(node, checker, sqlID).parse
    }
  }

  // gets the data format by parsing the description of the node
  def getWriteFormatString(node: String): String = {
    // We need to parse the input string to get the write format. Write format is either third
    // or fourth parameter in the input string. If the partition columns is provided, then the
    // write format will be the fourth parameter.
    // Example string in the eventlog:
    // Execute InsertIntoHadoopFsRelationCommand
    // gs://08f3844/, false, [op_cmpny_cd#25, clnt_rq_sent_dt#26], ORC, Map(path -> gs://08f3844)
    val parsedString = node.split(",", 3).last.trim // remove first 2 parameters from the string
    if (parsedString.startsWith("[")) {
      // Optional parameter is present in the eventlog. Get the fourth parameter by skipping the
      // optional parameter string.
      parsedString.split("(?<=],)").map(_.trim).slice(1, 2)(0).split(",")(0)
    } else {
      parsedString.split(",")(0) // return third parameter from the input string
    }
  }
}
