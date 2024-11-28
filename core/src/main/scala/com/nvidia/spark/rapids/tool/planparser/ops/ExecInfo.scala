/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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


package com.nvidia.spark.rapids.tool.planparser.ops

import scala.collection.mutable.{ArrayBuffer, WeakHashMap}

import org.apache.spark.sql.execution.ui.SparkPlanGraphNode
import org.apache.spark.sql.rapids.tool.{ExecHelper, RDDCheckHelper}

object OpActions extends Enumeration {
  type OpAction = Value
  val NONE, IgnoreNoPerf, IgnorePerf, Triage = Value
}

object OpTypes extends Enumeration {
  type OpType = Value
  val ReadExec, ReadRDD, WriteExec, Exec, Expr, UDF, DataSet = Value
}

object UnsupportedReasons extends Enumeration {
  type UnsupportedReason = Value
  val IS_UDF, CONTAINS_UDF,
  IS_DATASET, CONTAINS_DATASET,
  IS_UNSUPPORTED, CONTAINS_UNSUPPORTED_EXPR,
  UNSUPPORTED_IO_FORMAT = Value

  // Mutable map to cache custom reasons
  private val customReasonsCache = WeakHashMap.empty[String, Value]

  // Method to get or create a custom reason
  def CUSTOM_REASON(reason: String): Value = {
    customReasonsCache.getOrElseUpdate(reason, new Val(nextId, reason))
  }

  def reportUnsupportedReason(unsupportedReason: UnsupportedReason): String = {
    unsupportedReason match {
      case IS_UDF => "Is UDF"
      case CONTAINS_UDF => "Contains UDF"
      case IS_DATASET => "Is Dataset or RDD"
      case CONTAINS_DATASET => "Contains Dataset or RDD"
      case IS_UNSUPPORTED => "Unsupported"
      case CONTAINS_UNSUPPORTED_EXPR => "Contains unsupported expr"
      case UNSUPPORTED_IO_FORMAT => "Unsupported IO format"
      case customReason @ _  => customReason.toString
    }
  }
}

case class UnsupportedExecSummary(
  sqlId: Long,
  execId: Long,
  execRef: OperatorRef,
  opType: OpTypes.OpType,
  reason: UnsupportedReasons.UnsupportedReason,
  opAction: OpActions.OpAction) {
  def isExpression: Boolean = execRef.opType == OpTypes.Expr
  val finalOpType: String = if (opType.equals(OpTypes.UDF) || opType.equals(OpTypes.DataSet)) {
    s"${OpTypes.Exec.toString}"
  } else {
    s"${opType.toString}"
  }
  val unsupportedOperator: String = execRef.value
  val details: String = UnsupportedReasons.reportUnsupportedReason(reason)
}
case class ExecInfo(
  sqlID: Long,
  execRef: OperatorRef,
  expr: String,
  duration: Option[Long],
  nodeId: Long,
  opType: OpTypes.OpType,
  isSupported: Boolean,
  children: Option[Seq[ExecInfo]], // only one level deep
  var stages: Set[Int],
  var shouldRemove: Boolean,
  var unsupportedExecReason: String,
  unsupportedExprs: Seq[UnsupportedExprOpRef],
  dataSet: Boolean,
  udf: Boolean,
  shouldIgnore: Boolean,
  expressions: Seq[OperatorRef]) {

  // USed to avoid breaking other code looking for speedupfactor
  def speedupFactor: Double = 1.0

  def exec: String = execRef.value

  private def childrenToString = {
    val str = children.map { c =>
      c.map("       " + _.toString).mkString("\n")
    }.getOrElse("")
    if (str.nonEmpty) {
      "\n" + str
    } else {
      str
    }
  }

  override def toString: String = {
    s"exec: ${execRef.value}, expr: $expr, sqlID: $sqlID, " +
      s"duration: $duration, nodeId: $nodeId, " +
      s"isSupported: $isSupported, children: " +
      s"$childrenToString, stages: ${stages.mkString(",")}, " +
      s"shouldRemove: $shouldRemove, shouldIgnore: $shouldIgnore"
  }

  def setStages(stageIDs: Set[Int]): Unit = {
    stages = stageIDs
  }

  def appendToStages(stageIDs: Set[Int]): Unit = {
    stages ++= stageIDs
  }

  def setShouldRemove(value: Boolean): Unit = {
    shouldRemove ||= value
  }

  def setUnsupportedExecReason(reason: String): Unit = {
    unsupportedExecReason = reason
  }

  def determineUnsupportedReason(reason: String,
    knownReason: UnsupportedReasons.Value): UnsupportedReasons.Value = {
    if (reason.nonEmpty) UnsupportedReasons.CUSTOM_REASON(reason) else knownReason
  }

  def getOpAction: OpActions.OpAction = {
    // shouldRemove is checked first because sometimes an exec could have both flag set to true,
    // but then we care about having the "NoPerf" part
    if (isSupported) {
      OpActions.NONE
    } else {
      if (shouldRemove) {
        OpActions.IgnoreNoPerf
      } else if (shouldIgnore) {
        OpActions.IgnorePerf
      } else  {
        OpActions.Triage
      }
    }
  }

  private def getUnsupportedReason: UnsupportedReasons.UnsupportedReason = {
    if (children.isDefined) {
      // TODO: Handle the children
    }
    if (udf) {
      UnsupportedReasons.CONTAINS_UDF
    } else if (dataSet) {
      if (unsupportedExprs.isEmpty) { // case when the node itself is a DataSet or RDD
        UnsupportedReasons.IS_DATASET
      } else {
        UnsupportedReasons.CONTAINS_DATASET
      }
    } else if (unsupportedExprs.nonEmpty) {
      UnsupportedReasons.CONTAINS_UNSUPPORTED_EXPR
    } else {
      opType match {
        case OpTypes.ReadExec | OpTypes.WriteExec => UnsupportedReasons.UNSUPPORTED_IO_FORMAT
        case _ => UnsupportedReasons.IS_UNSUPPORTED
      }
    }
  }

  def getUnsupportedExecSummaryRecord(execId: Long): Seq[UnsupportedExecSummary] = {
    // Get the custom reason if it exists
    val execUnsupportedReason = determineUnsupportedReason(unsupportedExecReason,
      getUnsupportedReason)

    // Initialize the result with the exec summary
    val res = ArrayBuffer(UnsupportedExecSummary(sqlID, execId, execRef, opType,
      execUnsupportedReason, getOpAction))

    // TODO: Should we iterate on exec children?
    // add the unsupported expressions to the results, if there are any custom reasons add them
    // to the result appropriately
    if (unsupportedExprs.nonEmpty) {
      val exprKnownReason = execUnsupportedReason match {
        case UnsupportedReasons.CONTAINS_UDF => UnsupportedReasons.IS_UDF
        case UnsupportedReasons.CONTAINS_DATASET => UnsupportedReasons.IS_DATASET
        case UnsupportedReasons.UNSUPPORTED_IO_FORMAT => UnsupportedReasons.UNSUPPORTED_IO_FORMAT
        case _ => UnsupportedReasons.IS_UNSUPPORTED
      }

      unsupportedExprs.foreach { expr =>
        val exprUnsupportedReason = determineUnsupportedReason(expr.unsupportedReason,
          exprKnownReason)
        res += UnsupportedExecSummary(sqlID, execId, expr, OpTypes.Expr,
          exprUnsupportedReason, getOpAction)
      }
    }
    res
  }
}

object ExecInfo {
  // Used to create an execInfo without recalculating the dataSet or Udf.
  // This is helpful when we know that node description may contain some patterns that can be
  // mistakenly identified as UDFs
  def createExecNoNode(sqlID: Long,
    exec: String,
    expr: String,
    duration: Option[Long],
    nodeId: Long,
    opType: OpTypes.OpType,
    isSupported: Boolean,
    children: Option[Seq[ExecInfo]], // only one level deep
    stages: Set[Int] = Set.empty,
    shouldRemove: Boolean = false,
    unsupportedExecReason: String = "",
    unsupportedExprs: Seq[UnsupportedExprOpRef] = Seq.empty,
    dataSet: Boolean = false,
    udf: Boolean = false,
    expressions: Seq[String] = Seq.empty): ExecInfo = {
    // Set the ignoreFlag
    // 1- we ignore any exec with UDF
    // 2- we ignore any exec with dataset
    // 3- Finally we ignore any exec matching the lookup table
    // if the opType is RDD, then we automatically enable the datasetFlag
    val finalDataSet = dataSet || opType.equals(OpTypes.ReadRDD)
    val shouldIgnore = udf || finalDataSet || ExecHelper.shouldIgnore(exec)
    val removeFlag = shouldRemove || ExecHelper.shouldBeRemoved(exec)
    val finalOpType = if (udf) {
      OpTypes.UDF
    } else if (dataSet) {
      // we still want the ReadRDD to stand out from other RDDs. So, we use the original
      // dataSetFlag
      OpTypes.DataSet
    } else {
      opType
    }
    // Set the supported Flag
    val supportedFlag = isSupported && !udf && !finalDataSet
    ExecInfo(
      sqlID,
      ExecOpRef(exec),
      expr,
      duration,
      nodeId,
      finalOpType,
      supportedFlag,
      children,
      stages,
      removeFlag,
      unsupportedExecReason,
      unsupportedExprs,
      finalDataSet,
      udf,
      shouldIgnore,
      expressions.map(ExecOpRef(_))
    )
  }

  def apply(
    node: SparkPlanGraphNode,
    sqlID: Long,
    exec: String,
    expr: String,
    duration: Option[Long],
    nodeId: Long,
    isSupported: Boolean,
    children: Option[Seq[ExecInfo]], // only one level deep
    stages: Set[Int] = Set.empty,
    shouldRemove: Boolean = false,
    unsupportedExecReason:String = "",
    unsupportedExprs: Seq[UnsupportedExprOpRef] = Seq.empty,
    dataSet: Boolean = false,
    udf: Boolean = false,
    opType: OpTypes.OpType = OpTypes.Exec,
    expressions: Seq[String] = Seq.empty): ExecInfo = {
    // Some execs need to be trimmed such as "Scan"
    // Example: Scan parquet . ->  Scan parquet.
    // scan nodes needs trimming
    val nodeName = node.name.trim
    // we don't want to mark the *InPandas and ArrowEvalPythonExec as unsupported with UDF
    val containsUDF = udf || ExecHelper.isUDF(node)
    // check is the node has a dataset operations and if so change to not supported
    val rddCheckRes = RDDCheckHelper.isDatasetOrRDDPlan(nodeName, node.desc)
    val ds = dataSet || rddCheckRes.isRDD

    // if the expression is RDD because of the node name, then we do not want to add the
    // unsupportedExpressions because it becomes bogus.
    val finalUnsupportedExpr = if (rddCheckRes.nodeDescRDD) {
      Seq.empty[UnsupportedExprOpRef]
    } else {
      unsupportedExprs
    }
    createExecNoNode(
      sqlID,
      exec,
      expr,
      duration,
      nodeId,
      opType,
      isSupported,
      children,
      stages,
      shouldRemove,
      unsupportedExecReason,
      finalUnsupportedExpr,
      ds,
      containsUDF,
      expressions)
  }
}
