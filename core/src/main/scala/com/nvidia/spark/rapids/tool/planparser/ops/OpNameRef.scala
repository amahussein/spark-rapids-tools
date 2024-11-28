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

import java.util.concurrent.ConcurrentHashMap

import org.apache.spark.sql.rapids.tool.util.StringUtils


trait OperatorRefTrait {
  def getOpName: String
  def getOpNameCSV: String
  def getOpType: String
  def getOpTypeCSV: String
}

abstract class OperatorRef(val value: String, val opType: OpTypes.OpType) extends OperatorRefTrait {
  val csvValue: String = StringUtils.reformatCSVString(value)
  val csvOpType: String = StringUtils.reformatCSVString(opType.toString)

  override def getOpName: String = value
  override def getOpNameCSV: String = csvValue
  override def getOpType: String = opType.toString
  override def getOpTypeCSV: String = csvOpType
}

case class OpNameRef(override val value: String,
    override val opType: OpTypes.OpType) extends OperatorRef(value, opType)

object OpNameRef {
  // Dummy AccNameRef to represent None accumulator names. This is an optimization to avoid
  // storing an option[string] for all accumulable names which leads to "get-or-else" everywhere.
  private val EMPTY_OP_NAME_REF: OpNameRef = new OpNameRef("N/A", OpTypes.Exec)
  // A global table to store reference to all accumulator names. The map is accessible by all
  // threads (different applications) running in parallel. This avoids duplicate work across
  // different threads.
  val OP_NAMES: ConcurrentHashMap[String, OpNameRef] = {
    val initMap = new ConcurrentHashMap[String, OpNameRef]()
    initMap.put(EMPTY_OP_NAME_REF.value, EMPTY_OP_NAME_REF)
    // Add the accum to the map because it is being used internally.
    initMap
  }

  def fromExpr(name: String): OpNameRef = {
    OP_NAMES.computeIfAbsent(name, OpNameRef.apply(_, OpTypes.Expr))
  }

  def fromExec(name: String): OpNameRef = {
    OP_NAMES.computeIfAbsent(name, OpNameRef.apply(_, OpTypes.Expr))
  }
}

case class UnsupportedExprOpRef(override val value: String,
    unsupportedReason: String) extends OperatorRef(value, OpTypes.Expr) {

}

object UnsupportedExprOpRef {
  def getOrCreate(value: String, unsupportedReason: String): UnsupportedExprOpRef = {
    val opRef = OpNameRef.fromExpr(value)
    UnsupportedExprOpRef(opRef.value, unsupportedReason)
  }
  def apply(opRef: OperatorRef, unsupportedReason: String): UnsupportedExprOpRef = {
    UnsupportedExprOpRef(opRef.value, unsupportedReason)
  }
}

object ExprOpRef {
  def apply(name: String): OperatorRef = {
    OpNameRef.fromExpr(name)
  }
}

object ExecOpRef {
  def apply(name: String): OperatorRef = {
    OpNameRef.fromExec(name)
  }
}
