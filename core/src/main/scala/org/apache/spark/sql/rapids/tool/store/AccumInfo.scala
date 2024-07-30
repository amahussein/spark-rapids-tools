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

package org.apache.spark.sql.rapids.tool.store

import scala.collection.mutable

import org.apache.spark.scheduler.AccumulableInfo
import org.apache.spark.sql.rapids.tool.util.EventUtils.parseAccumFieldToLong

class AccumInfo(val infoRef: AccMetaRef) {
  // TODO: Should we use sorted maps for stageIDs and taskIds?
  private val taskUpdatesMap: mutable.HashMap[Long, Long] =
    new mutable.HashMap[Long, Long]()
  private val stageValuesMap: mutable.HashMap[Int, Long] =
    new mutable.HashMap[Int, Long]()

  def addAccToStage(stageId: Int,
      accumulableInfo: AccumulableInfo,
      update: Option[Long] = None): Unit = {
    val value = accumulableInfo.value.flatMap(parseAccumFieldToLong)
    value match {
      case Some(v) =>
        stageValuesMap.put(stageId, v)
      case _ =>
        // this could be the case when a task update has triggered the stage update
        stageValuesMap.put(stageId, update.getOrElse(0L))
    }
  }

  def addAccToTask(stageId: Int, taskId: Long, accumulableInfo: AccumulableInfo): Unit = {
    val update = accumulableInfo.update.flatMap(parseAccumFieldToLong)
    // we have to update the stageMap if the stageId does not exist in the map
    var updateStageFlag = !taskUpdatesMap.contains(stageId)
    // TODO: Task can update an accum multiple times. Should account for that case.
    update match {
      case Some(v) =>
        taskUpdatesMap.put(taskId, v)
        // update teh stage if the task's update is non-zero
        updateStageFlag ||= v != 0
      case None =>
        taskUpdatesMap.put(taskId, 0L)
    }
    // update the stage value map if necessary
    if (updateStageFlag) {
      addAccToStage(stageId, accumulableInfo, update)
    }
  }

  def getStageIds: Set[Int] = {
    stageValuesMap.keySet.toSet
  }
}
