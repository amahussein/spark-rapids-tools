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

/**
 * Accumulator Meta Reference
 * This maintains the reference to the metadata associated with an accumulable
 * @param id - Accumulable id
 * @param name - Reference to the accumulator name
 */
case class AccumMetaRef(id: Long, name: AccumNameRef) {
  def getName(): String = name.value
}

object AccumMetaRef {
  val EMPTY_ACCUM_META_REF: AccumMetaRef = new AccumMetaRef(0L, AccumNameRef.EMPTY_ACC_NAME_REF)
  def apply(id: Long, name: Option[String]): AccumMetaRef =
    new AccumMetaRef(id, AccumNameRef.getOrCreateAccumNameRef(name))
}
