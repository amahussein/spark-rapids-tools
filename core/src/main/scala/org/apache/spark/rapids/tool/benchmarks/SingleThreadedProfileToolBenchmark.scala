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

package org.apache.spark.rapids.tool.benchmarks

import com.nvidia.spark.rapids.tool.profiling.{ProfileArgs, ProfileMain}

object ProfToolBenchmark extends BenchmarkBase {
  override def runBenchmarkSuite(inputArgs: Array[String]): Unit = {
    // Currently the input arguments are assumed to be common across cases
    // This will be improved in a follow up PR to enable passing as a config
    // file with argument support for different cases
    runBenchmark("Benchmark_Profiling_CSV") {
      val (prefix, suffix) = inputArgs.splitAt(inputArgs.length - 1)
      addCase("Profiling_CSV") { _ =>
        ProfileMain.mainInternal(new ProfileArgs(prefix :+ "--num-threads"
          :+ "1" :+ "--csv" :+ suffix.head),
          enablePB = true)
      }
      run()
    }
  }
}
