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

"""Implementation of storage related functionalities."""

import os
from shutil import rmtree


class LocalFS:
    """Implementation of storage functionality for local disk."""
    @classmethod
    def remove_dir(cls, dir_path: str, fail_on_error: bool = True):
        try:
            rmtree(dir_path)
        except OSError as error:
            if fail_on_error:
                raise RuntimeError(f'Could not remove directory {dir_path}') from error

    @classmethod
    def make_dirs(cls, dir_path: str, exist_ok: bool = True):
        try:
            os.makedirs(dir_path, exist_ok=exist_ok)
        except OSError as error:
            raise RuntimeError(f'Error Creating directories {dir_path}') from error
