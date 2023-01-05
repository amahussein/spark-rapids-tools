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

"""Definition of global utilities and helpers methods."""

import logging.config
import os
import secrets
import string
import subprocess
import sys
from typing import Callable


def gen_random_string(str_length: int) -> str:
    return ''.join(secrets.choice(string.hexdigits) for _ in range(str_length))


def resource_path(resource_name: str) -> str:
    # pylint: disable=import-outside-toplevel
    if sys.version_info < (3, 9):
        import importlib_resources
    else:
        import importlib.resources as importlib_resources

    pkg = importlib_resources.files('spark_rapids_tools')
    return pkg / 'resources' / resource_name


class ToolLogging:
    """Holds global utilities used for logging."""
    @classmethod
    def get_log_dict(cls, args):
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'simple': {
                    'format': '{asctime} {levelname} {name}: {message}',
                    'style': '{',
                },
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'simple',
                },
            },
            'root': {
                'handlers': ['console'],
                'level': 'DEBUG' if args.get('debug') else 'INFO',
            },
        }

    @classmethod
    def enable_debug_mode(cls):
        set_rapids_tools_env('LOG_DEBUG', 'True')

    @classmethod
    def is_debug_mode_enabled(cls):
        return get_rapids_tools_env('LOG_DEBUG')

    @classmethod
    def get_and_setup_logger(cls, type_label: str, debug_mode: bool = False):
        debug_enabled = bool(get_rapids_tools_env('LOG_DEBUG', debug_mode))
        logging.config.dictConfig(cls.get_log_dict({'debug': debug_enabled}))
        logger = logging.getLogger(type_label)
        log_file = get_rapids_tools_env('LOG_FILE')
        if log_file:
            # create file handler which logs even debug messages
            fh = logging.FileHandler(log_file)
            # TODO: set the formatter and handler for file logging
            # fh.setLevel(log_level)
            # fh.setFormatter(ExtraLogFormatter())
            logger.addHandler(fh)
        return logger


def find_full_rapids_tools_env_key(actual_key: str) -> str:
    return f'RAPIDS_TOOLS_{actual_key}'


def get_rapids_tools_env(k: str, def_val=None):
    val = os.environ.get(find_full_rapids_tools_env_key(k))
    if val is None and def_val is not None:
        return def_val
    return val


def set_rapids_tools_env(k: str, val):
    os.environ[f'RAPIDS_TOOLS_{k}'] = str(val)


def exec_sys_cmd(cmd, expected: int = 0, fail_ok: bool = False,
                 cmd_input: str = None,
                 process_streams_cb: Callable = None) -> (str, str):
    """Run command and check return code, capture output etc."""
    stdout = subprocess.PIPE
    stderr = subprocess.PIPE

    if not isinstance(cmd, str):
        # this is a list of string. we need to join the arguments
        actual_cmd = ' '.join(cmd)
    else:
        actual_cmd = cmd
    # pylint: disable=subprocess-run-check
    if cmd_input is None:
        c = subprocess.run(actual_cmd, executable='/bin/bash', shell=True, stdout=stdout, stderr=stderr)
    else:
        # apply input to the command
        c = subprocess.run(actual_cmd,
                           executable='/bin/bash',
                           shell=True,
                           input=cmd_input,
                           text=True,
                           stdout=stdout, stderr=stderr)
    # pylint: enable=subprocess-run-check
    if not fail_ok:
        if expected != c.returncode:
            stderror_content = c.stderr if isinstance(c.stderr, str) else c.stderr.decode('utf-8')
            std_error_lines = [f'\t| {line}' for line in stderror_content.splitlines()]
            stderr_str = ''
            if len(std_error_lines) > 0:
                error_lines = '\n'.join(std_error_lines)
                stderr_str = f'\n{error_lines}'
            cmd_err_msg = f'Error invoking CMD <{c.args}>: {stderr_str}'
            raise RuntimeError(f'{cmd_err_msg}')

    std_output = c.stdout if isinstance(c.stdout, str) else c.stdout.decode('utf-8')
    std_error = c.stderr if isinstance(c.stderr, str) else c.stderr.decode('utf-8')
    if process_streams_cb is not None:
        process_streams_cb(std_output, std_error)
    return std_output, std_error
