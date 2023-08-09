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

"""
Abstract representation of a file path that can access local/URI values.
Similar to cloudpathlib project, this implementation uses dict registry to
register an implementation. However, the path representation is built on top of
pyArrow FS API. As a result, there is no need to write a full storage client to
access remote files. This comes with a tradeoff in providing limited set of file
operations.
"""

import abc
import sys
from collections import defaultdict
from functools import cached_property
from pathlib import Path as PathlibPath
from typing import Union, Type, TypeVar, Any, Dict, Callable, overload, Optional, TYPE_CHECKING, List

from pyarrow import fs
from pyarrow.fs import FileType, FileSystem, FileInfo
from pydantic import ValidationError, model_validator, FilePath, AnyHttpUrl, StringConstraints, TypeAdapter
from pydantic.dataclasses import dataclass
from pydantic_core import PydanticCustomError
from typing_extensions import Annotated

from ..exceptions import (
    InvalidProtocolPrefixError,
    FSMismatchError, ASFileExistsError
)

from ..utils.util import get_path_as_uri

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from .fs import AsFs


CSPPathString = Annotated[str, StringConstraints(pattern=r'^\w+://.*')]


class ASPathImplementation:
    """
    A metaclass implementation that describes the behavior of the path class
    """
    name: str
    _path_class: Type['ASFsPath']
    _fs_class: Type['AsFs']
    _fslib_class: Type['FileSystem']

    @property
    def fs_class(self) -> Type['AsFs']:
        return self._fs_class

    @fs_class.setter
    def fs_class(self, clazz):
        self._fs_class = clazz

    @property
    def path_class(self) -> Type['ASFsPath']:
        return self._path_class

    @path_class.setter
    def path_class(self, clazz):
        self._path_class = clazz

    @property
    def fslib_class(self) -> Type['FileSystem']:
        return self._fslib_class

    @fslib_class.setter
    def fslib_class(self, clazz):
        self._fslib_class = clazz


path_impl_registry: Dict[str, ASPathImplementation] = defaultdict(ASPathImplementation)

T = TypeVar('T')
ASFsPathT = TypeVar('ASFsPathT', bound='ASFsPath')


def register_path_class(key: str) -> Callable[[Type[ASFsPathT]], Type[ASFsPathT]]:
    def decorator(cls: Type[ASFsPathT]) -> Type[ASFsPathT]:
        if not issubclass(cls, ASFsPath):
            raise TypeError('Only subclasses of CloudPath can be registered.')
        path_impl_registry[key].path_class = cls
        cls._path_meta = path_impl_registry[key]  # pylint: disable=protected-access
        return cls

    return decorator


@dataclass
class AcceptedFilePath:
    """
    Class used to represent input that can be accepted as file paths.
    """
    file_path: Union[CSPPathString, FilePath, AnyHttpUrl]
    extensions: Optional[List[str]] = None

    @model_validator(mode='after')
    def validate_file_extensions(self) -> 'AcceptedFilePath':
        if self.extensions:
            if not any(str(self.file_path).endswith(ext) for ext in self.extensions):
                raise PydanticCustomError(
                    'file_path',
                    (f'Invalid file extension for of input file {self.file_path}. '
                     f'Accepted: {self.extensions}'))

    def is_http_file(self) -> bool:
        try:
            TypeAdapter(AnyHttpUrl).validate_python(self.file_path)
            return True
        except ValidationError:
            # ignore
            return False


class ASFsPathMeta(abc.ABCMeta):
    """
    Class meta used to add hooks to the type of the ASFsPath as needed.
    This is used typically to dynamically assign any class type as subclass to ASFsPath.
    """

    @overload
    def __call__(
            cls: Type[T], entry_path: ASFsPathT, *args: Any, **kwargs: Any
    ) -> ASFsPathT:
        ...

    @overload
    def __call__(
            cls: Type[T], entry_path: Union[str, 'ASFsPath'], *args: Any, **kwargs: Any
    ) -> T:
        ...

    def __call__(
            cls: Type[T], entry_path: Union[str, ASFsPathT], *args: Any, **kwargs: Any
    ) -> Union[T, ASFsPathT]:
        # cls is a class that is the instance of this metaclass, e.g., CloudPath
        if not issubclass(cls, ASFsPath):
            raise TypeError(
                f'Only subclasses of {ASFsPath.__name__} can be instantiated from its meta class.'
            )
        if isinstance(entry_path, str):
            # convert the string to uri if it is not
            entry_path = get_path_as_uri(entry_path)
        # Dispatch to subclass if base ASFsPath
        if cls is ASFsPath:
            for path_clz_entry in path_impl_registry.values():
                path_class = path_clz_entry.path_class
                if path_class is not None and path_class.is_valid_asfspath(
                        entry_path, raise_on_error=False
                ):
                    # Instantiate path_class instance
                    new_obj = object.__new__(path_class)
                    path_class.__init__(new_obj, entry_path, *args, **kwargs)
                    return new_obj
        new_obj = object.__new__(cls)
        cls.__init__(new_obj, entry_path, *args, **kwargs)  # type: ignore[type-var]
        return new_obj


class ASFsPath(metaclass=ASFsPathMeta):
    """
    Base class for storage systems, based on pyArrow's FileSystem. The class provides support for
    URI/local file path like "gs://", "s3://", "abfss://", "file://".

    Instances represent an absolute path in a storage with filesystem path semantics, and for basic
    operations like streaming, and opening a file for read/write operations.
    Only basic metadata about file entries, such as the file size and modification time, is made
    available.

    Examples
    --------
    Create a new path subclass from a gs URI:

    >>> gs_path = ASFsPath('gs://bucket-name/folder_00/subfolder_01')
    <as_pytools.storagelib.gs.gcpath.GSAsPath object at ...>

    or from S3 URI:

    >>> s3_path = ASFsPath('s3://bucket-name/folder_00/subfolder_01')
    <as_pytools.storagelib.s3.s3path.S3AsPath object at ...>

    or from local file URI:

    >>> local_path1, local_path2 = (ASFsPath('~/my_folder'), ASFsPath('file:///my_folder'))
    <as_pytools.storagelib.local.localpath.LocalAsPath object at ...,
      as_pytools.storagelib.local.localpath.LocalAsPath object at ...>

    Print the data from the file with `open_input_file()`:

    >>> with as_path.open_input_file() as f:
    ...     print(f.readall())
    b'data'

    Check that path is file
    >>> gs_path = ASFsPath('gs://bucket-name/folder_00/subfolder_01')
    >>> print(gs_path.is_file())
    """

    protocol_prefix: str
    _path_meta: ASPathImplementation

    @staticmethod
    def is_file_path(file_path: Union[str, PathlibPath],
                     extensions: List[str] = None,
                     raise_on_error: bool = True):
        try:
            TypeAdapter(AcceptedFilePath).validate_python({'file_path': file_path, 'extensions': extensions})
            return True
        except ValidationError as err:
            if raise_on_error:
                raise err
        return False

    @overload
    @classmethod
    def is_valid_asfspath(cls, path: str, raise_on_error: bool = ...) -> bool:
        ...

    @overload
    @classmethod
    def is_valid_asfspath(cls, path: 'ASFsPath', raise_on_error: bool = ...) -> TypeGuard[Self]:
        ...

    @classmethod
    def is_valid_asfspath(
            cls, path: Union[str, 'ASFsPath'], raise_on_error: bool = False
    ) -> Union[bool, TypeGuard[Self]]:
        valid = cls.is_protocol_prefix(str(path))
        if raise_on_error and not valid:
            raise InvalidProtocolPrefixError(
                f'"{path}" is not a valid path since it does not start with "{cls.protocol_prefix}"'
            )
        return valid

    def __init__(
            self,
            entry_path: Union[str, Self],
            fs_obj: Optional['AsFs'] = None
    ) -> None:
        self.is_valid_asfspath(entry_path, raise_on_error=True)
        self._fpath = str(entry_path)
        if fs_obj is None:
            if isinstance(entry_path, ASFsPath):
                fs_obj = entry_path.fs_obj
            else:
                fs_obj = self._path_meta.fs_class.get_default_client()
        if not isinstance(fs_obj, self._path_meta.fs_class):
            raise FSMismatchError(
                f'Client of type [{fs_obj.__class__}] is not valid for cloud path of type '
                f'[{self.__class__}]; must be instance of [{self._path_meta.fs_class}], or '
                f'None to use default client for this cloud path class.'
            )
        self.fs_obj = fs_obj
        self._file_info = None

    def __str__(self) -> str:
        return self._fpath

    @classmethod
    def is_protocol_prefix(cls, value: str) -> bool:
        return value.lower().startswith(cls.protocol_prefix.lower())

    @cached_property
    def no_prefix(self) -> str:
        return self._fpath[len(self.protocol_prefix):]

    def _pull_file_info(self) -> FileInfo:
        return self.fs_obj.get_file_info(self.no_prefix)

    @cached_property
    def file_info(self) -> FileInfo:
        self._file_info = self._pull_file_info()
        return self._file_info

    def is_file(self):
        return self.file_info.is_file

    def is_dir(self):
        return self.file_info.type == FileType.Directory

    def exists(self) -> bool:
        f_info = self.file_info
        return f_info.type in [FileType.File, FileType.Directory]

    def base_name(self) -> str:
        return self.file_info.base_name

    def create_dirs(self, exist_ok: bool = True):
        if not exist_ok:
            # check that the file does not exist
            if self.exists():
                raise ASFileExistsError(f'Path already Exists: {self}')
        self.fs_obj.create_dir(self.no_prefix)
        # force the file information object to be retrieved again by invalidating the cached property
        if 'file_info' in self.__dict__:
            del self.__dict__['file_info']

    def open_input_stream(self):
        return self.fs_obj.open_input_stream(self.no_prefix)

    def open_output_stream(self):
        return self.fs_obj.open_output_stream(self.no_prefix)

    @classmethod
    def download_files(cls, src_url: str, dest_url: str):
        fs.copy_files(src_url, dest_url)
