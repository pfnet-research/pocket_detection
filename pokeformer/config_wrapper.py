from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import (
    IO,
    Any,
    Callable,
    Iterable,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from omegaconf import DictConfig, OmegaConf
from omegaconf.basecontainer import BaseContainer

logger = logging.getLogger(__name__)

ConfigContainer = TypeVar("ConfigContainer", bound=BaseContainer)


def load_class(class_name: str, default_module_name: Optional[str] = None) -> Type:
    if "." not in class_name:
        if default_module_name is not None:
            class_name = default_module_name + "." + class_name
        else:
            raise ValueError(f"cannot load class from empty module path: {class_name}")
    name_list = class_name.split(".")
    modnm = ".".join(name_list[:-1])
    clsnm = name_list[-1]
    logger.info(f"loading {clsnm} from {modnm}")
    m = importlib.import_module(modnm)
    cls: Type = getattr(m, clsnm)
    return cls


def _try_load_class(clsname: str, load_from: str) -> type | None:
    cls_type: type
    try:
        cls_type = load_class(clsname, load_from)
        return cls_type
    except AttributeError:
        return None


def search_class(clsname: str, candidate_modules: Iterable[str]) -> type:
    cls_type: type | None = None
    for module in candidate_modules:
        cls_type = _try_load_class(clsname, module)
        if cls_type is not None:
            return cls_type
    raise ValueError(f"Cannot load class {clsname} from candidate modules")


def _validate_config_impl(cls: Type, user_config: ConfigContainer) -> ConfigContainer:
    return validate_config_by_dataclass(cls.get_config_class(), user_config)


def validate_config_by_dataclass(
    dat_cls: Type, config: ConfigContainer
) -> ConfigContainer:
    schema = OmegaConf.structured(dat_cls)
    typed_config = OmegaConf.merge(schema, config)
    return cast(ConfigContainer, typed_config)


def validate_config_by_class(
    cls: Type, user_config: ConfigContainer
) -> ConfigContainer:
    if not hasattr(cls, "get_config_class"):
        logger.warning(
            f"cls {cls} does not have get_config_class method. "
            "no type check is performed."
        )
        return user_config
    return _validate_config_impl(cls, user_config)


def obj_from_config(
    class_factory: Callable[[str], Any],
    config: ConfigContainer,
    *args: Any,
    allow_default_class: bool = False,
) -> Tuple[Any, Optional[ConfigContainer]]:
    if "class" not in config:
        if not allow_default_class:
            raise ValueError('key "class" not found in config')
        else:
            cls_name = ""
    else:
        cls_name = config["class"]
    cls = class_factory(cls_name)
    if hasattr(cls, "get_config_class"):
        copied_config = cast(DictConfig, config).copy()
        if "class" in copied_config:
            copied_config.pop("class")
        typed_config = _validate_config_impl(cls, copied_config)
        result_config = OmegaConf.merge({"class": cls_name}, typed_config)
        return cls.from_config(typed_config, *args), cast(
            ConfigContainer, result_config
        )
    else:
        logger.warning(f"class {cls} does not have get_config_class method.")
        return cls(config, *args), None


def _merge_configs(
    yaml_files: Sequence[str],
    ovwr_config: Optional[BaseContainer] = None,
) -> BaseContainer:
    user_config: BaseContainer = OmegaConf.create()
    for fname in yaml_files:
        conf = OmegaConf.load(fname)
        user_config = OmegaConf.merge(user_config, conf)
    if ovwr_config is not None:
        user_config = OmegaConf.merge(user_config, ovwr_config)
    return user_config


T = TypeVar("T", bound="ConfigWrapper")


class ClassFactoryProto(Protocol):
    @staticmethod
    def get_class(class_name: str) -> Any:
        pass


class ConfigWrapper:
    @classmethod
    def from_cli(cls: Type[T], schema_dataclass: Type) -> T:
        cli_config = OmegaConf.from_cli()
        user_config: BaseContainer
        if "yaml" in cli_config:
            logger.info("Using yaml config from CLI.")
            yaml_files: Sequence[str]
            if isinstance(cli_config.yaml, str):
                logger.info(f"merge from yaml file: {cli_config.yaml}")
                yaml_files = [cli_config.yaml]
            elif isinstance(cli_config.yaml, Sequence):
                logger.info(f"merge from yaml file list: {cli_config.yaml}")
                yaml_files = cli_config.yaml
            elif isinstance(cli_config.yaml, DictConfig):
                logger.info(
                    f"merge from yaml file dict: {cli_config.yaml}, "
                    "keys of dict ignored."
                )
                yaml_files = [v for v in cli_config.yaml.values()]
            else:
                raise RuntimeError(f"unknown type: {cli_config.yaml}")
            copied_cli_config = cli_config.copy()
            copied_cli_config.pop("yaml")
            # Overwrite by CLI specified options here
            user_config = _merge_configs(yaml_files, copied_cli_config)
        else:
            user_config = cli_config
        return cls(schema_dataclass, user_config)

    @classmethod
    def from_yaml_files(
        cls: Type[T],
        schema_dataclass: Type,
        yaml_files: Sequence[str],
    ) -> T:
        user_config = _merge_configs(yaml_files)
        return cls(schema_dataclass, user_config)

    @classmethod
    def from_omegaconf(
        cls: Type[T], schema_dataclass: Type, config: BaseContainer
    ) -> T:
        # alias for __init__
        return cls(schema_dataclass, config)

    def __init__(
        self,
        schema_dataclass: Type,
        config: BaseContainer,
        dump_config_path_key: str = "dump_config_path",
    ) -> None:
        logger.debug(f"config: {config}")
        OmegaConf.resolve(config)

        schema = OmegaConf.structured(schema_dataclass)
        self.user_config = config
        self.dict_config = cast(DictConfig, OmegaConf.merge(schema, config))
        self.result_config = self.dict_config.copy()
        self._dump_config(config_key=dump_config_path_key)

    def create_obj(
        self,
        class_factory: ClassFactoryProto,
        key: str,
        *args: Any,
        allow_default_class: bool = False,
    ) -> Any:
        obj, res_cfg = obj_from_config(
            class_factory.get_class,
            self.dict_config[key],
            *args,
            allow_default_class=allow_default_class,
        )
        logger.debug(f"{key}: {obj}")
        self.result_config[key] = res_cfg
        return obj

    def dump_result_config(self, path: Union[str, Path, IO[Any]]) -> None:
        OmegaConf.save(self.result_config, path)

    def dump_orig_config(self, path: Union[str, Path, IO[Any]]) -> None:
        OmegaConf.save(self.dict_config, path)

    def _dump_config(self, config_key: str = "dump_config_path") -> None:
        if config_key not in self.dict_config:
            return
        config = self.dict_config[config_key]
        if config is None:
            return
        path = Path(config)
        outdir = path.parent
        outdir.mkdir(parents=True, exist_ok=True)
        self.dump_result_config(path)
