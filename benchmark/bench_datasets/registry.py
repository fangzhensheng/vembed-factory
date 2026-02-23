import importlib
import os
import pkgutil
from types import ModuleType


def discover_dataset_modules() -> dict[str, ModuleType]:
    modules: dict[str, ModuleType] = {}
    pkg_dir = os.path.dirname(__file__)
    for mod in pkgutil.iter_modules([pkg_dir]):
        if mod.ispkg:
            continue
        if mod.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__package__}.{mod.name}")
        name = getattr(module, "NAME", None)
        if isinstance(name, str) and name:
            modules[name] = module
    return modules
