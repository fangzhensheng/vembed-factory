import importlib
import os
import pkgutil

# Automatically import all modules in this directory to trigger registry
package_dir = os.path.dirname(__file__)
for _, module_name, _ in pkgutil.iter_modules([package_dir]):
    if module_name != "__init__":
        importlib.import_module(f".{module_name}", package=__name__)
