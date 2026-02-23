import importlib
import os
import pkgutil

# Automatically import all modules in this directory to trigger registry
# This allows users to add new model implementations by simply adding a new .py file
package_dir = os.path.dirname(__file__)
for _finder, module_name, _ispkg in pkgutil.iter_modules([package_dir]):
    if module_name != "__init__":
        importlib.import_module(f".{module_name}", package=__name__)
