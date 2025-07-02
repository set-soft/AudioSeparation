# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# Why such a complex thing?
# Python imports are broken by design, and here we hit a huge limitation:
# 1. ComfyUI nodes MUST use relative imports, if you don't do it things like "import utils" becomes ambiguous
#    You might think this can be overcome polluting the sys.path, but this isn't true. If ComfyUI, or some
#    other node, already imported an module named "utils" you'll get the already imported module, not the one
#    you want. And polluting sys.path can make other nodes, or ComfyUI itself, import the wrong module when
#    they use a "non top-level import".
#    Conclusion: The only safe way to import from a ComfyUI node is by using relative imports.
# 2. We have tools in the "tool" subdir, they are intended to run as standalone scripts. They can't use
#    relative imports to access the node sub modules because you'll hit the error "ImportError: attempted
#    relative import with no known parent package". So imports in tools MUST be absolute. This can be solved
#    adding the node root to sys.path. Here we are not polluting the sys.path because we are the top-level.
#    Conclusion: This script is used to solve adding the correct path to sys.path
# 3. As you MUST use relative imports in the node sub-modules, when a sub-module depends on another sub-module
#    it will do something like "from ..XXXX", this is OK when all is relative. But tools are using absolute
#    imports, so you'll hit the error "ImportError: attempted relative import beyond top-level package"
#    Conclusion: All sub-modules must be wrapped by an umbrella sub-module, this is what "source" is.
#
# Structure:
# Node-name/       <-- is a module from ComfyUI point of view, dynamically imported using importlib
# \-- __init__.py  <-- needed because we are a module, relative imports
# \-- nodes.py     <-- relative imports
# |
# \-- source/      <-- umbrella package, just to solve relative imports
# |   \-- __init__.py
# |   |
# |   \-- utils/   <-- a submodule, uses relative imports
# |   |   \-- __init__.py
# |   |   \-- misc.py
# |   |
# |   \-- db/               <-- another submodule, uses relative imports
# |       \-- __init__.py
# |       \-- models_db.py  <-- Must use relative imports, can access ..utils.misc
# |
# \-- tool/
#     \-- batch_convert.py  <-- a tool, must use absolute import of "source"
#     |
#     \-- bootstrap/
#         \-- __init__.py   <-- THIS file, adds to sys.path to make "source" available

# --- BOOTSTRAP: Make the script aware of the project root ---
import os
import sys
# 1. Get the absolute path of the current script's directory (tool/)
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 2. Get the project root by going one directory up (MyAwesomeNode/)
project_root = os.path.dirname(script_dir)
# 3. Add the project root to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
