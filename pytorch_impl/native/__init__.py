# coding: utf-8
###
 # @file   __init__.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2019 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # Native (i.e. C++/CUDA) implementations automated building and loading.
###

# ---------------------------------------------------------------------------- #
# Initialization procedure

def _build_and_load():
  """ Incrementally rebuild all libraries and bind all local modules in the global.
  """
  glob = globals()
  # Standard imports
  import os
  import pathlib
  import traceback
  import warnings
  # External imports
  import torch
  import torch.utils.cpp_extension
  # Internal imports
  import tools
  # Constants
  base_directory     = pathlib.Path(__file__).parent.resolve()
  dependencies_file  = ".deps"
  debug_mode_envname = "NATIVE_OPT"
  debug_mode_in_env  = debug_mode_envname in os.environ
  if debug_mode_in_env:
    raw = os.environ[debug_mode_envname]
    value = raw.lower()
    if value in ["0", "n", "no", "false"]:
      debug_mode = True
    elif value in ["1", "y", "yes", "true"]:
      debug_mode = False
    else:
      tools.fatal("%r defined in the environment, but with unexpected soft-boolean %r" % (debug_mode_envname, "%s=%s" % (debug_mode_envname, raw)))
  else:
    debug_mode = __debug__
  cpp_std_envname    = "NATIVE_STD"
  cpp_std            = os.environ.get(cpp_std_envname, "c++14")
  ident_to_is_python = {"so_": False, "py_": True}
  source_suffixes    = {".cpp", ".cc", ".C", ".cxx", ".c++"}
  extra_cflags       = ["-Wall", "-Wextra", "-Wfatal-errors", "-std=%s" % cpp_std]
  if torch.cuda.is_available():
    source_suffixes.update(set((".cu" + suffix) for suffix in source_suffixes))
    source_suffixes.add(".cu")
    extra_cflags.append("-DTORCH_CUDA_AVAILABLE")
  extra_cuda_cflags  = ["-DTORCH_CUDA_AVAILABLE", "--expt-relaxed-constexpr", "-std=%s" % cpp_std]
  extra_ldflags      = ["-Wl,-L" + base_directory.root]
  extra_include_path = base_directory / "include"
  try:
    extra_include_paths = [str(extra_include_path.resolve())]
  except Exception:
    extra_include_paths = None
    warnings.warn("Not found include directory: " + repr(str(extra_include_path)))
  # Print configuration information
  cpp_std_message = "Native modules compiled with %s standard; (re)define %r in the environment to compile with another standard" % (cpp_std, "%s=<standard>" % cpp_std_envname)
  if debug_mode:
    tools.warning(cpp_std_message)
    tools.warning("Native modules compiled in debug mode; %sdefine %r in the environment or%s run python with -O/-OO options to compile in release mode" % ("re" if debug_mode_in_env else "", "%s=1" % debug_mode_envname, " undefine it and" if debug_mode_in_env else ""))
    extra_cflags += ["-O0", "-g"]
  else:
    quiet_envname = "NATIVE_QUIET"
    if quiet_envname not in os.environ:
      tools.trace(cpp_std_message)
      tools.trace("Native modules compiled in release mode; %sdefine %r in the environment or%s run python without -O/-OO options to compile in debug mode" % ("re" if debug_mode_in_env else "", "%s=0" % debug_mode_envname, " undefine it and" if debug_mode_in_env else ""))
      tools.trace("Define %r in the environment to hide these messages in release mode" % quiet_envname)
    extra_cflags += ["-O3", "-DNDEBUG"]
  # Variables
  done_modules = []
  fail_modules = []
  # Local procedures
  def build_and_load_one(path, deps=[]):
    """ Check if the given directory is a module to build and load, and if yes recursively build and load its dependencies before it.
    Args:
      path Given directory path
      deps Dependent module paths
    Returns:
      True on success, False on failure, None if not a module
    """
    nonlocal done_modules
    nonlocal fail_modules
    with tools.Context(path.name, "info"):
      ident = path.name[:3]
      if ident in ident_to_is_python.keys():
        # Is a module directory
        if len(path.name) <= 3 or path.name[3] == "_":
          tools.warning("Skipped invalid module directory name " + repr(path.name))
          return None
        if not path.exists():
          tools.warning("Unable to build and load " + repr(str(path.name)) + ": module does not exist")
          fail_modules.append(path) # Mark as failed
          return False
        is_python_module = ident_to_is_python[ident]
        # Check if already built and loaded, or failed
        if path in done_modules:
          if len(deps) == 0 and debug_mode:
            tools.info("Already built and loaded " + repr(str(path.name)))
          return True
        if path in fail_modules:
          if len(deps) == 0:
            tools.warning("Was unable to build and load " + repr(str(path.name)))
          return False
        # Check for dependency cycle (disallowed as they may mess with the linker)
        if path in deps:
          tools.warning("Unable to build and load " + repr(str(path.name)) + ": dependency cycle found")
          fail_modules.append(path) # Mark as failed
          return False
        # Build and load dependencies
        this_ldflags = list(extra_ldflags)
        depsfile = path / dependencies_file
        if depsfile.exists():
          for modname in depsfile.read_text().splitlines():
            res = build_and_load_one(base_directory / modname, deps + [path])
            if res == False: # Unable to build a dependency
              if len(deps) == 0:
                tools.warning("Unable to build and load " + repr(str(path.name)) + ": dependency " + repr(modname) + " build and load failed")
              fail_modules.append(path) # Mark as failed
              return False
            elif res == True: # Module and its sub-dependencies was/were built and loaded successfully
              this_ldflags.append("-Wl,--library=:" + str((base_directory / modname / (modname + ".so")).resolve()))
        # List sources
        sources = []
        for subpath in path.iterdir():
          if subpath.is_file() and ("").join(subpath.suffixes) in source_suffixes:
            sources.append(str(subpath))
        # Build and load this module
        try:
          res = torch.utils.cpp_extension.load(name=path.name, sources=sources, extra_cflags=extra_cflags, extra_cuda_cflags=extra_cuda_cflags, extra_ldflags=this_ldflags, extra_include_paths=extra_include_paths, build_directory=str(path), verbose=debug_mode, is_python_module=is_python_module)
          if is_python_module:
            glob[path.name[3:]] = res
        except Exception as err:
          tools.warning("Unable to build and load " + repr(str(path.name)) + ": " + str(err))
          fail_modules.append(path) # Mark as failed
          return False
        done_modules.append(path) # Mark as built and loaded
        return True
  # Main loop
  for path in base_directory.iterdir():
    if path.is_dir():
      try:
        build_and_load_one(path)
      except Exception as err:
        tools.warning("Exception while processing " + repr(str(path)) + ": " + str(err))
        with tools.Context("traceback", "trace"):
          traceback.print_exc()

# ---------------------------------------------------------------------------- #
# Initialization

import tools as _tools
with _tools.Context("native", None):
  _build_and_load()
del _tools
del _build_and_load
