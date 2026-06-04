py.install_sources(
  [
    '__init__.py',
    '_obj.py',
    '_ply.py',
    '_vtk.py',
    '_vtp.py',
    '_vtr.py',
  ],
  subdir: 'polyxios/codecs',
)

py.extension_module(
  '_vtk_parse',
  '_vtk_parse.pyx',
  include_directories: inc_np,
  dependencies: py_dep,
  install: true,
  subdir: 'polyxios/codecs',
)

py.extension_module(
  '_vtp_base64',
  '_vtp_base64.pyx',
  include_directories: inc_np,
  dependencies: py_dep,
  install: true,
  subdir: 'polyxios/codecs',
)
