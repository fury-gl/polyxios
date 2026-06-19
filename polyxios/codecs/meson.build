py.install_sources(
  [
    '__init__.py',
    '_obj.py',
    '_ply.py',
    '_vtk.py',
    '_pvti.py',
    '_pvtp.py',
    '_pvtr.py',
    '_pvts.py',
    '_pvtu.py',
    '_vtk_xml.py',
    '_vti.py',
    '_vtm.py',
    '_vtp.py',
    '_vtr.py',
    '_vts.py',
    '_vtu.py',
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
