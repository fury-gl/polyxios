py.install_sources(
  [
    '__init__.py',
    '_backend.py',
    '_core.pxd',
    '_core_fallback.py',
    '_element_types.py',
    '_registry.py',
    '_types.py',
    'exceptions.py',
    'transforms.py',
    'validate.py',
  ],
  subdir: 'polyxios',
)

omp_dep = dependency('openmp', required: false)

py.extension_module(
  '_core',
  '_core.pyx',
  include_directories: inc_np,
  dependencies: omp_dep.found() ? [py_dep, omp_dep] : [py_dep],
  install: true,
  subdir: 'polyxios',
)

subdir('codecs')
