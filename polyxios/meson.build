meson.add_dist_script(
  ['../tools/gitversion.py', '--meson-dist', '--write', 'polyxios/version.py']
)

if not fs.exists('version.py')
  generate_version = custom_target(
    'generate-version',
    install: true,
    build_always_stale: true,
    build_by_default: true,
    output: 'version.py',
    input: '../tools/gitversion.py',
    command: ['../tools/gitversion.py', '--meson-dist', '--write', 'polyxios/version.py'],
    install_dir: py.get_install_dir() / 'polyxios',
    install_tag: 'python-runtime',
  )
else
  py.install_sources(['version.py'], subdir: 'polyxios')
endif

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
    'fetcher.py'
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
