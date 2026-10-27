# Fuzzing polyxios readers

One [Atheris](https://github.com/google/atheris) target per format that is
read from bytes. Every target asks the same of its reader: arbitrary input
gives back either a mesh that passes `polyxios.validate` (offsets running
from 0 to the connectivity length) or a `PolyxiosError`. Any other exception,
a hang past the timeout, or an allocation past the RSS limit is a finding.

HDF5 and netCDF formats are not fuzzed here: libhdf5 and libnetcdf parse
their bytes and are fuzzed upstream. Index formats (`.pvd`, `.pvt*`, `.vtm`)
are reached through the files they point at.

## Setup

Atheris needs a clang that ships libFuzzer. Linux distributions' clang does;
on macOS, Apple's does not, so build against Homebrew's LLVM:

```sh
python -m venv .fuzz && . .fuzz/bin/activate
CLANG_BIN=/opt/homebrew/opt/llvm/bin/clang pip install "atheris>=2.3"   # macOS
pip install ".[laz]"          # a regular, non-editable install
python fuzz/make_corpus.py    # seeds in fuzz/corpus/<target>/
```

## Running

```sh
mkdir -p fuzz/work/stl fuzz/crashes && cp fuzz/corpus/stl/* fuzz/work/stl/
python fuzz/fuzz_stl.py fuzz/work/stl -max_total_time=600 -timeout=10 \
    -rss_limit_mb=3072 -max_len=65536 -artifact_prefix=fuzz/crashes/stl-
```

libFuzzer stops at the first finding and writes the input that caused it to
`fuzz/crashes/stl-crash-<sha1>`. Replay it with
`python fuzz/fuzz_stl.py fuzz/crashes/stl-crash-<sha1>`.

Only Python code is instrumented for coverage (`atheris.instrument_all`).
The Cython modules (`_core`, `_vtk_parse`, `_vtp_base64`, `_lzf`) are
exercised but give no coverage feedback unless polyxios is built with
`CFLAGS=-fsanitize=fuzzer-no-link,address`, as OSS-Fuzz builds it.

## Adding a target

Add the name and extension to `TARGETS` in `_harness.py`, copy any
`fuzz_<name>.py`, and give the writer's flavours a line in `OPTIONS` in
`make_corpus.py` if it has more than one.

A fixed finding gets a regression test beside the codec's other tests in
`tests/codecs/`, built from the minimised input.
