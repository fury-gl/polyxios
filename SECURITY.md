# Security policy

## Supported versions

Security fixes go into the latest minor release. Older release series do not
receive backports; upgrade to the newest version on
[PyPI](https://pypi.org/project/polyxios/) to get them.

| Version              | Supported |
| -------------------- | --------- |
| latest minor release | yes       |
| older releases       | no        |

## Reporting a vulnerability

Please do not open a public issue, pull request or discussion for a
security problem.

Report it privately through GitHub's
[private vulnerability reporting](https://github.com/fury-gl/polyxios/security/advisories/new)
on this repository. Include:

- the polyxios version, Python version and platform;
- the smallest input file or script that reproduces the problem, attached
  to the report rather than linked;
- what happens (crash, hang, memory growth, file written where it should
  not be, ...) and what you expected.

We aim to acknowledge a report within 7 days and to agree with you on a
disclosure date once the issue is understood. Fixes are released as a patch
version and published as a GitHub security advisory, crediting the reporter
unless you ask otherwise.

## Scope

polyxios reads files that often come from untrusted sources, so the parsers
are the main attack surface. In scope, for example:

- memory-safety bugs in the compiled extensions (Cython modules);
- a crafted file that makes a reader allocate memory or loop out of
  proportion to its size, or hang;
- a reader that follows a path or reference inside a file (sidecar data,
  external entities, archive members) to read or write outside what the
  caller asked for;
- `polyxios.fetcher` accepting a download whose checksum does not match, or
  writing outside `POLYXIOS_HOME`.

Out of scope:

- an exception (`ReadError`, `ValueError`, ...) raised on a malformed file:
  refusing bad input is the intended behaviour;
- bugs in optional dependencies (h5py, netCDF4, lazrs) or in numpy, which
  should be reported to those projects;
- issues that require the attacker to already control the Python
  environment, `POLYXIOS_HOME` or the `POLYXIOS_MODELS_URL` setting.

## Verifying a release

Every release is built and published by GitHub Actions from its tag, with
no token stored in the repository:

- the files on PyPI carry a provenance attestation, shown on each file's
  page under *Provenance*, that ties them to the `release.yml` run that
  built them;
- the GitHub Release attaches a CycloneDX SBOM,
  `polyxios-<version>.cdx.json`, listing polyxios and the dependency
  versions a `pip install` resolved on the day of the release;
- the repository's [OpenSSF Scorecard](https://scorecard.dev/viewer/?uri=github.com/fury-gl/polyxios)
  is recomputed weekly.

Ordinary bugs, including wrong output on a valid file, belong in the
[issue tracker](https://github.com/fury-gl/polyxios/issues).
