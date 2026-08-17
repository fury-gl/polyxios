SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
# Warnings fail the build, here as well as on CI, so a local `make html` and
# the CI job agree. --keep-going reports every warning instead of stopping at
# the first. Override on the command line to build a draft: SPHINXOPTS=
SPHINXOPTS    = -W --keep-going

.PHONY: help html clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR)
