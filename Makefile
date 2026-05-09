.PHONY: help install test lint check-manifest verify clean build publish-test publish

help:
	@echo "Targets:"
	@echo "  install         Install package in editable mode with dev deps"
	@echo "  test            Run test suite"
	@echo "  lint            Run flake8 via tox"
	@echo "  check-manifest  Verify MANIFEST.in vs VCS"
	@echo "  verify          Run tests + lint + check-manifest"
	@echo "  clean           Remove build artifacts"
	@echo "  build           Clean, build sdist+wheel, twine check"
	@echo "  publish-test    Upload current dist/ to TestPyPI"
	@echo "  publish         Upload current dist/ to PyPI"

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/

lint:
	tox -e style

check-manifest:
	tox -e check-manifest

verify: test lint check-manifest

clean:
	rm -rf dist/ build/ hyppo_hsi.egg-info hyppo.egg-info

build: clean
	python -m build
	twine check dist/*

publish-test:
	twine upload -r testpypi dist/*

publish:
	twine upload dist/*
