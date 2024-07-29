
.PHONY: install docs

install:
	pip install .

install_dev:
	# https://stackoverflow.com/a/28842733
	pip install -e .[dev]

docs: install_dev
	@cd doc_sources && make html

style: install_dev
	black flexdata

conda_package:
	conda build conda/ -c astra-toolbox -c nvidia

pypi_wheels:
	python -m build --wheel
