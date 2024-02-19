
.PHONY: default repo

PROJ_NAME ?= "ml_template"

default:
	@echo "No target specified. Use one of the following commands:"
	@echo "make repo PROJ_NAME=<project_name>"

repo:
	@echo $(PROJ_NAME)
	@echo `pwd`
	/bin/rm -rf .git
	if [ "$(shell uname -s)" == "Darwin" ]; then \
		find . -type f -exec sed -i '' 's/ml_template/$(PROJ_NAME)/g' {} \; ; \
	else \
		find . -type f -exec sed -i 's/ml_template/$(PROJ_NAME)' {} \; ; \
	fi
	git init
	git add .gitignore
	git commit -m "feat: Initialize git"
	mv src/ml_template src/$(PROJ_NAME)
	echo "# $(PROJ_NAME)" > README.md
	git add .
	git commit -m "feat: Add templated project structure"

release:
	python -m build
	python -m twine upload --skip-existing --repository pypi dist/*
