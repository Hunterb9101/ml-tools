
.PHONY: default

PROJ_NAME ?= "ml_template"

default:
	@echo $(PROJ_NAME)
	@echo `pwd`
	/bin/rm -rf .git
	find . -type f -exec sed -i 's/ml_template/$(PROJ_NAME)' {} \;
	git init
	git add .gitignore
	git commit -m "feat: Initialize git"
	mv src/ml_template src/$(PROJ_NAME)
	echo "# $(PROJ_NAME)" > README.md
	git add .
	git commit -m "feat: Add templated project structure"