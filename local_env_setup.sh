cd /Users/ryanwest/OMSCS/cs6440/final-project

conda env create -f environment.yml

conda env export > environment_development.yml

conda env update --file environment.yml --prune
