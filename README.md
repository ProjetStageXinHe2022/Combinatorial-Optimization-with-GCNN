# Combinatorial-Optimization-with-GCNN
## Install ecole
The part of this project to get features depends on library [ecole](https://markdown.com.cn).But we made some changes in ecole, so you need to install our version by :https://github.com/LeonaireHo/ecole.
The environment required to build ecole through conda：
~~~
conda env create -n ecole -f dev/conda.yaml

conda activate ecole
conda config --append channels conda-forge
conda config --set channel_priority flexible
~~~
In the Ecole source repository,configure with CMake using
~~~
./dev/run.sh configure -D ECOLE_DEVELOPER=ON
~~~
In the Ecole source repository,install with setup.py
~~~
python setup.py install
~~~

## Generate of Dataset
~~~
~~~
