# euprima
<b>Eu</b>ler (characteristic) <b>pr</b>ofiles for <b>im</b>age <b>a</b>nalysis.

# Disclaimer
This project was started as a prototype implementation for my bachelors thesis on the _efficient computation of Euler characteristic profiles for three-channel digital images_. 
Once the thesis has been accepted, it will also be linked to from this README. 

At the moment this repository is still very much a **work in progress** and hence very unorganized and messy. 
Once I finally come up with a better (i.e. more organized) folder structure, this repository will be reorganized accordingly. 

# Installing the Python package
You can find the Python interface on PyPI as the [`euprima`](https://pypi.org/project/euprima) package. 
It can be installed using 
```bash
pip install euprima
```
For the installation you will need a C++ compiler, `cmake` and `pybind11`. 

# Closing the repository
This repository uses the [`eulearning`](https://github.com/vadimlebovici/eulearning) repository by Hacquard and Lebovici as a Git submodule for benchmarking.
Clone the repository using the `--recursive` flag to pull down the benchmark files automatically:

```bash
git clone --recursive https://github.com/mazmap/euprima.git
```

In case you forgot the `--recursive` flag, you can execute
```bash
git submodule update --init --recursive
```
after cloning

# Testing and Benchmarks
For executing the tests and benchmarks in `tests/` you will need various Python packages such as `numpy` and `pandas`. 
To install all required packages, you can use the [`uv`](https://docs.astral.sh/uv/) package/project manager. 
All dependencies are listed in `tests/pyproject.toml` (including the `euprima` package). 
For the benchmarks against the [`eulearning`](https://github.com/vadimlebovici/eulearning) implementation by Hacquard and Lebovici, you have to clone the `eulearning/` folder from their repository into `tests/eulearning/` or clone it as a submodule as described above. 

# Documentation
I am working on a documentation with clear usage examples for the functions we provide via the `pybind11` interface.
For now, consult the test and benchmark files in `tests/` to get an idea of how the functions are supposed to be used. 
