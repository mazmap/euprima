# euprima
<b>Eu</b>ler (characteristic) <b>pr</b>ofiles for <b>im</b>age <b>a</b>nalysis.

# Disclaimer
This project was started as a prototype implementation for my bachelors thesis on the _efficient computation of Euler characteristic profiles for three-channel digital images_. 
Once the thesis has been accepted, it will also be linked to from this README. 

At the moment this repository is still very much a **work in progress** and hence very unorganized and messy. 
Once I finally come up with a better (i.e. more organized) folder structure, this repository will be reorganized accordingly. 

# Testing and Benchmarks
For executing the tests and benchmarks in `tests/` you will need various Python packages such as `numpy` and `pandas`. 
To install all required packages, you can use the [`uv`](https://docs.astral.sh/uv/) package/project manager. 
For the benchmarks against the [`eulearning`](https://github.com/vadimlebovici/eulearning) implementation by Hacquard and Lebovici, you have to clone the `eulearning/` folder from their repository into `tests/eulearning/`. 

# Documentation
I am working on a documentation with clear usage examples for the functions we provide via the `pybind11` interface.
For now, consult the test and benchmark files in `tests/` to get an idea of how the functions are supposed to be used. 
