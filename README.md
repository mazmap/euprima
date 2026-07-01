# euprima
Euler (characteristic) profiles for image analysis.

# Disclaimer
This project was started for my bachelors thesis on the _efficient computation of Euler characteristic profiles for three-channel digital images_. 
Once the thesis has been accepted, it will also be linked to from this README. 

At the moment this repository is still very much a **work in progress** and hence very unorganized and messy. 
Once I finally come up with a better (i.e. more organized) folder structure, this repository will be reorganized accordingly. 

The CMAKE Build-Setup has only been tested on my own 2021 Apple Silicon Macbook Pro running Sequoia version 15.7.7. 

For the benchmarks against the [`eulearning`](https://github.com/vadimlebovici/eulearning) implementation by Hacquard and Lebovici, we include their full implementation in the `tests/` folder. 

# Building
To build the [pybind11](https://pybind11.readthedocs.io/en/stable/index.html) interface, create a `build/` folder, navigate into it and execute `cmake ..` followed by `make`. 
The C++ implementation needs no external dependencies. 

For executing the tests and benchmarks in `tests/` you will need various Python packages such as `numpy` and `pandas`. 

# Documentation
I am working on a documentation with clear usage examples for the functions we provide via the `pybind11` interface.
For now, consult the test and benchmark files in `tests/` to get an idea of how the functions are supposed to be used. 
