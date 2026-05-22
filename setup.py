# file: setup.py
import os
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "bangla_entropy_backend",
        ["src/bindings.cpp"],
        include_dirs=["include"],
        cxx_std=17,
    ),
]

setup(
    name="bangla_entropy_tokenizer",
    version="0.1.0",
    author="Your Name",
    description="An entropy-based Bengali tokenizer with C++ backend",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    py_modules=["bangla_entropy_tokenizer"],
    zip_safe=False,
    python_requires=">=3.7",
)