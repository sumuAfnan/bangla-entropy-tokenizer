// file: src/bindings.cpp
#include <pybind11/pybind11.h>
#include "tokenizer_engine.cpp"

namespace py = pybind11;

PYBIND11_MODULE(bangla_entropy_backend, m) {
    py::class_<TokenizerEngine>(m, "TokenizerEngine")
        .def(py::init<>())
        .def("train_from_file", &TokenizerEngine::train_from_file);
}