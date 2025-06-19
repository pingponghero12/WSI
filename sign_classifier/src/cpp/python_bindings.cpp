#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "sign_classifier.h"

namespace py = pybind11;

PYBIND11_MODULE(sign_classifier, m) {
    m.doc() = "CUDA Neural Network for Sign Classification";
    
    py::class_<SignClassifierNetwork>(m, "SignClassifierNetwork")
        .def(py::init<int>(),
             py::arg("batch_size"),
             "Initialize sign classification network")
        .def("forward", &SignClassifierNetwork::forward,
             "Forward pass through the network",
             py::arg("X"))
        .def("train_step", &SignClassifierNetwork::train_step,
             "Single training step",
             py::arg("X"), py::arg("Y"), py::arg("learning_rate"))
        .def("calculate_accuracy", &SignClassifierNetwork::calculate_accuracy,
             "Calculate accuracy",
             py::arg("predictions"), py::arg("labels"))
        .def("train", &SignClassifierNetwork::train,
             "Train the network",
             py::arg("X"), py::arg("Y"), py::arg("epochs"), py::arg("learning_rate"))
        .def_static("generate_training_data", &SignClassifierNetwork::generate_training_data,
                    "Generate training data - returns (X, Y) tuple",
                    py::arg("n_samples"));
    
    m.attr("__version__") = "1.0.0";
}
