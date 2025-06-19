#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cuda_neural_network.h"

namespace py = pybind11;

PYBIND11_MODULE(cuda_neural_network, m) {
    m.doc() = "CUDA-accelerated Neural Network for deep learning";
    
    py::class_<CudaNeuralNetwork>(m, "CudaNeuralNetwork")
        .def(py::init<int, int, int, int>(),
             py::arg("input_size"), py::arg("hidden_size"), 
             py::arg("output_size"), py::arg("batch_size"),
             "Initialize CUDA Neural Network")
        .def("forward", &CudaNeuralNetwork::forward,
             "Perform forward pass and return predictions",
             py::arg("X"))  // No longer need predictions parameter
        .def("train_step", &CudaNeuralNetwork::train_step,
             "Perform one training step",
             py::arg("X"), py::arg("Y"), py::arg("learning_rate"))
        .def("calculate_accuracy", &CudaNeuralNetwork::calculate_accuracy,
             "Calculate prediction accuracy",
             py::arg("predictions"), py::arg("labels"))
        .def("train", &CudaNeuralNetwork::train,
             "Train the neural network",
             py::arg("X"), py::arg("Y"), py::arg("epochs"), py::arg("learning_rate"));
    
    m.attr("__version__") = "1.0.0";
}
