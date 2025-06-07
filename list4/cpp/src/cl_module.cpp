#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kmeans.hpp"
#include "dbscan.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cl_module, m) {
    m.doc() = "C++ implementation of clustering";
    
    py::class_<KMeans>(m, "KMeans")
        .def(py::init<int, std::string, int, int, int>(),
             py::arg("n_clusters") = 10,
             py::arg("init") = "k-means++",
             py::arg("n_init") = 1,
             py::arg("max_iter") = 300,
             py::arg("random_state") = 0)
        .def("fit_predict", &KMeans::fit_predict)
        .def_property_readonly("inertia_", &KMeans::get_inertia)
        .def_property_readonly("cluster_centers_", &KMeans::get_cluster_centers)
        .def_property_readonly("labels_", &KMeans::get_labels);

    py::class_<DBSCAN>(m, "DBSCAN")
        .def(py::init<double, int>(),
             py::arg("eps") = 0.5,
             py::arg("min_samples") = 5)
        .def("fit_predict", &DBSCAN::fit_predict)
        .def_property_readonly("labels_", &DBSCAN::get_labels)
        .def_property_readonly("n_clusters_", &DBSCAN::get_n_clusters)
        .def_property_readonly("n_noise_", &DBSCAN::get_n_noise);
}
