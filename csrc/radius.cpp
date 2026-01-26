#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/torch.h>
#include <torch/library.h>

#include "cpu/radius_cpu.h"

#ifdef WITH_CUDA
#include "cuda/radius_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__radius_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__radius_cpu(void) { return NULL; }
#endif
#endif
#endif

CLUSTER_API torch::Tensor radius(torch::Tensor x, torch::Tensor y,
                                 std::optional<torch::Tensor> ptr_x,
                                 std::optional<torch::Tensor> ptr_y, double r,
                                 int64_t max_num_neighbors,
                                 int64_t num_workers,
                                 bool ignore_same_index) {
  return radius_cpu(x, y, ptr_x, ptr_y, r, max_num_neighbors, num_workers,
                    ignore_same_index);
}

TORCH_LIBRARY(torch_cluster, m) {
  m.def("radius(Tensor x, Tensor y, Tensor? ptr_x, Tensor? ptr_y, float r, int max_num_neighbors, int num_workers = 1, bool ignore_same_index = False) -> Tensor");
}

TORCH_LIBRARY_IMPL(torch_cluster, CPU, m) {
  m.impl("radius", &radius);
}

#ifdef WITH_CUDA
TORCH_LIBRARY_IMPL(torch_cluster, CUDA, m) {
  m.impl("radius", &radius_cuda);
}
#endif
