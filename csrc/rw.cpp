#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/torch.h>
#include <torch/library.h>

#include "cpu/rw_cpu.h"

#ifdef WITH_CUDA
#include "cuda/rw_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__rw_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__rw_cpu(void) { return NULL; }
#endif
#endif
#endif

CLUSTER_API std::tuple<torch::Tensor, torch::Tensor>
random_walk(torch::Tensor rowptr, torch::Tensor col, torch::Tensor start,
            int64_t walk_length, double p, double q) {
  return random_walk_cpu(rowptr, col, start, walk_length, p, q);
}

TORCH_LIBRARY(torch_cluster, m) {
  m.def("random_walk(Tensor rowptr, Tensor col, Tensor start, int walk_length, float p = 1, float q = 1) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(torch_cluster, CPU, m) {
  m.impl("random_walk", &random_walk);
}

#ifdef WITH_CUDA
TORCH_LIBRARY_IMPL(torch_cluster, CUDA, m) {
  m.impl("random_walk", &random_walk_cuda);
}
#endif
