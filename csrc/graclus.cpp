#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/torch.h>
#include <torch/library.h>

#include "cpu/graclus_cpu.h"

#ifdef WITH_CUDA
#include "cuda/graclus_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__graclus_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__graclus_cpu(void) { return NULL; }
#endif
#endif
#endif

CLUSTER_API torch::Tensor graclus(torch::Tensor rowptr, torch::Tensor col,
                                  std::optional<torch::Tensor> optional_weight) {
  return graclus_cpu(rowptr, col, optional_weight);
}

TORCH_LIBRARY(torch_cluster, m) {
  m.def("graclus(Tensor rowptr, Tensor col, Tensor? weight = None) -> Tensor");
}

TORCH_LIBRARY_IMPL(torch_cluster, CPU, m) {
  m.impl("graclus", &graclus);
}

#ifdef WITH_CUDA
TORCH_LIBRARY_IMPL(torch_cluster, CUDA, m) {
  m.impl("graclus", &graclus_cuda);
}
#endif
