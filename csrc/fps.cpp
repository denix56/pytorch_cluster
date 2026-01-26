#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/torch.h>
#include <torch/library.h>

#include "cpu/fps_cpu.h"

#ifdef WITH_CUDA
#include "cuda/fps_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__fps_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__fps_cpu(void) { return NULL; }
#endif
#endif
#endif

CLUSTER_API torch::Tensor fps(torch::Tensor src, torch::Tensor ptr,
                              torch::Tensor ratio, bool random_start) {
  return fps_cpu(src, ptr, ratio, random_start);
}

TORCH_LIBRARY(torch_cluster, m) {
  m.def("fps(Tensor src, Tensor ptr, Tensor ratio, bool random_start = False) -> Tensor");
}

TORCH_LIBRARY_IMPL(torch_cluster, CPU, m) {
  m.impl("fps", &fps);
}

#ifdef WITH_CUDA
TORCH_LIBRARY_IMPL(torch_cluster, CUDA, m) {
  m.impl("fps", &fps_cuda);
}
#endif
