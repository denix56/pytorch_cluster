#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/torch.h>
#include <torch/library.h>

#include "extensions.h"

#ifdef WITH_CUDA
#include "cuda/nearest_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__nearest_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__nearest_cpu(void) { return NULL; }
#endif
#endif
#endif

TORCH_LIBRARY(torch_cluster, m) {
  m.def("nearest(Tensor x, Tensor y, Tensor ptr_x, Tensor ptr_y) -> Tensor");
}

#ifdef WITH_CUDA
TORCH_LIBRARY_IMPL(torch_cluster, CUDA, m) {
  m.impl("nearest", &nearest_cuda);
}
#endif
