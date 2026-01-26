#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/torch.h>
#include <torch/library.h>

#include "cpu/knn_cpu.h"

#ifdef WITH_CUDA
#include "cuda/knn_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__knn_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__knn_cpu(void) { return NULL; }
#endif
#endif
#endif

CLUSTER_API torch::Tensor knn(torch::Tensor x, torch::Tensor y,
                  std::optional<torch::Tensor> ptr_x,
                  std::optional<torch::Tensor> ptr_y, int64_t k, bool cosine,
                  int64_t num_workers) {
    TORCH_CHECK(!cosine, "`cosine` argument not supported on CPU");
    return knn_cpu(x, y, ptr_x, ptr_y, k, num_workers);
}

TORCH_LIBRARY(torch_cluster, m) {
  m.def("knn(Tensor a, Tensor b, Tensor? ptr_x, Tensor? ptr_y, int k, bool cosine = False, int num_workers = 1) -> Tensor");
}

TORCH_LIBRARY_IMPL(torch_cluster, CPU, m) {
  m.impl("knn", &knn);
}

#ifdef WITH_CUDA
    TORCH_LIBRARY_IMPL(torch_cluster, CUDA, m) {
      m.impl("knn", &knn_cuda);
    }
#endif
