// C shim for accessing llama.cpp struct fields from Rust
// Compiled by build.rs, linked into the bench binary

#include <stdio.h>
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

void bench_shim_load_backends(const char *dir_path) {
    if (dir_path && dir_path[0] != '\0') {
        ggml_backend_load_all_from_path(dir_path);
    } else {
        ggml_backend_load_all();
    }
}

struct llama_model_params bench_shim_model_params(int32_t n_gpu_layers) {
    struct llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    return params;
}

struct llama_context_params bench_shim_context_params(
    uint32_t n_ctx,
    uint32_t n_batch,
    uint32_t n_ubatch,
    int32_t n_threads,
    int32_t flash_attn,
    int32_t type_k,
    int32_t type_v
) {
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = n_batch;
    params.n_ubatch = n_ubatch;
    params.n_threads = n_threads;
    params.n_threads_batch = n_threads;
    params.flash_attn_type = (enum llama_flash_attn_type)flash_attn;
    params.type_k = (enum ggml_type)type_k;
    params.type_v = (enum ggml_type)type_v;
    params.no_perf = false;
    params.offload_kqv = true;
    return params;
}
