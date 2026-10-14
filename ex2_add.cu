#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(err)); \
        return 1; \
    } \
} while (0)

__global__ void add(const int *a, const int *b, int *c) {
    *c = *a + *b;
}

int main() {
    int a = 2, b = 7, c = 0;
    int *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_b, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_c, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice));
    add<<<1, 1>>>(d_a, d_b, d_c);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Result is %d\n", c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return c == 9 ? 0 : 1;
}
