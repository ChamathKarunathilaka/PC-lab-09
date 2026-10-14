#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 10000000
#define THREADS_PER_BLOCK 512
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(err)); \
        return 1; \
    } \
} while (0)

__global__ void vecMul(const int *a, const int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

void random_ints(int *x, int count) {
    for (int i = 0; i < count; ++i) x[i] = rand() % 100;
}

int main() {
    const size_t bytes = (size_t)N * sizeof(int);
    int *a = (int *)malloc(bytes), *b = (int *)malloc(bytes), *c = (int *)malloc(bytes);
    if (!a || !b || !c) { fprintf(stderr, "Host allocation failed\n"); return 1; }
    random_ints(a, N);
    random_ints(b, N);

    int *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    printf("Launching %d blocks x %d threads\n", blocks, THREADS_PER_BLOCK);
    vecMul<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

    for (int i = N - 1000; i < N; ++i)
        printf("%d) %d x %d = %d\n", i, a[i], b[i], c[i]);

    int errors = 0;
    for (int i = 0; i < N; ++i)
        if (c[i] != a[i] * b[i]) ++errors;
    printf("Verification: %d mismatches out of %d\n", errors, N);

    free(a); free(b); free(c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return errors ? 1 : 0;
}
