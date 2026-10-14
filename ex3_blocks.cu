#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 512
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(err)); \
        return 1; \
    } \
} while (0)

__global__ void add(const int *a, const int *b, int *c) {
    int i = blockIdx.x;
    c[i] = a[i] + b[i];
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
    add<<<N, 1>>>(d_a, d_b, d_c);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
        if (c[i] != a[i] + b[i]) ++errors;
    }
    printf("Verification: %d mismatches out of %d\n", errors, N);
    free(a); free(b); free(c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return errors ? 1 : 0;
}
