#ifndef MATMUL
#define MATMUL

void matmul(Tensor* a, Tensor* b, Tensor* c);
void tensor_gemv(Tensor* a, Tensor* b, Tensor* out);
void tensor_gemm(Tensor* a, Tensor* b, Tensor* out);

#endif
