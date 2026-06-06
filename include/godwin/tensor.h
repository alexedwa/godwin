#ifndef TENSOR
#define TENSOR

typedef struct{
    float* data;
    int* shape;
    int* stride;
    int n_dims;
} Tensor;

Tensor* tensor(int* shape, int n_dims);
void free_tensor(Tensor* t);
Tensor* tensor_ones(int* shape, int n_dims);
Tensor* tensor_custom(float* data, int* shape, int n_dims);

#endif
