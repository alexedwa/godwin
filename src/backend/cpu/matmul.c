#include <godwin/godwin.h>

void matmul(Tensor* a, Tensor* b, Tensor* c){
    if (a->n_dims == 2 && b->n_dims == 1){
        tensor_gemv(Tensor* a, Tensor* b, Tensor* c);
    }
    else if (a->n_dims == 2 && b->n_dims == 2){
        tensor_gemm(Tensor* a, Tensor* b, Tensor* c);
    }
    else{
        printf("Error: Incorrect number of tensor dimensions\n");
    }
}
