#include <godwin/godwin.h>

void matmul(Tensor* a, Tensor* b, Tensor* c){
    if (a->n_dims == 2 && b->n_dims == 1){
        tensor_gemv(a, b, c);
    }
    else if (a->n_dims == 2 && b->n_dims == 2){
        tensor_gemm(a, b, c);
    }
    else{
        printf("Error: Incorrect number of tensor dimensions\n");
    }
}

void tensor_gemv(Tensor* a, Tensor* b, Tensor* out){
    int a_row = a->shape[0];
    int a_col = a->shape[1];


    for(int i = 0; i < a_row; ++i){
        float val = 0.0f;
        for(int j = 0; j<  a_col; ++j){
            val += a->data[i * a->stride[0] + j] * b->data[j];
        }
        out->data[i] = val;
    }
}

void tensor_gemm(Tensor* a, Tensor* b, Tensor* out){
    int a_row = b->shape[0];
    int a_col = a->shape[1];
    int b_col = b->shape[1];

    for(int i = 0; i < a_row; ++i){
        for(int j = 0; j < b_col; ++j){
            float val = 0.0f;
            for(int k = 0; k < a_col; ++k){
                int a_stride = i * a->stride[0] + k * a->stride[1];
                int b_stride = k * b->stride[0] + j * b->stride[1];

                val += a->data[a_stride] * b->data[b_stride];
            }
            out->data[i * out->stride[0] + j * out->stride[1]] = val;
        }
    }
}
