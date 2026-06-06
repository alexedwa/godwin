#include <godwin/godwin.h>

void tensor_add(Tensor* a, Tensor* b, Tensor* out){
    int a_row = a->shape[0], a_col = a->shape[1];

    for(int i = 0; i < a_row; i++){
        for(int j = 0; j < a_col; j++){
            int a_stride = i * a->stride[0] + j * a->stride[1];
            int out_stride = i * out->stride[0] + j * out->stride[1];

            out->data[out_stride] = a->data[a_stride] + b->data[j * b->stride[0]];
        }
    }
}

void tensor_sub(Tensor* a, Tensor* b, Tensor* out){
    int a_row = a->shape[0], a_col = a->shape[1];

    for(int i = 0; i < a_row; i++){
        for(int j = 0; j < a_col; j++){
            int a_stride = i * a->stride[0] + j * a->stride[1];
            int out_stride = i * out->stride[0] + j * out->stride[1];

            out->data[out_stride] = a->data[a_stride] - b->data[j * b->stride[0]];
        }
    }

}

void tensor_mul(Tensor* a, Tensor* b, Tensor* out){
    int a_row = a->shape[0], a_col = a->shape[1];

    for(int i = 0; i < a_row; i++){
        for(int j = 0; j < a_col; j++){
            int a_stride = i * a->stride[0] + j * a->stride[1];
            int out_stride = i * out->stride[0] + j * out->stride[1];

            out->data[out_stride] = a->data[a_stride] * b->data[j * b->stride[0]];
        }
    }
}

void tensor_div(Tensor* a, Tensor* b, Tensor* out){
    int a_row = a->shape[0], a_col = a->shape[1];

    for(int i = 0; i < a_row; i++){
        for(int j = 0; j < a_col; j++){
            int a_stride = i * a->stride[0] + j * a->stride[1];
            int out_stride = i * out->stride[0] + j * out->stride[1];

            out->data[out_stride] = a->data[a_stride] / b->data[j * b->stride[0]];
        }
    }

}

