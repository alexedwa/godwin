#include <godwin/godwin.h>
#include <math.h>

void softmax_regression(Tensor* x, Tensor* w, Tensor* b, Tensor* out){
    linear_forward(x, w, b, out);

    softmax(out, out);
}

void softmax(Tensor* x, Tensor* out){
    int rows = x->shape[0], cols = x->shape[1];

    for(int i = 0; i < rows; ++i){
        float sum = 0.0f;
        for(int j = 0; j < cols; ++j){
            int stride = i * x->stride[0] + j * x->stride[1];
            out->data[stride] = expf(x->data[stride]);
            sum += out->data[stride];
        }

        for(int j = 0; j < cols; ++j){
            int stride = i * x->stride[0] + j * x->stride[1];
            out->data[stride] /= sum;
        }
    }
}
