#include <godwin/godwin.h>
#include <math.h>

void logistic_regression(Tensor* x, Tensor* w, Tensor* b, Tensor* out){
    linear_forward(x, w, b, out);

    sigmoid(out, out);
}

void sigmoid(Tensor* x, Tensor* out){
    int num_elements = 1;


    for(int i = 0; i < x->n_dims; ++i){
        num_elements *= x->shape[i];
    }

    for (int i = 0; i < num_elements; ++i){
        out->data[i] = 1 / (1 + exp(-(x->data[i])));
    }
}
