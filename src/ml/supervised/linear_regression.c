#include <godwin/godwin.h>

void linear_forward(Tensor* x, Tensor* w, Tensor* b, Tensor* out) {
    int temp_shape[] = {w->shape[1], w->shape[0]};

    Tensor* w_t = tensor(temp_shape, w->n_dims);
    w_t->data = w->data; 

    matmul(x, w_t, out);
    tensor_add(out, b, out);

    free(w_t->stride);
    free(w_t->shape);
    free(w_t);
}
