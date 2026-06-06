#include <godwin/godwin.h>

void linear_forward(Tensor* x, Tensor* w, Tensor* b, Tensor* out) {
    int temp_shape[]  = {w->shape[1], w->shape[0]};
    int temp_stride[] = {w->stride[1], w->stride[0]};

    Tensor w_t = {
        .data   = w->data,
        .shape  = temp_shape,
        .stride = temp_stride,
        .n_dims = w->n_dims
    };

    matmul(x, &w_t, out);
    tensor_add(out, b, out);
}
