#include <godwin/godwin.h>

Tensor* tensor(int* shape, int n_dims){
    Tensor* t = malloc(sizeof(Tensor));

    t->shape = malloc(n_dims * sizeof(int));
    t->stride = malloc(n_dims * sizeof(int));
    t->n_dims = n_dims;

    // copying the shape values
    for(int i = 0; i < n_dims; ++i){
        t->shape[i] = shape[i];
    }

    t->stride[n_dims - 1] = 1;

    for(int i = n_dims - 2; i >= 0; --i){
        t->stride[i] = t->stride[i + 1] * shape[i + 1];
    }

    int total_elements = 1;
    for (int i = 0; i < n_dims; ++i){
        total_elements *= shape[i];
    }
    t->data = malloc(total_elements * sizeof(float));

    return t;
}

void free_tensor(Tensor* t){
    free(t->shape);
    free(t->stride);
    free(t->data);

    t->shape == NULL;
    t->stride == NULL;
    t->data == NULL;

    free(t);
}
