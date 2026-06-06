#include <godwin/godwin.h>

void two_d_tensor(){
    int test_stride[] = {3, 1}, test_shape[] = {2, 3};
    Tensor* t = tensor(test_shape, 2);

    assert(t != NULL);
    assert(t->data != NULL);
    assert(t->n_dims == 2);

    for(int i = 0; i < t->n_dims; ++i){
        assert(t->shape[i] == test_shape[i]);
        assert(t->stride[i] == t->stride[i]);
    }

    free_tensor(t);
}

void three_d_tensor(){
    int test_shape[] = {2, 3, 4}, test_stride[] = {12, 4, 1};
    Tensor* t = tensor(test_shape, 3);

    assert(t != NULL);
    assert(t->data != NULL);
    assert(t->n_dims == 3);

    for(int i = 0; i < t->n_dims; ++i){
        assert(t->shape[i] == test_shape[i]);
        assert(t->stride[i] == test_stride[i]);
    }

    free_tensor(t);
}

void four_d_tensor(){
    int test_shape[] = {2, 4, 6, 9}, test_stride[] = {216, 54, 9, 1};
    Tensor* t = tensor(test_shape, 4);

    assert(t != NULL);
    assert(t->data != NULL);
    assert(t->n_dims == 4);

    for(int i = 0; i < t->n_dims; ++i){
        assert(t->shape[i] == test_shape[i]);
        assert(t->stride[i] == test_stride[i]);
    }

    free_tensor(t);
}

int main(){
    two_d_tensor();
    three_d_tensor();
    four_d_tensor();

    printf("\nTensor Initialisation Test passed!\n\n");

    return 0;
}

