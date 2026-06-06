#include <assert.h>
#include <string.h>

#include <godwin/godwin.h>

void gemv_test(){
    int shape1[] = {2, 3}, shape2[] = {3}, shape3[] = {2};
    Tensor* t1 = tensor(shape1, 2);
    Tensor* t2 = tensor(shape2, 1);

    float test_val1[] = {
        1.0, 4.0f, 5.0f,
        5.0f, 8.0f, 8.0f
    };

    float test_val2[] = {
        5.0f, 1.0f, 6.0f
    };

    memcpy(t1->data, test_val1, sizeof(float) * 6);
    memcpy(t2->data, test_val2, sizeof(float) * 3);

    Tensor* t3 = tensor(shape3, 1);

    matmul(t1, t2, t3);

    assert(t3->data[0] == 39.0f);
    assert(t3->data[1] == 81.0f);

    free_tensor(t1);
    free_tensor(t2);
    free_tensor(t3);
}

void gemm_test(){
    int shape[] = {2, 2};
    Tensor* t1 = tensor(shape, 2);
    Tensor* t2 = tensor(shape, 2);

    float test_val1[] = {
        1.0, 4.0,
        7.0, 4.0
    };

    float test_val2[] = {
        4.0, 5.0,
        1.0, 7.0
    };

    memcpy(t1->data, test_val1, sizeof(float) * 4);
    memcpy(t2->data, test_val2, sizeof(float) * 4);

    Tensor* t3 = tensor(shape, 2);

    matmul(t1, t2, t3);

    assert(t3->data[0] == 8.0f);
    assert(t3->data[1] == 33.0f);
    assert(t3->data[2] == 32.0f);
    assert(t3->data[3] == 63.0f);

    free_tensor(t1);
    free_tensor(t2);
    free_tensor(t3);
}

int main() {
    gemv_test();
    gemm_test();

    printf("\nMatrix Multiplication Test passed!\n\n");
    return 0;
}

