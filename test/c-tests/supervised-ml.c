#include <godwin/godwin.h>
#include <string.h>
#include <assert.h>
#include <math.h>

void linear_reg_test(){
    int shape[] = {2, 2};
    Tensor* x = tensor(shape, 2);
    Tensor* w = tensor(shape, 2);
    Tensor* b = tensor(shape, 2);

    float data1[] = {
        2.0f, 2.0f,
        6.0f, 8.0f
    };

    float data2[] = {
        2.0f, 4.0f,
        3.0f, 78.0f
    };

    float data3[] = {
        1.0f, 4.0f,
        6.0f, 8.0f
    };

    memcpy(x->data, data1, sizeof(float) * 4);
    memcpy(w->data, data2, sizeof(float) * 4);
    memcpy(b->data, data3, sizeof(float) * 4);

    Tensor* out = tensor(shape, 2);

    linear_forward(x, w, b, out);

    assert(out->data[0] == 13.0f);
    assert(out->data[1] == 168.0f);
    assert(out->data[2] == 45.0f);
    assert(out->data[3] == 648.0f);

    free_tensor(x);
    free_tensor(w);
    free_tensor(b);
    free_tensor(out);
}

void logistic_regression_test(){
    int shape[] = {2, 2}, shape2[] = {2};
    Tensor* x = tensor(shape, 2);
    Tensor* w = tensor(shape, 2);
    Tensor* b = tensor(shape2, 1);
    Tensor* out = tensor(shape, 2);

    float data1[] = {
        2.0f, 2.0f,
        6.0f, 8.0f
    };

    float data2[] = {
        2.0f, 2.0f,
        6.0f, 8.0f
    };


    float data3[] = {
        2.0f, 4.0
    };

    memcpy(x->data, data1, sizeof(float) * 4);
    memcpy(w->data, data2, sizeof(float) * 4);
    memcpy(b->data, data3, sizeof(float) * 2);

    logistic_regression(x, w, b, out);

    assert(fabs(out->data[0] == 0.999955) < 1e-4f);
    assert(out->data[1] == 1.0f);
    assert(out->data[2] == 1.0f);
    assert(out->data[3] == 1.0f);

    free_tensor(x);
    free_tensor(b);
    free_tensor(w);
    free_tensor(out);
}

int main(){
    linear_reg_test();
    printf("\nLinear Regression Tests passed!\n");
    
    logistic_regression_test();
    printf("\nLogistic Regression Tests passed!\n");
    return 0;
}
