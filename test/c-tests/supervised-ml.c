#include <godwin/godwin.h>

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

    assert(out->data[0] == 11.0f);
    assert(out->data[1] == 170.0f);
    assert(out->data[2] == 37.0f);
    assert(out->data[3] == 654.0f);

    free_tensor(x);
    free_tensor(w);
    free_tensor(b);
    free_tensor(out);

    printf("\nTest: Linear Regression Passed!\n");
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
        2.0f, 4.0f
    };

    memcpy(x->data, data1, sizeof(float) * 4);
    memcpy(w->data, data2, sizeof(float) * 4);
    memcpy(b->data, data3, sizeof(float) * 2);

    logistic_regression(x, w, b, out); 

    assert(out->data[0] == 1.0f);
    assert(out->data[1] == 1.0f);
    assert(out->data[2] == 1.0f);
    assert(out->data[3] == 1.0f);

    free_tensor(x);
    free_tensor(b);
    free_tensor(w);
    free_tensor(out);

    printf("\nTest: Logistical Regression Test Passed!\n");
}

void softmax_regression_test(){
    int shape1[] = {2, 2}, shape2[] = {3, 2}, shape3[] = {3}, shape4[] = {2, 3}; 
    Tensor* x = tensor(shape1, 2);
    Tensor* w = tensor(shape2, 2);
    Tensor* b = tensor(shape3, 1);
    Tensor* out = tensor(shape4, 2);

    float data1[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    };

    float data2[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f
    };


    float data3[] = {
        0.0f, 0.0f, 0.0f
    };

    memcpy(x->data, data1, sizeof(float) * 4);
    memcpy(w->data, data2, sizeof(float) * 6);
    memcpy(b->data, data3, sizeof(float) * 3);

    softmax_regression(x, w, b, out);

    assert(fabs(out->data[0] - 0.705384) < 1e-4f);
    assert(fabs(out->data[1] - 0.259496) < 1e-4f);
    assert(fabs(out->data[2] - 0.035119) < 1e-4f);
    assert(fabs(out->data[3] - 0.951747) < 1e-4f);
    assert(fabs(out->data[4] - 0.047385) < 1e-4f);
    assert(fabs(out->data[5] - 0.000868) < 1e-4f);

    float loss = cross_entropy_loss(out, w);
    
    assert(fabs(loss - 1.699233) < 1e-4f);

    free_tensor(x);
    free_tensor(w);
    free_tensor(b);
    free_tensor(out);

    printf("\nTest: Softmax Regression Passed!\n");
}

int main(){
    linear_reg_test();
    logistic_regression_test();
    softmax_regression_test();

    return 0;
}
