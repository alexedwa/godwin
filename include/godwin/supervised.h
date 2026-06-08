#ifndef SUPERVISED
#define SUPERVISED

void linear_forward(Tensor* x, Tensor* w, Tensor* b, Tensor* out);
void logistic_regression(Tensor* x, Tensor* w, Tensor* b, Tensor* out);
void softmax_regression(Tensor* x, Tensor* w, Tensor* b, Tensor* out);

void sigmoid(Tensor* x, Tensor* out);
void softmax(Tensor* x, Tensor* out);

#endif

