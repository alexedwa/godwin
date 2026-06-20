#include <godwin/godwin.h>

float cross_entropy_loss(Tensor* prob, Tensor* label){
    int row = prob->shape[0];
    int col = label->shape[1];
    float loss = 0.0f;

    for(int i = 0; i < row; ++i){
        for(int j = 0; j < col; ++j){
            int prob_idx = i * prob->stride[0] + j * prob->stride[1];
            int label_idx = i *  label->stride[0] + j * label->stride[1];

            if(label->data[label_idx] == 1.0f){
                loss -= logf(prob->data[prob_idx] + 1e-7f);
            }
        }
    }

    return loss / row;
}
