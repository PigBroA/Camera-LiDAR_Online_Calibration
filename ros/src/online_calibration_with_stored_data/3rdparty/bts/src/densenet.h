#pragma once

#include <iostream>
#include <torch/torch.h>
#include <torch/nn/functional.h>
#include <torch/utils.h>

#include "util.h"


class _DenseLayerImpl : public torch::nn::Module {
public:
    _DenseLayerImpl(int64_t num_input_features, int64_t growth_rate, int64_t bn_size, float drop_rate, bool memory_efficient = false);
    torch::Tensor bn_function(std::vector<torch::Tensor> inputs);
    bool any_requires_grad(std::vector<torch::Tensor> input);
    torch::Tensor forward(torch::Tensor input);
    torch::Tensor forward(std::vector<torch::Tensor> input);
private:
    torch::nn::BatchNorm2d norm1;
    torch::nn::ReLU relu1;
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d norm2;
    torch::nn::ReLU relu2;
    torch::nn::Conv2d conv2;
    float drop_rate;
    bool memory_efficient;
};
TORCH_MODULE(_DenseLayer);


class _DenseBlockImpl : public torch::nn::Module {
public:
    _DenseBlockImpl(int64_t num_layers, int64_t num_input_features, int64_t bn_size, int64_t growth_rate, float drop_rate, bool memory_efficient = false);
    torch::Tensor forward(torch::Tensor init_features);
private:
};
TORCH_MODULE(_DenseBlock);

StackSequential _Transition(int64_t num_input_features, int64_t num_output_features);

class DenseNetImpl : public torch::nn::Module {
public:
    DenseNetImpl(int64_t growth_rate = 32, std::vector<int64_t> block_config = {6, 12, 24, 16},
                 int64_t num_init_features = 64, int64_t bn_size = 4, float drop_rate = 0.0, int64_t num_classes = 1000, bool memory_efficient = false);
    torch::Tensor forward(torch::Tensor x);
    // this value is will be access at outside that's why in public
    // public variable don't have to be initialized in constructor's :, but I still remain it because I'm not good at memorization
    torch::nn::Sequential features;
private:
    torch::nn::Linear classifier;
};
TORCH_MODULE(DenseNet);