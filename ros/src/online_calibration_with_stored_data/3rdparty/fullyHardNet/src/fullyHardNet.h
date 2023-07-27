#pragma once

#include <iostream>
#include <torch/torch.h>

// class ConvLayerImpl : public torch::nn::Module {
// public:
//     ConvLayerImpl(int64_t in_channels, int64_t out_channels, int64_t kernel = 3, int64_t stride = 1, float dropout = 0.1);
//     torch::Tensor forward(torch::Tensor x);
// private:
//     torch::nn::Sequential modules;
// };
// TORCH_MODULE(ConvLayer);

torch::nn::Sequential ConvLayer(int64_t in_channels, int64_t out_channels, int64_t kernel = 3, int64_t stride = 1, float dropout = 0.1);

class HarDBlockImpl : public torch::nn::Module {
public:
    std::tuple<int, int, std::vector<int>> get_link(int64_t layer, int64_t base_ch, int64_t growth_rate, float grmul);
    int get_out_ch();
    HarDBlockImpl(int64_t in_channels, int64_t growth_rate, float grmul, int64_t n_layers, bool keepBase = false, bool residual_out = false);
    torch::Tensor forward(torch::Tensor x);
private:
    int in_channels;
    int growth_rate;
    float grmul;
    int n_layers;
    bool keepBase;
    std::vector<std::vector<int>> links;
    int out_channels;
    torch::nn::ModuleList layers;
};
TORCH_MODULE(HarDBlock);

class TransitionUpImpl : public torch::nn::Module {
public:
    TransitionUpImpl(int64_t in_channels, int64_t out_channel);
    torch::Tensor forward(torch::Tensor x, torch::Tensor skip, bool concat = true);
};
TORCH_MODULE(TransitionUp);

class HarDNetImpl : public torch::nn::Module {
public:
    HarDNetImpl(int64_t n_classes = 19);
    torch::Tensor forward(torch::Tensor x);
private:
    torch::nn::ModuleList base;
    std::vector<int> shortcut_layers;
    int n_blocks;
    torch::nn::ModuleList transUpBlocks;
    torch::nn::ModuleList denseBlocksUp;
    torch::nn::ModuleList conv1x1_up;
    torch::nn::Conv2d finalConv;
};
TORCH_MODULE(HarDNet);