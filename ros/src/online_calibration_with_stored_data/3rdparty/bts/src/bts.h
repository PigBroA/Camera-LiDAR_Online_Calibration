#pragma once

#include <iostream>
#include <torch/torch.h>
#include <torch/nn/functional.h>
#include <torch/utils.h>

#include "util.h"
#include "densenet.h"


class AtrousConvImpl : public torch::nn::Module {
public:
    AtrousConvImpl(int64_t in_channels, int64_t out_channels, int64_t dilation, bool apply_bn_first = true);
    torch::Tensor forward(torch::Tensor x);
private:
    torch::nn::Sequential atrous_conv;
};
TORCH_MODULE(AtrousConv);

class UpConvImpl : public torch::nn::Module {
public:
    UpConvImpl(int64_t in_channels, int64_t out_channels, double ratio=2.0);
    torch::Tensor forward(torch::Tensor x);
private:
    torch::nn::ELU elu;
    torch::nn::Conv2d conv;
    double ratio;
};
TORCH_MODULE(UpConv);

class Reduction1x1Impl : public torch::nn::Module {
public:
    Reduction1x1Impl(int64_t num_in_filters, int64_t num_out_filters, float max_depth, bool is_final = false);
    torch::Tensor forward(torch::Tensor net);
private:
    float max_depth;
    bool is_final;
    torch::nn::Sigmoid sigmoid;
    torch::nn::Sequential reduc;
};
TORCH_MODULE(Reduction1x1);

class LocalPlanarGuidanceImpl : public torch::nn::Module {
public:
    LocalPlanarGuidanceImpl(float upratio);
    torch::Tensor forward(torch::Tensor plane_eq);
private:
    float upratio;
    torch::Tensor u;
    torch::Tensor v;
};
TORCH_MODULE(LocalPlanarGuidance);

class DecoderImpl : public torch::nn::Module {
public:
    DecoderImpl(std::string dataset, float max_depth = 80.0,
                std::vector<int64_t> feat_out_channels = {96, 96, 192, 384, 2208}, int64_t num_features = 512); // Densenet161 based initialization
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(std::vector<torch::Tensor> features, torch::Tensor focal);
private:
    std::string dataset;
    float max_depth;
    UpConv upconv5;
    torch::nn::BatchNorm2d bn5;
    torch::nn::Sequential conv5;
    UpConv upconv4;
    torch::nn::BatchNorm2d bn4;
    torch::nn::Sequential conv4;
    torch::nn::BatchNorm2d bn4_2;
    AtrousConv daspp_3;
    AtrousConv daspp_6;
    AtrousConv daspp_12;
    AtrousConv daspp_18;
    AtrousConv daspp_24;
    torch::nn::Sequential daspp_conv;
    Reduction1x1 reduc8x8;
    LocalPlanarGuidance lpg8x8;
    UpConv upconv3;
    torch::nn::BatchNorm2d bn3;
    torch::nn::Sequential conv3;
    Reduction1x1 reduc4x4;
    LocalPlanarGuidance lpg4x4;
    UpConv upconv2;
    torch::nn::BatchNorm2d bn2;
    torch::nn::Sequential conv2;
    Reduction1x1 reduc2x2;
    LocalPlanarGuidance lpg2x2;
    UpConv upconv1;
    Reduction1x1 reduc1x1;
    torch::nn::Sequential conv1;
    torch::nn::Sequential get_depth;
    
};
TORCH_MODULE(Decoder);

class EncoderImpl : public torch::nn::Module {
public:
    EncoderImpl(int64_t growth_rate = 48, std::vector<int64_t> block_config = {6, 12, 36, 24},
                int64_t num_init_features = 96, int64_t bn_size = 4, float drop_rate = 0.0, int64_t num_classes = 1000, bool memory_efficient = false,
                std::vector<std::string> feat_names = {"relu0", "pool0", "transition1", "transition2", "norm5"}); // Densenet161 based initialization
    std::vector<torch::Tensor> forward(torch::Tensor x);
private:
    torch::nn::Sequential base_model;
    std::vector<std::string> feat_names;
};
TORCH_MODULE(Encoder);

class BTSImpl : public torch::nn::Module {
public:
    BTSImpl(std::string dataset = "", float max_depth = 80.0,
            std::vector<int64_t> feat_out_channels = {96, 96, 192, 384, 2208}, int64_t num_features = 512,
            int64_t growth_rate = 48, std::vector<int64_t> block_config = {6, 12, 36, 24},
            int64_t num_init_features = 96, int64_t bn_size = 4, float drop_rate = 0.0, int64_t num_classes = 1000, bool memory_efficient = false,
            std::vector<std::string> feat_names = {"relu0", "pool0", "transition1", "transition2", "norm5"});
    //focal is just need for KITTI. don't need for another db, if the db has just one focal length(just push is the value, but it doesn't be used).
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor skip_feat, torch::Tensor focal);
private:
    Encoder encoder;
    Decoder decoder;
};
TORCH_MODULE(BTS);

class SilogLossImpl : public torch::nn::Module {
public:
    SilogLossImpl(float variance_focus);
    torch::Tensor forward(torch::Tensor depth_est, torch::Tensor depth_gt, torch::Tensor mask);
private:
    float variance_focus;
};
TORCH_MODULE(SilogLoss);

void weights_init_xavier(torch::nn::Module& module);