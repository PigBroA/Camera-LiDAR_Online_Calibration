#include "bts.h"


AtrousConvImpl::AtrousConvImpl(int64_t in_channels, int64_t out_channels, int64_t dilation, bool apply_bn_first)
    : atrous_conv() {
    if(apply_bn_first) {
        this->atrous_conv->push_back("first_bn", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(in_channels).momentum(0.01).affine(true).track_running_stats(true).eps(1.1e-5)));
    }
    this->atrous_conv->push_back("aconv_sequence", StackSequential(torch::nn::ReLU(),
                                                                       torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels * 2, 1).bias(false).stride(1).padding(0)),
                                                                       torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels * 2).momentum(0.01).affine(true).track_running_stats(true)),
                                                                       torch::nn::ReLU(),
                                                                       torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels * 2, out_channels, 3).bias(false).stride(1).padding({dilation, dilation}).dilation(dilation))));
    this->register_module("atrous_conv", this->atrous_conv);
}

torch::Tensor AtrousConvImpl::forward(torch::Tensor x) {
    return this->atrous_conv->forward(x);
}

UpConvImpl::UpConvImpl(int64_t in_channels, int64_t out_channels, double ratio)
    : elu(),
      conv(torch::nn::Conv2dOptions(in_channels, out_channels, 3).bias(false).stride(1).padding(1)) {
    this->ratio = ratio;
    this->register_module("conv", this->conv);
}

torch::Tensor UpConvImpl::forward(torch::Tensor x) {
    torch::Tensor up_x = torch::nn::functional::interpolate(x, torch::nn::functional::InterpolateFuncOptions().scale_factor(std::vector<double>({this->ratio, this->ratio})).mode(torch::kNearest).recompute_scale_factor(true));
    torch::Tensor out = this->conv(up_x);
    out = this->elu(out);
    return out;
}

Reduction1x1Impl::Reduction1x1Impl(int64_t num_in_filters, int64_t num_out_filters, float max_depth, bool is_final)
    : sigmoid(),
      reduc() {
    this->max_depth = max_depth;
    this->is_final = is_final;
    while(num_out_filters >= 4) {
        if(num_out_filters < 8) {
            if(this->is_final) {
                this->reduc->push_back("final", StackSequential(torch::nn::Conv2d(torch::nn::Conv2dOptions(num_in_filters, 1, 1).bias(false).stride(1).padding(0)),
                                                                torch::nn::Sigmoid()));
            }
            else {
                this->reduc->push_back("plane_params", torch::nn::Conv2d(torch::nn::Conv2dOptions(num_in_filters, 3, 1).bias(false).stride(1).padding(0)));
            }
            break;
        }
        else {
            this->reduc->push_back("inter_" + std::to_string(num_in_filters) + "_" + std::to_string(num_out_filters),
                                   StackSequential(torch::nn::Conv2d(torch::nn::Conv2dOptions(num_in_filters, num_out_filters, 1).bias(false).stride(1).padding(0)),
                                                   torch::nn::ELU()));
        }
        num_in_filters = num_out_filters;
        num_out_filters = (int)(num_out_filters/2);
    }
    this->register_module("reduc", this->reduc);
}

torch::Tensor Reduction1x1Impl::forward(torch::Tensor net) {
    net = this->reduc->forward(net);
    if(!this->is_final) {
        torch::Tensor theta = this->sigmoid(net.index({torch::indexing::Slice(), 0, torch::indexing::Slice(), torch::indexing::Slice()}))*3.14159265358979323846/3;
        torch::Tensor phi = this->sigmoid(net.index({torch::indexing::Slice(), 1, torch::indexing::Slice(), torch::indexing::Slice()}))*3.14159265358979323846*2;
        torch::Tensor dist = this->sigmoid(net.index({torch::indexing::Slice(), 2, torch::indexing::Slice(), torch::indexing::Slice()}))*this->max_depth;
        torch::Tensor n1 = torch::mul(torch::sin(theta), torch::cos(phi)).unsqueeze(1);
        torch::Tensor n2 = torch::mul(torch::sin(theta), torch::sin(phi)).unsqueeze(1);
        torch::Tensor n3 = torch::cos(theta).unsqueeze(1);
        torch::Tensor n4 = dist.unsqueeze(1);
        net = torch::cat({n1, n2, n3, n4}, 1);
    }
    return net;
}

LocalPlanarGuidanceImpl::LocalPlanarGuidanceImpl(float upratio) {
    this->upratio = upratio;
    this->u = torch::arange((int)(this->upratio)).reshape({1, 1, (int)(this->upratio)}).to(torch::kFloat);
    this->v = torch::arange((int)(this->upratio)).reshape({1, (int)(this->upratio), 1}).to(torch::kFloat);
}

torch::Tensor LocalPlanarGuidanceImpl::forward(torch::Tensor plane_eq) {
    torch::Device device = plane_eq.device();
    torch::Tensor plane_eq_expanded = torch::repeat_interleave(plane_eq, (int)(this->upratio), 2);
    plane_eq_expanded = torch::repeat_interleave(plane_eq_expanded, (int)(this->upratio), 3);
    torch::Tensor n1 = plane_eq_expanded.index({torch::indexing::Slice(), 0, torch::indexing::Slice(), torch::indexing::Slice()});
    torch::Tensor n2 = plane_eq_expanded.index({torch::indexing::Slice(), 1, torch::indexing::Slice(), torch::indexing::Slice()});
    torch::Tensor n3 = plane_eq_expanded.index({torch::indexing::Slice(), 2, torch::indexing::Slice(), torch::indexing::Slice()});
    torch::Tensor n4 = plane_eq_expanded.index({torch::indexing::Slice(), 3, torch::indexing::Slice(), torch::indexing::Slice()});
    torch::Tensor u = this->u.repeat({plane_eq.size(0), plane_eq.size(2)*((int)(this->upratio)), plane_eq.size(3)}).to(device);
    u = (u - (this->upratio - 1)*0.5)/this->upratio;
    torch::Tensor v = this->v.repeat({plane_eq.size(0), plane_eq.size(2), plane_eq.size(3)*((int)(this->upratio))}).to(device);
    v = (v - (this->upratio - 1)*0.5)/this->upratio;
    return n4/(n1*u + n2*v + n3);
}

DecoderImpl::DecoderImpl(std::string dataset, float max_depth, std::vector<int64_t> feat_out_channels, int64_t num_features)
    : dataset(dataset),
      max_depth(max_depth),
      upconv5(feat_out_channels[4], num_features),
      bn5(torch::nn::BatchNorm2dOptions(num_features).momentum(0.01).affine(true).eps(1.1e-5)),
      conv5(torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions(num_features + feat_out_channels[3], num_features, 3).stride(1).padding(1).bias(false)),
                                  torch::nn::ELU())),
      upconv4(num_features, (int)(num_features/2)),
      bn4(torch::nn::BatchNorm2dOptions((int)(num_features/2)).momentum(0.01).affine(true).eps(1.1e-5)),
      conv4(torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions((int)(num_features/2) + feat_out_channels[2], (int)(num_features/2), 3).stride(1).padding(1).bias(false)),
                                  torch::nn::ELU())),
      bn4_2(torch::nn::BatchNorm2dOptions((int)(num_features/2)).momentum(0.01).affine(true).eps(1.1e-5)),
      daspp_3((int)(num_features/2), (int)(num_features/4), 3, false),
      daspp_6((int)(num_features/2) + (int)(num_features/4) + feat_out_channels[2], (int)(num_features/4), 6),
      daspp_12(num_features + feat_out_channels[2], (int)(num_features/4), 12),
      daspp_18(num_features + (int)(num_features/4) + feat_out_channels[2], (int)(num_features/4), 18),
      daspp_24(num_features + (int)(num_features/2) + feat_out_channels[2], (int)(num_features/4), 24),
      daspp_conv(torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions(num_features + (int)(num_features/2) + (int)(num_features/4), (int)(num_features/4), 3).stride(1).padding(1).bias(false)),
                                       torch::nn::ELU())),
      reduc8x8((int)(num_features/4), (int)(num_features/4), this->max_depth),
      lpg8x8(8),
      upconv3((int)(num_features/4), (int)(num_features/4)),
      bn3(torch::nn::BatchNorm2dOptions((int)(num_features/4)).momentum(0.01).affine(true).eps(1.1e-5)),
      conv3(torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions((int)(num_features/4) + feat_out_channels[1] + 1, (int)(num_features/4), 3).stride(1).padding(1).bias(false)),
                                  torch::nn::ELU())),
      reduc4x4((int)(num_features/4), (int)(num_features/8), this->max_depth),
      lpg4x4(4),
      upconv2((int)(num_features/4), (int)(num_features/8)),
      bn2(torch::nn::BatchNorm2dOptions((int)(num_features/8)).momentum(0.01).affine(true).eps(1.1e-5)),
      conv2(torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions((int)(num_features/8) + feat_out_channels[0] + 1, (int)(num_features/8), 3).stride(1).padding(1).bias(false)),
                                  torch::nn::ELU())),
      reduc2x2((int)(num_features/8), (int)(num_features/16), this->max_depth),
      lpg2x2(2),
      upconv1((int)(num_features/8), (int)(num_features/16)),
      reduc1x1((int)(num_features/16), (int)(num_features/32), this->max_depth, true),
      conv1(torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions((int)(num_features/16) + 4, (int)(num_features/16), 3).stride(1).padding(1).bias(false)),
                                  torch::nn::ELU())),
      get_depth(torch::nn::Sequential(torch::nn::Conv2d(torch::nn::Conv2dOptions((int)(num_features/16), 1, 3).stride(1).padding(1).bias(false)),
                                      torch::nn::Sigmoid())) {
    this->register_module("upconv5", this->upconv5);
    this->register_module("bn5", this->bn5);
    this->register_module("conv5", this->conv5);
    this->register_module("upconv4", this->upconv4);
    this->register_module("bn4", this->bn4);
    this->register_module("conv4", this->conv4);
    this->register_module("bn4_2", this->bn4_2);
    this->register_module("daspp_3", this->daspp_3);
    this->register_module("daspp_6", this->daspp_6);
    this->register_module("daspp_12", this->daspp_12);
    this->register_module("daspp_18", this->daspp_18);
    this->register_module("daspp_24", this->daspp_24);
    this->register_module("daspp_conv", this->daspp_conv);
    this->register_module("reduc8x8", this->reduc8x8);
    this->register_module("lpg8x8", this->lpg8x8);
    this->register_module("upconv3", this->upconv3);
    this->register_module("bn3", this->bn3);
    this->register_module("conv3", this->conv3);
    this->register_module("reduc4x4", this->reduc4x4);
    this->register_module("lpg4x4", this->lpg4x4);
    this->register_module("upconv2", this->upconv2);
    this->register_module("bn2", this->bn2);
    this->register_module("conv2", this->conv2);
    this->register_module("reduc2x2", this->reduc2x2);
    this->register_module("lpg2x2", this->lpg2x2);
    this->register_module("upconv1", this->upconv1);
    this->register_module("reduc1x1", this->reduc1x1);
    this->register_module("conv1", this->conv1);
    this->register_module("get_depth", this->get_depth);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> DecoderImpl::forward(std::vector<torch::Tensor> features, torch::Tensor focal) {
    torch::Tensor skip0 = features[0];
    torch::Tensor skip1 = features[1];
    torch::Tensor skip2 = features[2];
    torch::Tensor skip3 = features[3];
    torch::Tensor dense_features = torch::nn::ReLU()(features[4]);
    torch::Tensor upconv5 = this->upconv5(dense_features);
    upconv5 = this->bn5(upconv5);
    torch::Tensor concat5 = torch::cat({upconv5, skip3}, 1);
    torch::Tensor iconv5 = this->conv5->forward(concat5);

    torch::Tensor upconv4 = this->upconv4(iconv5);
    upconv4 = this->bn4(upconv4);
    torch::Tensor concat4 = torch::cat({upconv4, skip2}, 1);
    torch::Tensor iconv4 = this->conv4->forward(concat4);
    iconv4 = this->bn4_2(iconv4);

    torch::Tensor daspp_3 = this->daspp_3(iconv4);
    torch::Tensor concat4_2 = torch::cat({concat4, daspp_3}, 1);
    torch::Tensor daspp_6 = this->daspp_6(concat4_2);
    torch::Tensor concat4_3 = torch::cat({concat4_2, daspp_6}, 1);
    torch::Tensor daspp_12 = this->daspp_12(concat4_3);
    torch::Tensor concat4_4 = torch::cat({concat4_3, daspp_12}, 1);
    torch::Tensor daspp_18 = this->daspp_18(concat4_4);
    torch::Tensor concat4_5 = torch::cat({concat4_4, daspp_18}, 1);
    torch::Tensor daspp_24 = this->daspp_24(concat4_5);
    torch::Tensor concat4_daspp = torch::cat({iconv4, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24}, 1);
    torch::Tensor daspp_feat = this->daspp_conv->forward(concat4_daspp);

    torch::Tensor reduc8x8 = this->reduc8x8(daspp_feat);
    torch::Tensor plane_normal_8x8 = reduc8x8.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(), torch::indexing::Slice()});
    plane_normal_8x8 = torch::nn::functional::normalize(plane_normal_8x8, torch::nn::functional::NormalizeFuncOptions().p(2.0).dim(1));
    torch::Tensor plane_dist_8x8 = reduc8x8.index({torch::indexing::Slice(), 3, torch::indexing::Slice(), torch::indexing::Slice()});
    torch::Tensor plane_eq_8x8 = torch::cat({plane_normal_8x8, plane_dist_8x8.unsqueeze(1)}, 1);
    torch::Tensor depth_8x8 = this->lpg8x8(plane_eq_8x8);
    torch::Tensor depth_8x8_scaled = depth_8x8.unsqueeze(1)/this->max_depth;
    torch::Tensor depth_8x8_scaled_ds = torch::nn::functional::interpolate(depth_8x8_scaled, torch::nn::functional::InterpolateFuncOptions().scale_factor(std::vector<double>({0.25, 0.25})).mode(torch::kNearest).recompute_scale_factor(true));
    
    torch::Tensor upconv3 = this->upconv3(daspp_feat);
    upconv3 = this->bn3(upconv3);
    torch::Tensor concat3 = torch::cat({upconv3, skip1, depth_8x8_scaled_ds}, 1);
    torch::Tensor iconv3 = this->conv3->forward(concat3);

    torch::Tensor reduc4x4 = this->reduc4x4(iconv3);
    torch::Tensor plane_normal_4x4 = reduc4x4.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(), torch::indexing::Slice()});
    plane_normal_4x4 = torch::nn::functional::normalize(plane_normal_4x4, torch::nn::functional::NormalizeFuncOptions().p(2.0).dim(1));
    torch::Tensor plane_dist_4x4 = reduc4x4.index({torch::indexing::Slice(), 3, torch::indexing::Slice(), torch::indexing::Slice()});
    torch::Tensor plane_eq_4x4 = torch::cat({plane_normal_4x4, plane_dist_4x4.unsqueeze(1)}, 1);
    torch::Tensor depth_4x4 = this->lpg4x4(plane_eq_4x4);
    torch::Tensor depth_4x4_scaled = depth_4x4.unsqueeze(1)/this->max_depth;
    torch::Tensor depth_4x4_scaled_ds = torch::nn::functional::interpolate(depth_4x4_scaled, torch::nn::functional::InterpolateFuncOptions().scale_factor(std::vector<double>({0.5, 0.5})).mode(torch::kNearest).recompute_scale_factor(true));
    
    torch::Tensor upconv2 = this->upconv2(iconv3);
    upconv2 = this->bn2(upconv2);
    torch::Tensor concat2 = torch::cat({upconv2, skip0, depth_4x4_scaled_ds}, 1);
    torch::Tensor iconv2 = this->conv2->forward(concat2);
    
    torch::Tensor reduc2x2 = this->reduc2x2(iconv2);
    torch::Tensor plane_normal_2x2 = reduc2x2.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), torch::indexing::Slice(), torch::indexing::Slice()});
    plane_normal_2x2 = torch::nn::functional::normalize(plane_normal_2x2, torch::nn::functional::NormalizeFuncOptions().p(2.0).dim(1));
    torch::Tensor plane_dist_2x2 = reduc2x2.index({torch::indexing::Slice(), 3, torch::indexing::Slice(), torch::indexing::Slice()});
    torch::Tensor plane_eq_2x2 = torch::cat({plane_normal_2x2, plane_dist_2x2.unsqueeze(1)}, 1);
    torch::Tensor depth_2x2 = this->lpg2x2(plane_eq_2x2);
    torch::Tensor depth_2x2_scaled = depth_2x2.unsqueeze(1)/this->max_depth;

    torch::Tensor upconv1 = this->upconv1(iconv2);
    torch::Tensor reduc1x1 = this->reduc1x1(upconv1);
    torch::Tensor concat1 = torch::cat({upconv1, reduc1x1, depth_2x2_scaled, depth_4x4_scaled, depth_8x8_scaled}, 1);
    torch::Tensor iconv1 = this->conv1->forward(concat1);
    torch::Tensor final_depth = this->max_depth*this->get_depth->forward(iconv1);
    if(this->dataset == "kitti") { //FIXME
        final_depth = final_depth*focal.view({-1, 1, 1, 1}).to(torch::kFloat)/715.0873;
    }
    return std::make_tuple(depth_8x8_scaled, depth_4x4_scaled, depth_2x2_scaled, reduc1x1, final_depth);
}

EncoderImpl::EncoderImpl(int64_t growth_rate, std::vector<int64_t> block_config,
                         int64_t num_init_features, int64_t bn_size, float drop_rate, int64_t num_classes, bool memory_efficient,
                         std::vector<std::string> feat_names)
    : base_model() {
    DenseNet backbone = DenseNet(growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes, memory_efficient);
    this->base_model = backbone->features;
    this->feat_names = feat_names;
    this->register_module("base_model", base_model);
}

std::vector<torch::Tensor> EncoderImpl::forward(torch::Tensor x) {
    torch::Tensor feature = x;
    std::vector<torch::Tensor> skip_feat;
    int i = 1;
    for(const auto& pair : this->base_model->named_children()) {
        std::string k = pair.key();
        auto v = pair.value();
        if(k.find("fc") != std::string::npos || k.find("avgpool") != std::string::npos) {
            continue;
        }
        // why can't get moduletype.., then I dont have to use if,else | need to refer to torch::nn::AnyModule, FIXME
        if(v->as<torch::nn::Conv2d>()) {
            feature = v->as<torch::nn::Conv2d>()->forward(feature);
        }
        else if(v->as<torch::nn::BatchNorm2d>()) {
            feature = v->as<torch::nn::BatchNorm2d>()->forward(feature);
        }
        else if(v->as<torch::nn::ReLU>()) {
            feature = v->as<torch::nn::ReLU>()->forward(feature);
        }
        else if(v->as<torch::nn::MaxPool2d>()) {
            feature = v->as<torch::nn::MaxPool2d>()->forward(feature);
        }
        else if(v->as<_DenseBlock>()) {
            feature = v->as<_DenseBlock>()->forward(feature);
        }
        else if(v->as<StackSequential>()) {
            feature = v->as<StackSequential>()->forward(feature);
        }
        //original use "x" here(used variable name repeatedly), so I name it "fuck" | "x" -> "fuck"
        for(const auto& fuck : this->feat_names) {
            if(k.find(fuck) != std::string::npos) {
                skip_feat.push_back(feature);
            }
        }
        i++;
    }
    return skip_feat;
}

BTSImpl::BTSImpl(std::string dataset, float max_depth,
                 std::vector<int64_t> feat_out_channels, int64_t num_features,
                 int64_t growth_rate, std::vector<int64_t> block_config,
                 int64_t num_init_features, int64_t bn_size, float drop_rate, int64_t num_classes, bool memory_efficient,
                 std::vector<std::string> feat_names)
    : encoder(growth_rate, block_config, num_init_features, bn_size, drop_rate, num_classes, memory_efficient, feat_names),
      decoder(dataset, max_depth, feat_out_channels, num_features) {
    this->register_module("encoder", encoder);
    this->register_module("decoder", decoder);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> BTSImpl::forward(torch::Tensor x, torch::Tensor focal) {
    std::vector<torch::Tensor> skip_feat = this->encoder(x);
    return this->decoder(skip_feat, focal);
}

SilogLossImpl::SilogLossImpl(float variance_focus) {
    this->variance_focus = variance_focus;
}

torch::Tensor SilogLossImpl::forward(torch::Tensor depth_est, torch::Tensor depth_gt, torch::Tensor mask) {
    torch::Tensor d = torch::log(depth_est.masked_select(mask) + 1e-7) - torch::log(depth_gt.masked_select(mask) + 1e-7);
    return torch::sqrt(torch::pow(d, 2).mean() - this->variance_focus*(torch::pow(d.mean(), 2)))*10.0;
}

void weights_init_xavier(torch::nn::Module& module) {
    if(auto* conv = dynamic_cast<torch::nn::Conv2dImpl*>(&module)) {
        torch::nn::init::xavier_uniform_(conv->weight);
        if(conv->bias.defined()) {
            torch::nn::init::zeros_(conv->bias);
        }
    }
}