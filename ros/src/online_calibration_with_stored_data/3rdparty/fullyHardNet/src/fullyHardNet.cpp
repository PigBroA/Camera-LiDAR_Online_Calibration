#include "fullyHardNet.h"

// ConvLayerImpl::ConvLayerImpl(int64_t in_channels, int64_t out_channels, int64_t kernel, int64_t stride, float dropout) {
//     modules->push_back("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel)
//                                                      .stride(stride)
//                                                      .padding((int)kernel/2)
//                                                      .bias(false)));
//     modules->push_back("norm", torch::nn::BatchNorm2d(out_channels));
//     modules->push_back("relu", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
//     register_module("convLayer", modules);
// }

// torch::Tensor ConvLayerImpl::forward(torch::Tensor x) {
//     return modules->forward(x);
// }

torch::nn::Sequential ConvLayer(int64_t in_channels, int64_t out_channels, int64_t kernel, int64_t stride, float dropout) {
    torch::nn::Sequential module;
    module->push_back("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel)
                                                     .stride(stride)
                                                     .padding((int)(kernel/2))
                                                     .bias(false)
                                                     ));
    module->push_back("norm", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels)
                                                        //  .eps(0.5)
                                                        //  .momentum(0.1)
                                                        //  .affine(false)
                                                        //  .track_running_stats(true)
                                                         ));
    module->push_back("relu", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
    return module;
}

std::tuple<int, int, std::vector<int>> HarDBlockImpl::get_link(int64_t layer, int64_t base_ch, int64_t growth_rate, float grmul) {
    std::vector<int> link;
    if(layer == 0) {
        return std::make_tuple(base_ch, 1, link);
    }
    float out_channels_ = growth_rate;
    for(int i = 0; i < 10; i++) {
        int dv = std::pow(2, i);
        if(layer%dv == 0) {
            int k = layer - dv;
            link.push_back(k);
            if(i > 0) {
                out_channels_ *= grmul;
            }
        }
    }
    int out_channels__ = (int)((int)(out_channels_ + 1)/2)*2;
    int in_channels_ = 0;
    for(int i : link) {
        int ch = std::get<0>(this->get_link(i, base_ch, growth_rate, grmul));
        in_channels_ += ch;
    }
    return std::make_tuple(out_channels__, in_channels_, link);
}

int HarDBlockImpl::get_out_ch() {
    return this->out_channels;
}

HarDBlockImpl::HarDBlockImpl(int64_t in_channels, int64_t growth_rate, float grmul, int64_t n_layers, bool keepBase, bool residual_out) {
    this->in_channels = in_channels;
    this->growth_rate = growth_rate;
    this->grmul = grmul;
    this->n_layers = n_layers;
    this->keepBase = keepBase;
    std::vector<torch::nn::Sequential> layers_;
    this->out_channels = 0;
    for(int i = 0; i < n_layers; i++) {
        std::tuple<int, int, std::vector<int>> tmp = this->get_link(i + 1, in_channels, growth_rate, grmul);
        int outch = std::get<0>(tmp);
        int inch = std::get<1>(tmp);
        std::vector<int> link = std::get<2>(tmp);
        this->links.push_back(link);
        bool use_relu = residual_out;
        layers_.push_back(ConvLayer(inch, outch));
        if(i%2 == 0 || i == (n_layers - 1)) {
            this->out_channels += outch;
        }
    }
    for(int i = 0; i < layers_.size(); i++) {
        this->layers->push_back(layers_[i]);
    }

    register_module("layers", this->layers);
}

torch::Tensor HarDBlockImpl::forward(torch::Tensor x) {
    std::vector<torch::Tensor> layers_;
    layers_.push_back(x);
    for(int layer = 0; layer < this->layers->size(); layer++) {
        std::vector<int> link = this->links[layer];
        std::vector<torch::Tensor> tin;
        for(int i : link) {
            tin.push_back(layers_[i]);
        }
        if(tin.size() > 1) {
            x = torch::cat(tin, 1);
        }
        else {
            x = tin[0];
        }
        torch::Tensor out = this->layers[layer]->as<torch::nn::Sequential>()->forward(x);
        layers_.push_back(out);
    }
    int t = layers_.size();
    std::vector<torch::Tensor> out_;
    for(int i = 0; i < t; i++) {
        if((i == 0 && this->keepBase) || (i == t - 1) || (i%2 == 1)) {
            out_.push_back(layers_[i]);
        }
    }
    torch::Tensor out = torch::cat(out_, 1);
    return out;
}

TransitionUpImpl::TransitionUpImpl(int64_t in_channels, int64_t out_channel) {

}

torch::Tensor TransitionUpImpl::forward(torch::Tensor x, torch::Tensor skip, bool concat) {
    torch::Tensor out = torch::nn::functional::interpolate(x, torch::nn::functional::InterpolateFuncOptions()
                                                                  .size(std::vector<int64_t>({skip.size(2), skip.size(3)}))
                                                                  .mode(torch::kBilinear)
                                                                  .align_corners(true)
                                                                  );
    if(concat) {
        out = torch::cat(std::vector<torch::Tensor>({out, skip}), 1);
    }
    return out;
}

HarDNetImpl::HarDNetImpl(int64_t n_classes) : finalConv(nullptr) {
    std::vector<int> first_ch({16, 24, 32, 48});
    std::vector<int> ch_list({64, 96, 160, 224, 320});
    float grmul = 1.7;
    std::vector<int> gr({10, 16, 18, 24, 32});
    std::vector<int> n_layers({4, 4, 8, 8, 8});
    int blks = n_layers.size();

    this->base->push_back(ConvLayer(3, first_ch[0], 3, 2));
    this->base->push_back(ConvLayer(first_ch[0], first_ch[1], 3));
    this->base->push_back(ConvLayer(first_ch[1], first_ch[2], 3, 2));
    this->base->push_back(ConvLayer(first_ch[2], first_ch[3], 3));

    std::vector<int> skip_connection_channel_counts;
    int ch = first_ch[3];
    for(int i = 0; i < blks; i++) {
        HarDBlock blk(ch, gr[i], grmul, n_layers[i]);
        ch = blk->get_out_ch();
        skip_connection_channel_counts.push_back(ch);
        this->base->push_back(blk);
        if(i < blks -1) {
            this->shortcut_layers.push_back(this->base->size() - 1);
        }
        this->base->push_back(ConvLayer(ch, ch_list[i], 1));
        ch = ch_list[i];
        if(i < blks - 1) {
            this->base->push_back(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2).stride(2)));
        }
    }
    int cur_channels_count = ch;
    int prev_block_channels = ch;
    int n_blocks_ = blks - 1;
    this->n_blocks = n_blocks_;

    for(int i = n_blocks_ - 1; i > -1; i--) {
        this->transUpBlocks->push_back(TransitionUp(prev_block_channels, prev_block_channels));
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[i];
        this->conv1x1_up->push_back(ConvLayer(cur_channels_count, (int)(cur_channels_count/2), 1));
        cur_channels_count = (int)(cur_channels_count/2);
        HarDBlock blk(cur_channels_count, gr[i], grmul, n_layers[i]);
        this->denseBlocksUp->push_back(blk);
        prev_block_channels = blk->get_out_ch();
        cur_channels_count = prev_block_channels;
    }
    this->finalConv = torch::nn::Conv2d(torch::nn::Conv2dOptions(cur_channels_count, n_classes, 1).stride(1).padding(0).bias(true));

    register_module("base", this->base);
    register_module("transUpBlocks", this->transUpBlocks);
    register_module("denseBlocksUp", this->denseBlocksUp);
    register_module("conv1x1_up", this->conv1x1_up);
    register_module("finalConv", this->finalConv);
}

torch::Tensor HarDNetImpl::forward(torch::Tensor x) {
    std::vector<torch::Tensor> skip_connections;
    torch::Tensor out;
    auto size_in = x.sizes();
    for(int i = 0; i < this->base->size(); i++) {
        if(this->base[i]->as<torch::nn::Sequential>()) {
            x = this->base[i]->as<torch::nn::Sequential>()->forward(x);
        }
        else if(this->base[i]->as<HarDBlock>()) {
            x = this->base[i]->as<HarDBlock>()->forward(x);
        }
        else if(this->base[i]->as<torch::nn::AvgPool2d>()) {
            x = this->base[i]->as<torch::nn::AvgPool2d>()->forward(x);
        }
        for(int j = 0; j < shortcut_layers.size(); j++) {
            if(i == shortcut_layers[j]) {
                skip_connections.push_back(x);
            }
        }
    }
    out = x;
    for(int i = 0; i < this->n_blocks; i++) {
        torch::Tensor skip = skip_connections.back();
        skip_connections.pop_back();
        out = this->transUpBlocks[i]->as<TransitionUp>()->forward(out, skip, true);
        out = this->conv1x1_up[i]->as<torch::nn::Sequential>()->forward(out);
        out = this->denseBlocksUp[i]->as<HarDBlock>()->forward(out);
    }
    out = this->finalConv(out);
    out = torch::nn::functional::interpolate(out, torch::nn::functional::InterpolateFuncOptions()
                                                      .size(std::vector<int64_t>({size_in[2], size_in[3]}))
                                                      .mode(torch::kBilinear)
                                                      .align_corners(true));
    return out;
}