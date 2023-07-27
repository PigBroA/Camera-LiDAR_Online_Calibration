#include "densenet.h"


_DenseLayerImpl::_DenseLayerImpl(int64_t num_input_features, int64_t growth_rate, int64_t bn_size, float drop_rate, bool memory_efficient)
    : norm1(torch::nn::BatchNorm2dOptions(num_input_features)),
      relu1(torch::nn::ReLUOptions(true)),
      conv1(torch::nn::Conv2dOptions(num_input_features, bn_size*growth_rate, 1).stride(1).bias(false)),
      norm2(torch::nn::BatchNorm2dOptions(bn_size*growth_rate)),
      relu2(torch::nn::ReLUOptions(true)),
      conv2(torch::nn::Conv2dOptions(bn_size*growth_rate, growth_rate, 1).stride(1).bias(false)) {
    this->register_module("norm1", norm1);
    this->register_module("relu1", relu1);
    this->register_module("conv1", conv1);
    this->register_module("norm2", norm2);
    this->register_module("relu2", relu2);
    this->register_module("conv2", conv2);
    this->drop_rate = drop_rate;
    this->memory_efficient = memory_efficient;
}

torch::Tensor _DenseLayerImpl::bn_function(std::vector<torch::Tensor> inputs) { 
    torch::Tensor concated_features = torch::cat(inputs, 1);
    torch::Tensor bottleneck_output = this->conv1(this->relu1(this->norm1(concated_features)));
    return bottleneck_output;
}

bool _DenseLayerImpl::any_requires_grad(std::vector<torch::Tensor> input) {
    for(torch::Tensor tensor : input) {
        if(tensor.requires_grad()) {
            return true;
        }
    }
    return false;
}

torch::Tensor _DenseLayerImpl::forward(torch::Tensor input) {
    std::vector<torch::Tensor> prev_features = {input};
    torch::Tensor bottleneck_output;
    if(this->memory_efficient && this->any_requires_grad(prev_features)) {
        //FIXME, Can't porting, won't run without memory_efficient=true
        //Here is Python Code -> Can't implement call_checkpoint_bottleneck function with c++
        //------------------------------------------------------------------
        //if torch.jit.is_scripting():
        //    raise Exception("Memory Efficient not supported in JIT")
        //bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        //------------------------------------------------------------------
    }
    else {
        bottleneck_output = this->bn_function(prev_features);
    }
    torch::Tensor new_features = this->conv2(this->relu2(this->norm2(bottleneck_output)));
    if(this->drop_rate > 0.0) {
        new_features = torch::nn::functional::dropout(new_features, torch::nn::functional::DropoutFuncOptions().p(this->drop_rate));
    }
    return new_features;
}

torch::Tensor _DenseLayerImpl::forward(std::vector<torch::Tensor> input) {
    std::vector<torch::Tensor> prev_features = input;
    torch::Tensor bottleneck_output;
    if(this->memory_efficient && this->any_requires_grad(prev_features)) {
        //FIXME, Can't porting, won't run without memory_efficient=true
        //Here is Python Code -> Can't implement call_checkpoint_bottleneck function with c++
        //------------------------------------------------------------------
        //if torch.jit.is_scripting():
        //    raise Exception("Memory Efficient not supported in JIT")
        //bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        //------------------------------------------------------------------
    }
    else {
        bottleneck_output = this->bn_function(prev_features);
    }
    torch::Tensor new_features = this->conv2(this->relu2(this->norm2(bottleneck_output)));
    if(this->drop_rate > 0.0) {
        new_features = torch::nn::functional::dropout(new_features, torch::nn::functional::DropoutFuncOptions().p(this->drop_rate));
    }
    return new_features;
}

_DenseBlockImpl::_DenseBlockImpl(int64_t num_layers, int64_t num_input_features, int64_t bn_size, int64_t growth_rate, float drop_rate, bool memory_efficient) {
    for(int i = 0; i < num_layers; i++) {
        _DenseLayer layer = _DenseLayer(num_input_features + i*growth_rate, growth_rate, bn_size, drop_rate, memory_efficient);
        this->register_module("denselayer" + std::to_string(i + 1), layer);
    }
}

torch::Tensor _DenseBlockImpl::forward(torch::Tensor init_features) {
    std::vector<torch::Tensor> features = {init_features};
    for(auto& layer : this->children()) {
        torch::Tensor new_features = layer->as<_DenseLayer>()->forward(features);
        features.push_back(new_features);
    }
    return torch::cat(features, 1);
}

StackSequential _Transition(int64_t num_input_features, int64_t num_output_features) {
    StackSequential sequence_modules;
    sequence_modules->push_back("norm", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_input_features)));
    sequence_modules->push_back("relu", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
    sequence_modules->push_back("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(num_input_features, num_output_features, 1).stride(1).bias(false)));
    sequence_modules->push_back("pool", torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({2, 2}).stride({2, 2})));
    return sequence_modules;
}

DenseNetImpl::DenseNetImpl(int64_t growth_rate, std::vector<int64_t> block_config,
                           int64_t num_init_features, int64_t bn_size, float drop_rate, int64_t num_classes, bool memory_efficient)
    : features(),
      classifier(nullptr) {
    this->features->push_back("conv0", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, num_init_features, 7).stride(2).padding(3).bias(false)));
    this->features->push_back("norm0", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_init_features)));
    this->features->push_back("relu0", torch::nn::ReLU(torch::nn::ReLUOptions(true)));
    this->features->push_back("pool0", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({3, 3}).stride({2, 2}).padding({1, 1})));
    int num_features = num_init_features;
    for(int i = 0; i < block_config.size(); i++) {
        int num_layers = block_config[i];
        _DenseBlock block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate, memory_efficient);
        this->features->push_back("denseblock" + std::to_string(i + 1), block);
        num_features = num_features + num_layers*growth_rate;
        if(i != block_config.size() - 1) {
            StackSequential trans = _Transition(num_features, (int)(num_features/2));
            this->features->push_back("transition" + std::to_string(i + 1), trans);
            num_features = (int)(num_features/2);
        }
    }
    this->features->push_back("norm5", torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(num_features)));
    this->classifier = torch::nn::Linear(num_features, num_classes);
    this->register_module("features", this->features);
    this->register_module("classifier", this->classifier);
    for(auto& m : this->modules(false)) {
        if(m->as<torch::nn::Conv2d>()) {
            torch::nn::init::kaiming_normal_(m->as<torch::nn::Conv2d>()->weight);
        }
        else if(m->as<torch::nn::BatchNorm2d>()) {
            torch::nn::init::constant_(m->as<torch::nn::BatchNorm2d>()->weight, 1);
            torch::nn::init::constant_(m->as<torch::nn::BatchNorm2d>()->bias, 0);
        }
        else if(m->as<torch::nn::Linear>()) {
            torch::nn::init::constant_(m->as<torch::nn::Linear>()->bias, 0);
        }
    }
}

torch::Tensor DenseNetImpl::forward(torch::Tensor x) {
    torch::Tensor features = this->features->forward(x);
    torch::Tensor out = torch::nn::functional::relu(features, torch::nn::ReLUOptions(true));
    out = torch::nn::functional::adaptive_avg_pool2d(out, torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
    out = torch::flatten(out, 1);
    out = this->classifier(out);
    return out;
}