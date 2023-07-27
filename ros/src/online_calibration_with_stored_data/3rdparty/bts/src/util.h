#pragma once

#include <iostream>
#include <torch/torch.h>
#include <torch/nn/functional.h>
#include <torch/utils.h>

//for using sequential insert to sequential, refer to https://github.com/pytorch/vision/blob/2f46070f3cb1ea894d82578f3dc5677f82f34958/torchvision/csrc/models/mnasnet.cpp#59
struct StackSequentialImpl : torch::nn::SequentialImpl {
  using SequentialImpl::SequentialImpl;

  torch::Tensor forward(torch::Tensor x) {
    return SequentialImpl::forward(x);
  }
};
TORCH_MODULE(StackSequential);