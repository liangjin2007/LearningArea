```
4. torch & cub & thrust
4.1 torch
Function torch::from_blob(void *, at::IntArrayRef, at::IntArrayRef, const Deleter&, const at::TensorOptions&)

torch::TensorOptions:
  torch::kUInt8
  points.options().dtype(torch::kFloat32);


image 如何定义tensor
auto tensor = torch::from_blob((void*)data,
  {h, w, channels}, // img size
  {w * channels, channels, 1}, // stride
  torch::kUInt8);

数据类型转化
tensor.to(torch::kFloat32).permute({2, 0, 1}).clone() / 255.f;
tensor.unsqueeze(-1)
tensor.repeat({1, 3})
tensor.set_requires_grad(true)
tensor.size(0);
tensor.device()
torch::Tensor tensor = torch::ones({2, 2});
// 在第一个维度上重复2次，在第二个维度上重复3次
torch::Tensor repeated = tensor.repeat({2, 3});

tensor.index_put_(start, )
    torch::Tensor A = torch::zeros({3, 3});
    torch::Tensor B = torch::ones({2, 2});
    // 将 B 的内容赋值到 A 的左上角
    A.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}, B);
    // A 现在看起来像：
    // 1 1 0
    // 1 1 0
    // 0 0 0
    return 0;
tensor.index
   using namespace torch::index;
   torch::Tensor first_row = tensor.index({0}); // 第一行
   torch::Tensor first_column = tensor.index({Slice(), 0}); // 第一列
   torch::Tensor diagonal = tensor.index({Slice(), Slice()}); //?
   torch::Tensor origins = poses.index({"...", Slice(None, 3), 3}); // poses为P x 4 x 4， 得到origins为P x 3 x 1
   torch::Tensor center = torch::mean(origins, 0); // 沿着第0个维度求平均， 得到center为3 x 1
   float f = 1.0f / torch::max(torch::abs(origins)).item<float>(); // torch::abs(origins)每个元素求绝对值， torch::max() 获取所有元素里最大值， item<float>

tensor.sum()
tensor.detach() // 拷贝一份，require_grads_是false， 也就是不会参与梯度计算。   
tensor.contiguous().data_ptr<float>() // 获取cuda device pointer

tensor.print()
// 计算最后一个维度的L2范数，并保持维度大小不变
torch::Tensor l2_norm = tensor.norm(2, {torch::Dim(-1)}, true);

tensor.ndimension()



{
  torch::Tensor tensor = torch::full(...); // 不用指针。
  return tensor;
}




// optimization framework
{
  // define tensors and setup require_grad_()
  // define Optimizer e.g. torch::optim::Adam({tensor}, torch::optim::AdamOptions(lr));
  
  optimizer->zero_grad();

  forward()
  
  
  optimizer->step();
}



torch::full({P}, false, torch::kBool);
torch::log(x)
torch::zeros()
torch::ones()
torch::ones_like()
torch::max
torch:abs
torch::mean(a, 0/*axis*/)
torch::sigmoid
torch::exp
torch::logit
torch::matmul
torch::diag(a)
torch::diag(torch::tensor({ 1.0f, -1.0f, -1.0f }, R.device())); // 生成一个3x3的对角矩阵, 对角线元素为1, -1, --1
torch::nn::functional::normalize
torch::cuda::synchronize();
torch::Device device = torch::kCUDA;
auto scale_modifier = torch::tensor(raster_settings.scale_modifier, device);

// The Function<> class is an abstract base class for all the functions that can be differentiated in PyTorch's autograd system. When you create a custom C++ operation that you want to be differentiable, you need to derive from this class and implement the forward and backward methods.
class SphericalHarmonics : public Function<SphericalHarmonics>{
public:
    static torch::Tensor forward(AutogradContext *ctx, 
            int degreesToUse, 
            torch::Tensor viewDirs, 
            torch::Tensor coeffs);
    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);
};

torch::autograd::Function<ProjectGaussians>::apply

torch::AutogradContext::saved_data["degreesToUse"] = degreesToUse;
torch::AutogradContext::save_for_backward // This method allows you to save tensors within the autograd context so that they can be accessed during the backward pass.



4.2 cub


4.3 thrust
```


