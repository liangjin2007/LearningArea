
# gsplat

目录
- [1.算法基本流程](#1算法基本流程)
- [2.Questions](#2Questions)
- [3.代码gaussian-splatting-cuda](#3代码gaussian-splatting-cuda)
- [4.代码OpenSplat](#4代码OpenSplat) 推荐读这个代码，结构清晰，比前者要快（虽然目前看起来它的输出我不太满意）
- [5.问题](#5问题)
     
## 1.算法基本流程

加载数据（COLMAP, nerfstudio, etc）

初始化GaussianModel

Define tensors and setup require_grad_()

Define n Optimizers e.g. torch::optim::Adam({tensor}, torch::optim::AdamOptions(lr));

for step in range(0, max_steps)
{
    optimizer->zero_grad(); // 所有optimizer
    
    pred_image = forward(); // 1. Project 3D Gaussian to get 2D Gaussian;   2. Convert each point's 16 harminic eofficients to rgb color;   3. rasterize 2D Gaussian
    
    loss = LossFunc(gt_image, pred_image);
    loss.backward();
  
    optimizer->step();
    // update learning rate
  
    delete_or_densification_splats(); // Remember to use torch::NoGradGuards
};
  
## 2. Questions
- Q:为什么每个Gaussian上学习的是球谐系数？
A: 说明建模的时候是认为一个Gaussian在从不同角度看过去的时候颜色是不同的。


  

## 3.代码gaussian-splatting-cuda
Seminar paper [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://github.com/graphdeco-inria/gaussian-splatting)
github代码没安装成功，转而切到c++版的一个实现https://github.com/MrNeRF/gaussian-splatting-cuda

数据集1 https://jonbarron.info/mipnerf360/
数据集2 https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
数据集3 https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip


```
官方只配过linux系统， windows系统需要趟个坑。
```

- 安装开发环境：
```
安装Visual Studio 2019 Community
安装NSight Compute-win64-2021.2.1.xxx
安装CUDA 11.8
安装NVIDIA_NSight_Visual_Studio_Edition
安装CMake
安装Python3.9
如果Visual Studio找不到CUDA（比如打不开依赖于CUDA的project）, 拷贝 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vxxx\extras\visual_studio_integration\MSBuildExtensions中的四个文件 到 C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations
```

- 配置工程
```
// Step 1.
在目标位置右键打开git Bash Here，敲下面命令
git clone --recursive https://github.com/MrNeRF/gaussian-splatting-cuda

// Step 2.
从pytorch官网去下载windows版libtorch 2.0.1， 解压放到external/libtorch
下载tbb库放到external/tbb
拷贝Python安装目录中的include, libs, 和python39.dll拷贝到external/pythonlib3.9

// Step 3.
修改CMakeLists.txt：
  添加变量：
  set(PYTHONLIB_INCLUDE_DIRS "${PROJ_ROOT_DIR}/external/pythonlib3.9/include")
  set(PYTHONLIB_LIBRARIES "${PROJ_ROOT_DIR}/external/pythonlib3.9/libs/python39.lib")
  comment #find_package(PythonLibs REQUIRED)
  添加：   
  target_include_directories(${PROJECT_NAME} xxx ${PYTHONLIB_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME}
        PRIVATE
        xxx 
        ${PYTHONLIB_LIBRARIES}

// Step 4. 修改一处编译不过的地方
    // Set up rasterization configuration
    GaussianRasterizationSettings raster_settings = {
        //.image_height = static_cast<int>(viewpoint_camera.Get_image_height()),
        //.image_width = static_cast<int>(viewpoint_camera.Get_image_width()),
        //.tanfovx = std::tan(viewpoint_camera.Get_FoVx() * 0.5f),
        //.tanfovy = std::tan(viewpoint_camera.Get_FoVy() * 0.5f),
        //.bg = bg_color,
        //.scale_modifier = scaling_modifier,
        //.viewmatrix = viewpoint_camera.Get_world_view_transform(),
        //.projmatrix = viewpoint_camera.Get_full_proj_transform(),
        //.sh_degree = gaussianModel.Get_active_sh_degree(),
        //.camera_center = viewpoint_camera.Get_camera_center(),
        //.prefiltered = false
        static_cast<int>(viewpoint_camera.Get_image_height()),
        static_cast<int>(viewpoint_camera.Get_image_width()),
        std::tan(viewpoint_camera.Get_FoVx() * 0.5f),
        std::tan(viewpoint_camera.Get_FoVy() * 0.5f),
        bg_color,
        scaling_modifier,
        viewpoint_camera.Get_world_view_transform(),
        viewpoint_camera.Get_full_proj_transform(),
        gaussianModel.Get_active_sh_degree(),
        viewpoint_camera.Get_camera_center(),
        false
    };

// Step 5. CMake GUI设置 Where is the source code 和 Where to build the binaries, 点Configure, 会报错，提示tbb找不到。设置好tbb，Configure成功。点Generate成功。打开Visual Studio 2019，
设置为RelWithDbgInfo + x64。

// Step 6. 其他编译问题
编译出错：找不到Python.h

  给external/simple-knn/CMakelists.txt添加
    target_include_directories(simple-knn 
        PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/simple-knn # we need this public to easily include the spatical.h header in our main program
        PRIVATE 
        ${TORCH_INCLUDE_DIRS}
        ${PYTHONLIB_INCLUDE_DIRS}      # 此次添加
    )
    
    target_link_libraries(simple-knn
        PUBLIC
        ${PYTHONLIB_LIBRARIES})        # 此次添加


由于我这边下的是libtorch2.4, 还碰到编译错误 c10::guts::to_string这个函数不存在，查了libtorch API, optimizer->states() 返回的是一个void * 到 xxxState的映射， 将c10::guts::to_string删除即可。
（libtorch 文档 https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuron/tutorials/tutorial-libtorch.html)

还有一个库找不到的问题 C:\Program Files\NVIDIA Corporation\NvToolsExt\lib\x64\nvToolsExt64_1.lib， 可以直接把这个库依赖删掉。


// Step 7. 读取optimization_params.json的问题
首先在parameters.cu/cuh给read_optim_params_from_json添加一个参数，用来直接传递json 文件路径。
        OptimizationParameters read_optim_params_from_json(std::string json_path) {
            // 删除掉这个地方
            // Check if the file exists before trying to open it
            if (!std::filesystem::exists(json_path)) {
                throw std::runtime_error("Error: " + json_path + " does not exist!");
            }
            ...
        }
main函數中修改：
    auto modelParams = gs::param::ModelParameters();
    auto optimParams = gs::param::OptimizationParameters();
    if (parse_cmd_line_args(args, modelParams, optimParams) < 0) {
        return -1;
    };

parse_cmd_line_args函數中添加：
    args::ValueFlag<std::string> parameter_path(parser, "parameter_path", "Path to the optimization parameter json", {'p', "parameter-path"});
    if (!parameter_path) {
        std::cerr << "No parameter path provided!" << std::endl;
        return -1;
    }
    optimParams = gs::param::read_optim_params_from_json(args::get(parameter_path));

编译成功。
```

- 原理
```
1. 3D Gaussians场景表示及光栅化
2. 优化 3D positions, opacity alpha, anisotropic covariance and spherical harmonic coefficients。 优化时通过添加和删除3D Gaussians来自适应地控制密度。问题： 结合batch sgd, 是否还能保证一定收敛？
3. realtime rendering solution fast GPU sorting algorithms and tile-based rasterization.
```


- 源代码解读
```
1.Dataset Parser   read_colmap_scene_info function:
  auto cameras = read_cameras_binary(file_path / "sparse/0/cameras.bin");
  auto images = read_images_binary(file_path / "sparse/0/images.bin");
  sceneInfos->_point_cloud = read_ply_file(file_path / "sparse/0/points3D.ply");
  sceneInfos->_ply_path = file_path / "sparse/0/points3D.ply";
  sceneInfos->_cameras = read_colmap_cameras(file_path / "images", cameras, images, resolution);

2. colmap数据：
输入数据是colmap数据，目录结构如下： 对train数据是301个图片
images
  00001.jpg
  ...
sparse
  0
    cameras.bin
    images.bin
    points3D.ply  ply格式点云
    points3D.bin  二进制点云

// 已知camera.R, camera.T， 获取camera在世界空间中的位置， 这部分代码比较奇怪。 
问题：为什么world To view matrix 是
  ( camera.R camera.T )
  (     0        1    ) ？
  说明colmap中存的数据是world to view matrix中的R和T。

gaussian_splatting_cuda代码中的camera_info->_R是cameras.bin中拿到的矩阵的transpose()

已知world to view matrix， 求逆得到 CTW矩阵 则 CTW*(0, 0, 0, 1) = CTW的第三列的数据是相机的位置。


3. Scene
  _gaussians          // GaussianModel对象，用torch实现。将ply格式转成torch对象
    Create_from_pcd(_scene_infos->_point_cloud, _scene_infos->_nerf_norm_radius)
    int _active_sh_degree = 0;
    int _max_sh_degree = 0;
    float _spatial_lr_scale = 0.f; // 初始化为_scene_infos->_nerf_norm_radius
    float _percent_dense = 0.f;

    Expon_lr_func _xyz_scheduler_args;
    torch::Tensor _denom;             // n x 1
    torch::Tensor _xyz;               // n x 3, _xyz = torch::from_blob(pcd._points.data(), {static_cast<long>(pcd._points.size()), 3}, pointType).to(torch::kCUDA).set_requires_grad(true);          
    torch::Tensor _features_dc;       // ? 
    torch::Tensor _features_rest;     // ?
    torch::Tensor _scaling;           // n x 3
    torch::Tensor _rotation;          // n x 4
    torch::Tensor _xyz_gradient_accum; // why only xyz has gradient accum
    torch::Tensor _opacity;           // n x 1

    

  _params             // 优化相关的参数 来自optimization_params.json和命令行
  _cameras            // Camera数组
    Camera
      camera_id, R, T, fovx, fovy, torch::Tensor image, image_name, uid, scale

  _scene_infos
    _point_cloud      // points3D.ply解析后放到了这里
    _ply_path
    _cameras          // CameraInfo数组, cameras.bin和images.bin数据解析后放到了这个而数组
    _nerf_norm_radius       // 包围所有相机的包围球的的半径 * 1.1
    _nerf_norm_translation  相机中心的中心 

训练：
  对每次迭代（30000）
    随即从n=301个相机中取出一个相机



SIBR_gaussianViewer_app.ex3e -m xxx/output/tandt/train
其中train目录结构为：
  point_cloud // 目录
    iteration_7000
      point_cloud.ply // 约100M
  cameras.json
  cfg_args

可以看到正确的结果。
```

## 4.代码OpenSplat
```

项目Setup: 相对gaussian-splatting-cuda要简单。只需要安装open_cv.
执行命令: openspalt.exe D:\CopyX\SplattingData\OpenSplat\train -n 7000 --valRender
输出： cameras.json和splat.ply(同样train场景，同样的step数, 此文件为60多M) 。

TODO: 即使将cameras.json和spalt.ply安装上面提到的目录结构组织，仍然无法用SIBR打开。

对比了一下与gaussian-splatting-cuda的代码，主要核心部分应该是非常接近的。 对比了一下性能，比前一个代码训练快一些。

```

## 5.问题 
```
后续已经有非常多的工作了，比如这个页面中有大量3dgs方面的工作，有许多方向，现在想在某个方向上找到效果最好地去基于OpenSplat去修改代码实现。
```






···
