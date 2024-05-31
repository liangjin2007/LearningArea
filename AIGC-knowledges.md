## [awesome-visual-transformer]( https://github.com/dk-liang/Awesome-Visual-Transformer )
## [awesome-avatars](https://github.com/pansanity666/Awesome-Avatars?tab=readme-ov-file)
## [awesome-3d-AIGC](https://github.com/mdyao/Awesome-3D-AIGC)
## [awesome-AIGC-3D](https://github.com/hitcslj/Awesome-AIGC-3D)
## [awesome-CVPR2024-AIGC](https://github.com/Kobaayyy/Awesome-CVPR2024-AIGC)
## [awesome-autonomous-vehicle](https://github.com/DeepTecher/awesome-autonomous-vehicle)
## [awesome-3d-generation](https://github.com/justimyhxu/awesome-3D-generation)
## [awesome-3d-diffusion](https://github.com/cwchenwang/awesome-3d-diffusion)
## [awesome-3d-guassiansplatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting)
### Seminar paper [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://github.com/graphdeco-inria/gaussian-splatting)
- github代码没安装成功，转而切到c++版的一个实现https://github.com/MrNeRF/gaussian-splatting-cuda
#### gaussian-splatting-cuda
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
  添加变量
    set(PYTHONLIB_INCLUDE_DIR "${PROJ_ROOT_DIR}/external/pythonlib3.9/include")
    set(PYTHONLIB_LIBRARY "${PROJ_ROOT_DIR}/external/pythonlib3.9/libs/python39.lib")
  target_include_directories(${PROJECT_NAME} xxx ${PYTHONLIB_INCLUDE_DIR})
  target_link_libraries(${PROJECT_NAME}
        PRIVATE
        xxx 
        ${PYTHONLIB_LIBRARY}

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
设置为RelWithDbgInfo + x64。编译成功。
```

- 下载数据
```  
从 https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip 下载tandt_db.zip, unzip到data目录中。
得到:
  data/db
  data/tandt
```

- 命令行执行
```
Visual Studio启动方式：
  在Visual Studio 的项目属性->调试->命令参数 中设置 -d "xxx/gaussian-splatting-cuda/data/tandt/train" -o "xxx/gaussian-splatting-cuda/output/tandt/train"
  在Visual Studio 的项目属性->调试->环境 中添加 PATH=%PATH%;path/to/cuda/bin;path/to/pythonlib3.9; path/to/libtorch/lib

```
- 所有命令行选项
```
命令行选项
Core Options
-h, --help
Display this help menu.

-d, --data_path [PATH]
Specify the path to the training data.

-f, --force
Force overwriting of output folder. If not set, the program will exit if the output folder already exists.

-o, --output_path [PATH]
Specify the path to save the trained model. If this option is not specified, the trained model will be saved to the "output" folder located in the root directory of the project.

-i, --iter [NUM]
Specify the number of iterations to train the model. Although the paper sets the maximum number of iterations at 30k, you'll likely need far fewer. Starting with 6k or 7k iterations should yield preliminary results. Outputs are saved every 7k iterations and also at the end of the training. Therefore, even if you set it to 5k iterations, an output will be generated upon completion.

Advanced Options
--empty-gpu-cache Empty CUDA memory after ever 100 iterations. Attention! This has a considerable performance impact

--enable-cr-monitoring
Enable monitoring of the average convergence rate throughout training. If done, it will stop optimizing when the average convergence rate is below 0.008 per default after 15k iterations. This is useful for speeding up the training process when the gain starts to dimish. If not enabled, the training will stop after the specified number of iterations --iter. Otherwise its stops when max 30k iterations are reached.

-c, --convergence_rate [RATE]
Set custom average onvergence rate for the training process. Requires the flag --enable-cr-monitoring to be set.
```

- 修复crash
```
先修改项目gaussian_splatting_cuda的项目属性中的CUDA C/C++及CUDA Linker中调试相关的部分，重编译代码。
Debug发现Crash在一个linux specific 路径，并默认了CMake的输出binaries路径在gaussian_splatting_cuda里，比如gaussian_splatting_cuda/build。 
解决办法： 可以在Working Directory中添加gaussian_splatting_cuda的源代码根目录，然后修改原来的代码直接使用相对路径。
修复完成。
```

- 训练
```
7000次迭代 138s
```  

- 查看训练结果
```
git clone --recursive https://gitlab.inria.fr/sibr/sibr_core SIBR_core
CMake GUI可直接配置成功。
打开Visual Studio 2019编译成功。

./SIBR_viewers/install/bin/SIBR_gaussianViewer_app.exe -m /path/to/output
```
