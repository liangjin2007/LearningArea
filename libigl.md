# libigl
An Open Source Framework For Geometry Processing Programming.

- 设计原则
- 下载源代码
- 编译
- 例子工程
- 网格表示
  - 内存高效，缓存友好，避免指针，拷贝和序列化比较方便。
- 网格IO
  - igl::readOFF(path, V, F);
  - igl::writeOBJ("cube.obj", V, F);
- 可视化曲面
  ```
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, F);
  viewer.launch();
  ```
- 键盘鼠标交互
  - viewer.callback_key_down = &key_down
- 可视化插件
- 可视化标量
  - viewer.data().set_colors(C); // #C must be #F or #V.
- 

# Eigen

Eigen::MatrixXd V;

Eigen::MatrixXi F;

# GUI


# Algorithm

