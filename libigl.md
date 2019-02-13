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
  - igl::jet(Z, true, C); // transform Z to C.
- Overlays（挂件？覆盖）
  - viewer.data().add_points(P, Eigen::RowVector3d(r,g,b));
  - viewer.data().add_edges(P1,P2,Eigen::RowVector3d(r,g,b));
  - viewer.data().add_label(p,str);
- 使用矩阵可以避免循环写法
  - colwise().minCoeff()
- 菜单
  - ImGui
  ```
  igl::opengl::glfw::Viewer viewer;
  igl::opengl::glfw::imgui::ImGuiMenu menu;
  viewer.plugins.push_back(&menu);
  menu.callback_draw_viewer_menu = [&](){}
  ```
  - [menu example](https://github.com/libigl/libigl/blob/master/tutorial/106_ViewerMenu/main.cpp)
-
# Eigen

Eigen::MatrixXd V; // #V x 3

Eigen::MatrixXi F; // #F x 3

Eigen::VectorXd Z; Z = V.col(2); // #V x 1

Eigen::RowVector3d(r, g, b); // 1 x 3

Eigen::Vector3d m = V.colwise().minCoeff(); // avoid to write for loops.
Eigen::Vector3d M = V.colwise().maxCoeff(); 
# GUI


# Algorithm

