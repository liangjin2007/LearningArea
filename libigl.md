# libigl
An Open Source Framework For Geometry Processing Programming.

- 第一章 基础知识
  - 设计原则
  - 下载源代码
  - 编译
  - 例子工程
  - 网格表示
    - 内存高效，缓存友好，避免指针，拷贝和序列化比较方便。
  - 基本API
    ```
    igl::readOFF(path, V, F);
    igl::writeOBJ("cube.obj", V, F);
    igl::jet(Z, true, C); // transform Z to C.
    
    // Compute per-face normals
    igl::per_face_normals(V,F,N_faces);
    
    // Compute per-vertex normals
    igl::per_vertex_normals(V,F,N_vertices);
    
    // Compute per-corner normals, |dihedral angle| > 20 degrees --> crease
    igl::per_corner_normals(V,F,20,N_corners);
    
    / Compute integral of Gaussian curvature
    igl::gaussian_curvature(V,F,K); 
    // Compute mass matrix
    SparseMatrix<double> M,Minv;
    igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_DEFAULT,M);
    igl::invert_diag(M,Minv);
    // Divide by area to get integral average
    K = (Minv*K).eval();
    
    ```
  - 可视化曲面
    ```
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F); // 拷贝mesh到viewer
    viewer.data().set_normals(N_faces); //
    viewer.data().set_colors(C); // 可视化标量，#C must be #F or #V.
    viewer.callback_key_down = &key_down; // 键盘鼠标交互 
    viewer.data().add_points(P, Eigen::RowVector3d(r,g,b)); // 挂件
    viewer.data().add_edges(P1,P2,Eigen::RowVector3d(r,g,b)); // 挂件
    viewer.data().add_label(p,str); // 挂件
    viewer.data().show_lines = false; // 关掉wireframe
    
    viewer.launch();
    ```
  - 菜单
    - ImGui
    ```
    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);
    menu.callback_draw_viewer_menu = [&](){}
    ```
    - [menu example](https://github.com/libigl/libigl/blob/master/tutorial/106_ViewerMenu/main.cpp)
  
  - 多网格
    - viewer.selected_data_index
    - igl::ViewerData
    
- 第二章 离散几何量和算子
    - 法向 https://github.com/libigl/libigl/blob/master/tutorial/201_Normals/main.cpp
      - Per-corner效果最好
    - 高斯曲率 https://github.com/libigl/libigl/blob/master/tutorial/202_GaussianCurvature/main.cpp
    - 
# Eigen

Eigen::MatrixXd V; // #V x 3

Eigen::MatrixXi F; // #F x 3

Eigen::VectorXd Z; Z = V.col(2); // #V x 1

Eigen::RowVector3d(r, g, b); // 1 x 3

Eigen::Vector3d m = V.colwise().minCoeff(); // avoid to write for loops.
Eigen::Vector3d M = V.colwise().maxCoeff(); 

Eigen::SparseMatrix<double>
# GUI


# Algorithm

