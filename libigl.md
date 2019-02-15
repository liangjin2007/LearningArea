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
    // Read scalar function values from a file, U: #V by 1
    VectorXd U;
    igl::readDMAT(TUTORIAL_SHARED_PATH "/cheburashka-scalar.dmat",U);
    
    // Compute per-face normals
    igl::per_face_normals(V,F,N_faces);
    
    // Compute per-vertex normals
    igl::per_vertex_normals(V,F,N_vertices);
    
    // Compute per-corner normals, |dihedral angle| > 20 degrees --> crease
    igl::per_corner_normals(V,F,20,N_corners);
    
    // Compute integral of Gaussian curvature
    igl::gaussian_curvature(V,F,K);
    
    // Compute mass matrix
    SparseMatrix<double> M,Minv;
    igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_DEFAULT,M);
    
    // Compute invert matrix
    igl::invert_diag(M,Minv);
    // Divide by area to get integral average
    K = (Minv*K).eval();
    
    MatrixXd HN;
    SparseMatrix<double> L,M,Minv;
    igl::cotmatrix(V,F,L);
    igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_VORONOI,M);
    igl::invert_diag(M,Minv);
    HN = -Minv*(L*V);
    H = HN.rowwise().norm(); //up to sign, H是每个定点上的平均曲率
    
    // Compute curvature directions via quadric fitting
    MatrixXd PD1,PD2;
    VectorXd PV1,PV2;
    igl::principal_curvature(V,F,PD1,PD2,PV1,PV2);
    // mean curvature
    H = 0.5*(PV1+PV2);
    
    // Average edge length for sizing
    const double avg = igl::avg_edge_length(V,F);
    
    // Compute gradient operator: #F*3 by #V
    SparseMatrix<double> G;
    igl::grad(V,F,G);
    // Compute gradient of U
    MatrixXd GU = Map<const MatrixXd>((G*U).eval().data(),F.rows(),3);
    // Compute gradient magnitude
    const VectorXd GU_mag = GU.rowwise().norm();

    // Compute Laplace-Beltrami operator: #V by #V
    igl::cotmatrix(V,F,L);

    // Alternative construction of same Laplacian
    SparseMatrix<double> G,K;
    // Gradient/Divergence
    igl::grad(V,F,G);
    // Diagonal per-triangle "mass matrix"
    VectorXd dblA;
    igl::doublearea(V,F,dblA);
    // Place areas along diagonal #dim times
    const auto & T = 1.*(dblA.replicate(3,1)*0.5).asDiagonal();
    // Laplacian K built as discrete divergence of gradient or equivalently
    // discrete Dirichelet energy Hessian
    K = -G.transpose() * T * G;
    cout<<"|K-L|: "<<(K-L).norm()<<endl;
    
    // Smoothing
    
    igl::exact_geodesic(V,F,VS,FS,VT,FT,d);
    
    // matrix and linear algebra
    igl::slice(A,R,C,B);
    igl::sort(X,1,true,Y,I);
    igl::sortrows(X,true,Y,I);
    // Find boundary edges
    MatrixXi E;
    igl::boundary_facets(F,E);
    // Find boundary vertices
    VectorXi b,IA,IC;
    igl::unique(E,b,IA,IC);
    // List of all vertex indices
    VectorXi all,in;
    igl::colon<int>(0,V.rows()-1,all);
    // List of interior indices
    igl::setdiff(all,b,in,IA);

    // Construct and slice up Laplacian
    SparseMatrix<double> L,L_in_in,L_in_b;
    igl::cotmatrix(V,F,L);
    igl::slice(L,in,in,L_in_in);
    igl::slice(L,in,b,L_in_b);

    // Dirichlet boundary conditions from z-coordinate
    VectorXd bc;
    VectorXd Z = V.col(2);
    igl::slice(Z,b,bc);

    // Solve PDE
    SimplicialLLT<SparseMatrix<double > > solver(-L_in_in);
    VectorXd Z_in = solver.solve(L_in_b*bc);
    // slice into solution
    igl::slice_into(Z_in,in,Z);

    // Alternative, short hand
    igl::min_quad_with_fixed_data<double> mqwf;
    // Linear term is 0
    VectorXd B = VectorXd::Zero(V.rows(),1);
    // Empty constraints
    VectorXd Beq;
    SparseMatrix<double> Aeq;
    // Our cotmatrix is _negative_ definite, so flip sign
    igl::min_quad_with_fixed_precompute((-L).eval(),b,Aeq,true,mqwf);
    igl::min_quad_with_fixed_solve(mqwf,B,bc,Beq,Z);
    ```
- 第二章 离散几何量和算子
    - 法向 https://github.com/libigl/libigl/blob/master/tutorial/201_Normals/main.cpp
      - Per-corner效果最好
    - 高斯曲率 https://github.com/libigl/libigl/blob/master/tutorial/202_GaussianCurvature/main.cpp
    - 平均曲率，主曲率 https://github.com/libigl/libigl/blob/master/tutorial/203_CurvatureDirections/main.cpp
    - 梯度 https://github.com/libigl/libigl/blob/master/tutorial/204_Gradient/main.cpp
      - hat function defined on 1-ring of a vertex.
    - 拉普拉斯相关
      - 欧氏空间中的拉普拉斯算子
        - 梯度的散度
      - Laplace-Beltrami operator
        - 一般化到曲面
        - 离散情况有许多种方式
        - cotangent Laplacian
        - divergence theorem to vertex one-rings.
    - 质量矩阵
    - 精确离散测地距离
- 第三章 矩阵和线性代数
  - Matlab风格的函数
  - 拉普拉斯方程
  
     
# Eigen

Eigen::MatrixXd V; // #V x 3

Eigen::MatrixXi F; // #F x 3

Eigen::VectorXd Z; Z = V.col(2); // #V x 1

Eigen::RowVector3d(r, g, b); // 1 x 3

Eigen::Vector3d m = V.colwise().minCoeff(); // avoid to write for loops.
Eigen::Vector3d M = V.colwise().maxCoeff(); 

Eigen::SparseMatrix<double>
# GUI
  - 可视化曲面
    ```
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F); // 拷贝mesh到viewer
    viewer.data().set_normals(N_faces); //
    viewer.data().set_colors(C); // 可视化标量，#C must be #F or #V.
    viewer.callback_key_down = &key_down; // 键盘鼠标交互
    viewer.selected_data_index; // multiple meshes
    viewer.data().add_points(P, Eigen::RowVector3d(r,g,b)); // 挂件
    viewer.data().add_edges(P1,P2,Eigen::RowVector3d(r,g,b)); // 挂件
    viewer.data().add_label(p,str); // 挂件
    viewer.data().show_lines = false; // 关掉wireframe
    viewer.data().set_vertices(U);
    viewer.data().compute_normals();
    viewer.core.align_camera_center(U,F);
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
    


