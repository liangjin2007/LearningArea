C++开发中经常会用到第三方库，此文档记录这些年的一些经验：

**目录:**
- 编译成目标平台的库
- 使用过的第三方库 
  - 求解器
  - 几何算法
  - DCC SDK
  - GPU Programming
  - Graphics
  - 数据格式
  - 引擎
  - 网络
  - 其他

## 编译成目标平台的库

看第三方库是哪个类型的工程？ 目前接触过的有：

  Visual Studio工程

  CMake 工程

  qt工程

  make 工程

  jam 工程

  只有c++源代码

  Header-Only 库

最容易的是Header-Only库，不需要编译。


## 使用过的第三方库

### 求解器
#### ceres [](https://github.com/ceres-solver/ceres-solver)
```
1. 解非线性最小二乘问题， 带边界约束。

2. 解无约束优化问题
```

代码示例：
```
using ceres::DynamicAutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

Problem problem;

struct ResidualFuncXXX {
ResidualFuncXXX(...);

// Critical member function to define weighted residual value
template <typename T> bool operator()(T const* const* w, T* residual) const {
	// e.g.
	residual[0] = xxx;
	residual[1] = xxx;
	residual[2] = xxx;
	
	return true;
}

// const members related to residual functions, e.g. constant coefficients, vectors, matrices etc.
}

// Let n be variable object count

// one parameter block x
std::vector<double> xs(n);
std::vector<double*> parameter_blocks;
parameter_blocks.push_back(&xs[0]);

// Add residuals to cost
auto cost_function_i = new DynamicAutoDiffCostFunction<ResidualFuncXXX, 4>(new ResidualFuncXXX(...));
cost_function_i->AddParameterBlock(n);
cost_function_i->SetNumResiduals(2);
problem.AddResidualBlock(cost_function_i, NULL, parameter_blocks);  

for (i = 0; i < n; i++) {
	problem.SetParameterLowerBound(parameter_blocks[0], i, lower_bound);
	problem.SetParameterUpperBound(parameter_blocks[0], i, upper_bound);
};

Solver::Options options;
options.linear_solver_type = ceres::ITERATIVE_SCHUR;
options.num_threads = 8;
options.num_linear_solver_threads = 8; // only SPARSE_SCHUR can use this
options.minimizer_progress_to_stdout = true;
options.max_num_iterations = 100;

Solver::Summary summary;
Solve(options, &problem, &summary);
std::cout << summary.BriefReport() << "\n";
```

#### mosek64_9_1 [](https://docs.mosek.com/latest/capi/index.html)
线性规划，二次规划等优化问题都可以用mosek求解。 从SDK中寻找[doc/capi.pdf](https://github.com/liangjin2007/math_related/blob/main/mosek_capi.pdf)，可以看到c语言版本的api 。

libigl中有一个将mosek解二次规划的问题封装成Eigen接口。 

这边我重写过一个版本。

基于超平面约束二次规划的头发分离工具。



### 几何算法
#### libigl
- https://cmake.org/cmake/help/latest/module/ExternalProject.html
深入使用
```
igl::path_to_executable()
igl::mosek::mosek_quadprog()
igl::unproject_onto_mesh()
igl::harmonic
```


#### openvdb
较深入地使用过，实现过VR中的实时体素刷子（带颜色的交并差）
#### gvdb
较深入地使用过，大型体素编辑器及实时动态高性能SDF生成。
#### PoissonRecon
#### tetgen 
- source code https://github.com/ufz/tetgen
- 文档 https://github.com/liangjin2007/math_related/blob/main/tetgen.pdf
```
使用不多，刚测试跑通
    tetgenio in, out;
    int i;

    // All indices start from 1.
    in.firstnumber = 0;

    in.numberofpoints = 8;
    in.pointlist = new REAL[in.numberofpoints * 3];
    in.pointlist[0] = 0;  // node 1.
    in.pointlist[1] = 0;
    in.pointlist[2] = 0;
    in.pointlist[3] = 2;  // node 2.
    in.pointlist[4] = 0;
    in.pointlist[5] = 0;
    in.pointlist[6] = 2;  // node 3.
    in.pointlist[7] = 2;
    in.pointlist[8] = 0;
    in.pointlist[9] = 0;  // node 4.
    in.pointlist[10] = 2;
    in.pointlist[11] = 0;
    // Set node 5, 6, 7, 8.
    for (i = 4; i < 8; i++) {
        in.pointlist[i * 3] = in.pointlist[(i - 4) * 3];
        in.pointlist[i * 3 + 1] = in.pointlist[(i - 4) * 3 + 1];
        in.pointlist[i * 3 + 2] = 12;
    }

    // For pure Delaunay triangulation, "Q" suppresses all output except errors, and "z" indexes the output starting from zero.
    tetrahedralize((char*)"Qz", &in, &out);

    //// Output mesh to files 'barout.node', 'barout.ele' and 'barout.face'.
    //out.save_nodes((char*)"barout");
    //out.save_elements((char*)"barout");
    //out.save_faces((char*)"barout");

    std::cout << "Generated " << out.numberoftetrahedra << " tetrahedra." << std::endl;
    for (int i = 0; i < out.numberoftetrahedra; i++)
    {
        int* tetrahedra = &(out.tetrahedronlist[i * 4]);
        std::cout << "tetrahedra " << i << ": " << tetrahedra[0] << ", " << tetrahedra[1] << ", " << tetrahedra[2] << ", " << tetrahedra[3] << std::endl;
    };

    
    // This will output a mesh tetgen-tmpfile.1.mesh in cmake build directory. Use Gmsh to visualize it.
    tetrahedralize((char*)"Qzg", &in, nullptr);

    // This will output a mesh tetgen-tmpfile.1.vtk in cmake build directory.
    //tetrahedralize((char*)"Qzk", &in, nullptr);
```
#### fTetWild
这个库没它所说的那么好用，经常出现奇奇怪怪的问题
#### pcl
```
typedef pcl::PointXYZ PointXYZT;
typedef pcl::PointXYZLNormal PointT;
typedef pcl::PointXYZRGBNormal PointColorT;

pcl::PointCloud<PointT> cloud;
pcl::KdTreeFLANN<PointT> kdtree;
pcl::sqrPointToLineDistance(pt, linept, linedir)
pcl::PLYWriter writer; writer.write<PointT>(filename, cloud, true, false);
pcl::io::loadPLYFile(filename.c_str(), cloud);

pcl::PointCloud<pcl::PointXYZ> points;
...
std::vector<pcl::Vertices> polygons;
for (auto& tri : Triangles)
{
pcl::Vertices triangle;
triangle.vertices.push_back(tri[0]);
triangle.vertices.push_back(tri[1]);
triangle.vertices.push_back(tri[2]);
polygons.push_back(triangle);
}
pcl::PolygonMesh mesh;
pcl::PCLPointCloud2 points_blob;
pcl::toPCLPointCloud2(*points, points_blob);
mesh.cloud = points_blob;
mesh.polygons = polygons;
pcl::io::savePLYFile(filename, mesh);
```


### DCC SDK
#### Maya c++ toolkit， Maya mel
从0到1开发过单目/双目驱动面部动画, 动画编辑器。
```
节点，DG图， plug, attribute。

设计插件。

实践：
Maya开启socket监听接口1000, commandPort -bs 10000000 -n ":1000"; c++侧可以通过socket连接到该端口，然后发送mel命令给Maya以执行批处理数据的功能。

```

- Maya c++ plugin MPxData
```


#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>
#include <maya/MPxData.h> // For self defined data

如果我们已经有一个类有一些成员变量和函数，可以直接把它封装成一个maya的数据。

class Algorithm
{
public:
	Algorithm();
	void Do();
private:
	Eigen::VectorXd input;
	Eigen::MatrixXd coeff;
};

class AlgorithmData : public MPxData
{
	public:
		AlgorithmData();
		~AlgorithmData() override;

		// Override methods in MPxData.
		//
		MStatus         readASCII(const MArgList&, unsigned& lastElement) override;
		MStatus         readBinary(istream& in, unsigned length) override;
		MStatus         writeASCII(ostream& out) override;
		MStatus         writeBinary(ostream& out) override;
		void			copy(const MPxData&) override;
		MTypeId                 typeId() const override;
		MString					name() const override;

		const Algorithm& algorithm() const;
		Algorithm& algorithm();
	
		static const MString    typeName;
		static const MTypeId    id;
		static void*            creator();

	private:
		Algorithm data;
};
```

- Maya c++ plugin MPxNode
```
class AlgorithmNode : public MPxNode
{
public:
    AlgorithmNode();
    ~AlgorithmNode() override;
    
    MStatus compute(const MPlug &plug, MDataBlock &data) override{
	MStatus status;
	if (plug == aAlgorithmOutput)
	{
		int count = datablock.inputValue(aInput1).asInt();
		MDataHandle h = datablock.inputValue(aAlgorithmData);
		AlgorithmData* data = dynamic_cast<AlgorithmData*>(h.asPluginData());
		Algorithm& algo = data->algorithm();
		algo.Do(count);

	        MArrayDataHandle arrayHandle = datablock.outputArrayValue(aAlgorithmOutput, &status);
		if (arrayHandle.elementCount() != count)
		{
			MArrayDataBuilder builder = arrayHandle.builder(&status);
			for (int i = 0; i < count; i++) {
				builder.addElement(i).asFloat();
			};

			arrayHandle.set(builder);
		}

		for (int i = 0; i < count; i++)
		{
			arrayHandle.jumpToElement(i);
			MDataHandle cvHandle = arrayHandle.outputValue();
			float &cvValue = cvHandle.asFloat();
			cvValue = algo.output[i];
		};

		arrayHandle.setAllClean();

		datablock.setClean(plug);
	}
	return status;
    }

    static void *creator(){
    	return new AlgorithmNode();
    }
    static MStatus initialize()
    {
	MStatus status;
	MFnNumericAttribute nAttr;
	MFnTypedAttribute tAttr;

	aInput1 = nAttr.create(attr_name_long1, attr_name_short1, MFnNumericData::kFloat);
	aAlgorithmData = tAttr.create(attr_name_long2, attr_name_short2, AlgorithmData::id, MObject::kNullObj);
	aAlgorithmOutput = nAttr.create(attr_name_long3, attr_name_short3, MFnNumericData::kFloat);
        status = nAttr.setArray(true);
        status = nAttr.setUsesArrayDataBuilder(true);
	status = addAttribute(aInput1);
	status = addAttribute(aAlgorithmData);
	status = addAttribute(aAlgorithmOutput);
	status = attributeAffects(aInput1, aAlgorithmOutput);
	status = attributeAffects(aAlgorithmData, aAlgorithmOutput);
	return status;
    }

    static MString nodeName;
    static MTypeId id;

    static MObject aInput1;
    static MObject aAlgorithmData;

    // Output.
    static MObject aAlgorithmOutput;
};
```

- Maya c++ plugin command
```
class AlgorithmCmd : MPxCommand
{
public:
    AlgorithmCmd();
    ~AlgorithmCmd() override;
    MStatus doIt(const MArgList &args) override
    {
	MArgParser argData(syntax(), args, &status);
	int argXXXX;
	...
	if (argData.isFlagSet(xxxx))
	    argData.getFlagArgument(xxxx, 0, argXXXX);
	...

	// Create node

	// Connect node attributes

	// Setup attribute values.
    }

    inline bool isUndoable() const override { return false;  }
    static void *creator();
    static void cleanup();
    static MSyntax newSyntax();
    static MString commandName;
	static MString importerName;
	static MString trainerName;
	static MString trackerName;
	static MString retargetingName;
};
```
  
- Maya c++ plugin registeration/unregistration
```
//register plugin
MStatus initializePlugin(MObject obj){
	MStatus status;
	MFnPlugin pluginFn(obj, compony_str, version_str, other_str, &status);

	status = pluginFn.registerData("AlgorithmData", AlgorithmData::id, AlgorithmData::creator);
	status = pluginFn.registerNode(AlgorithmNode::nodeName, AlgorithmNode::id, AlgorithmNode::creator, AlgorithmNode::initialize, MPxNode::kDependNode);
	status = pluginFn.registerCommand(AlgorithmCmd::commandName, AlgorithmCmd::creator, AlgorithmCmd::newSyntax);
	return status;
}
MStatus uninitializePlugin(MObject obj) {
	MStatus status;
	MFnPlugin pluginFn(obj);
	status = pluginFn.deregisterCommand(AlgorithmCmd::commandName);
	status = pluginFn.deregisterNode(AlgorithmNode::id);
	status = pluginFn.deregisterData(AlgorithmData::id);
	return status;
}
```


#### Houdini HDK, hou python scripting
从0到1开发过实时刷子
```
python viewer state for mouse interaction
c++ SOP plugin for geometry processing
vscode 打开 HoudiniX.Y/houdini/python3.10.libs/， HoudiniX.Y/toolkit, HoudiniX.Y/houdini/viewer_states, HoudiniX.Y/houdini/viewer_handles
浏览器打开 Houdini文档 https://www.sidefx.com/docs/houdini/index.html
ChatGPT 搜索相关概念和实现，往往有幻觉出入，需要自己验证。
```

### 视觉
#### opencv
使用过的版本为4.11的gpu版本。

在做深度学习期间，还使用过python版的opencv （pip install opencv-python）

```
// 读图片
cv::Mat img = cv::imread(input_img);

// 转换颜色空间，从bgr 2 hsv
cv::Mat img1;
cv::cvtColor(img, img1, cv::COLOR_BGR2HSV);

...

// 矩阵运算
cv::absdiff(img1, img2, img3);
cv::mean


// 灰度图的blob detection
// use blob detector to detect similar pixel components in images. 
std::vector<cv::KeyPoint> keypoints;

// ref: How to setup Blob parameters ? https://www.geeksforgeeks.org/find-circles-and-ellipses-in-an-image-using-opencv-python/
cv::SimpleBlobDetector::Params params;
// setup params
...

cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
detector->detect(gray, keypoints);


// indexing
uchar value = gray.at<uchar>(py, px);


// 显示图片
cv::imshow("title", img);
cv::waitKey(0);

// 图形绘制
cv::line
cv:circle
cv::putText
cv::rectangle

// 鼠标回调
cv::setMouseCallback(...);

// 常用数据结构
cv::Scalar, cv::Size, cv::Rect, cv::Point, cv::KeyPoint, cv::Point2f

// utilities
cv::format
```

#### colmap-3.5

著名的MVS库，编译非常复杂, 第三方库较多非常折腾。 CameraMatrix * ProjectMatrix * X = homogeneous 2d coordinate
Utility函数比较方便。

```


// utilities
colmap::JointPaths
colmap::StringTrim
colmap::StringPrintf
colmap::StringSplit
colmap::StringReplace
colmap::GetParentDir
colmap::CreateDirIfNotExist
colmap::ExistsFile
colmap::ExistsDir



// Structs
colmap::Camera
colmap::Image

// MVS function
std::vector<>colmap::TriangulatePoints
auto err = colmap::CalculateReprojectionError(point2d, point3d, projmatrix, camera);
Eigen::Vector2d uv = colmap::ProjectPointToImage(p, projmatrix, camera);

// Math
Eigen::Vector4d q = colmap::RotationMatrixToQuaternion(R);
Eigen::Matrix3d R = colmap::EulerAnglesToRotationMatrix(ex, ey, ez);
auto rad = colmap::DegToRad(deg);
double r = colmap::RandomReal<double>(-0.0001, 0.0001);

```

### GPU Programming

#### CUDA
熟悉CUDA概念和编程，善于涉及算法。NSight Compute和NSight System
```
修改当前Visual Studio 工程以支持CUDA。
	在Visual Studio .vcproj的两处添加CUDA 11.4.props。
运行时Runtime SDK
Kernel执行方式
   kernel_name<<<grid_dim, block_dim>>>(param, ...);
   cuLaunch
   cuda graph execution
同步执行和异步执行
异步执行的同步化
Context
Stream
Event
block， warp等概念。
如何知道一个显卡的最大线程数。
如何优化当前代码
如何设计算法

```

#### Optix
[之前有单独记录过一个页面](https://github.com/liangjin2007/LearningArea/blob/master/optix.md)

#### lib torch
未能尝试编写代码
```
#include "torch/script.h"
std::vector<float> v;
...
torch::Tensor tv = torch::tensor(v).t();
torch::Tensor c = torch::matmul(a, b).t();
torch::Tensor d = c.reshape({ m, n }).t();
torch::Tensor e = torch::inverse(rotate);

torch::jit::script::Module m = torch::jit::load("xxx.pt");
torch::Tensor out = m.forward({input_tensor}).toTensor();
```


### Graphicis
- opengl API
```
opengl API。
glsl shaders。
RenderDoc调试代码。
HairStrandsRendering中集成几十个G的体素可视化，修改render函数支持mesh, 头发， volume的深度遮挡关系。
可视化编辑几十个G的体素。
```
- DX12
```

```
### 数据格式
#### Alembic库
这个接触较多。
#### FBXSDK



### 引擎
- 如何将一个c++库封装成UE的插件
```
创建一个动态链接库（DLL）用纯C接口封装这个库的功能（UE不支持使用stl）。

#ifdef XXX_STATIC
#define XXX_API
#else
#ifdef XXX_EXPORT
#define XXX_API _declspec(dllexport)
#else
#define XXX_API _declspec(dllimport)
#endif
#endif
```  
- maya-implicit-skinning
```
开源代码，有助于快速接触maya动画。

```


### 网络
ZeroMQ, librdkafka, draco, ZmqNetwork, protobuf




### 其他
#### stl
#### boost
filesystem后续新版C++基本上支持相同功能。

#### Eigen
用得比较多，默认col major。

求解稀疏最小二乘

求解RBF，用于做面补驱动。 

Eigen从某一个版本开始支持在CUDA kernel或者device function中使用Eigen，有个类似于EIGEN_GPU的宏控制。

```
Eigen::Map
Eigen::MatrixXf
Eigen::VectorXd
Eigen::RowVectorXd
Eigen::Vector3d
Eigen::Vector3f
Eigen::Quaternion
Eigen::SparseMatrix
Eigen::Matrix
.block<3, 3>(start_row, start_col)
```

#### json nlohmann
```
#include "json.hpp"
using nlohmann::json json;

std::ifstream i("xxx.json");
json jobj = json::parse(i);

json jobj;
std::vector<float> value = jobj[index0][index1].get<vector<float>>();

```

#### .conf library cpptoml

需要修改代码来支持非 param = v的行。

```
auto config = cpptoml::parse_file(conffile);
auto a = *(config->get_qualified_as<std::string>(param_name_str));
auto b = *(config->get_array_of<int64_t>(param_name_str));
*(config->get_as<int>(param_name_str));
if(config->contains(param_name_str)){}
```

#### CLI -- command line interface
```
#include "CLI/CLI.hpp"

struct Args
{
std::string a;
bool b = false;
...
};

Args parse_args(int argc, char* argv[])
{
Args args;

CLI::App app{ "appication name" };

app.add_option("--xxx", args.a, default_value_str);
app.add_flag("--b", args.b, comment_str);
...
try 
{
app.parse(argc, argv);
}
catch (const CLI::ParseError& e) {
exit(app.exit(e));
}

return args;
}
```

#### cxxopts
```
#include "cxxopts.hpp"
cxxopts::Options options(str1, str2);
	options.add_options()
		("i,ixxxx", str1, cxxopts::value<std::string>())
		("b,bxxxx", str2, cxxopts::value<int>())
		("c,cxxxx", str3)
		("d,dxxxx", str4, cxxopts::value<int>()->default_value("3"))
		;
auto result = options.parse(argc, argv);

if (result.count("help"))
{
	std::cout << options.help() << std::endl;
	exit(0);
}

bool value = result["ixxxx"].as<bool>();

```
#### glog
- https://google.github.io/glog/stable/
```
#include <glog/logging.h>

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]); 
    LOG(INFO) << "Found " << num_cookies << " cookies"; 
}
```

#### gflags
类似于CLI, cxxopts，用于command line flags.
```
find_package(gflags REQUIRED)
add_executable(foo main.cc)
target_link_libraries(foo gflags::gflags)


#include <gflags/gflags.h>

DEFINE_bool(big_menu, true, "Include 'advanced' options in the menu listing");
DEFINE_string(languages, "english,french,german",
	 "comma-separated list of languages to offer in the 'lang' menu");

DEFINE_bool defines a boolean flag. Here are the types supported:
DEFINE_bool: boolean
DEFINE_int32: 32-bit integer
DEFINE_int64: 64-bit integer
DEFINE_uint64: unsigned 64-bit integer
DEFINE_double: double
DEFINE_string: C++ string

// Accessing flag 
if (FLAGS_consider_made_up_languages)
FLAGS_languages += ",klingon";   // implied by --consider_made_up_languages
if (FLAGS_languages.find("finnish") != string::npos)
HandleFinnish();


// 验证flag值
static bool ValidatePort(const char* flagname, int32 value) {
   if (value > 0 && value < 32768)   // value is ok
     return true;
   printf("Invalid value for --%s: %d\n", flagname, (int)value);
   return false;
}
DEFINE_int32(port, 0, "What port to listen on");
DEFINE_validator(port, &ValidatePort);

// 在另一个文件中使用flag
DECLARE_bool(big_menu);

// How to Set Up Flags
 gflags::ParseCommandLineFlags(&argc, &argv, true);

// Special Flags
--help
```













  


  
  
