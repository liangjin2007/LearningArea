# Caffe

Caffe以视觉任务为主，编译依赖里有opencv等第三方依赖库。


## [caffe-windows](https://github.com/happynear/caffe-windows) 注意这个貌似不是官方那个链接。 但此链接中有第三方库及手把手怎么编译起来的教程。
参照说明进行操作。
- 下载thirdparty，放到./windows/thirdparty里
- 拷贝./windows/CommonSettings.props.example为一个新的./windows/CommonSettings.props
- 因为我没有matlab，也不想使用python，所以把这个文件里的一些开关需要关闭。 将cuda version改为12.1.
- 想要在VS2019下编译
- 修改./windows/caffe/caffe.vcxproj， ./windows/libcaffe/libcaffe.vcxproj中的<WindowsTargetPlatformVersion>10.0.22000.0</WindowsTargetPlatformVersion>，及打开Visual Studio修改常用里的Platform Toolset为v142
- 注意cuDNN最好按照说明的下载V5， 并在./windows/CommonSettings.props中修改CuDNNPath为本地放cudnn的路径。
- 做完上面这些，再打开./windows/Caffe.sln设置好configuration为Release应该能编译好libcaffe项目。但是编译caffe项目时提示libboost_thread找不到。尝试找一个新版本的boost https://sourceforge.net/projects/boost/files/boost-binaries/1.65.1/, 下载boost_1_65_1-msvc-14.1-64.exe
- 拷贝一份-vc141-.lib到lib64-msvc-vc14.0目录中。
- caffe能编译通过了。
- CudaVersion必须设对，否则会导致caffe编译不过。
- https://blog.csdn.net/u013630299/article/details/105406952


## caffestudy https://github.com/koosyong/caffestudy/
简单例子演示Blob等。


## caffe学习资料
### [DIY Deep Learning](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit?pli=1#slide=id.g129385c8da_651_320)
#### [Model-Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
### [Tutorial](http://caffe.berkeleyvision.org/tutorial/)
- [command line](http://caffe.berkeleyvision.org/tutorial/interfaces.html)
- 
### cs231n
- Python Tutorial https://cs231n.github.io/python-numpy-tutorial/




