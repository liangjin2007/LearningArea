#### 2D pose
CMU

[2017]Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields
https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

[2019]OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields
https://github.com/CMU-Perceptual-Computing-Lab/openpose

#### facial 

#### hand
[2017]Hand keypoint detection in single images using multiview bootstrapping


#### siml-x https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/497/SMPL-X.pdf
```

```

#### [2021]Real-time RGBD-based Extended Body Pose Estimation




#### frankmocap
- Windows配置
```
conda create -n frankmocap python=3.7
conda activate frankmocap
//# CUDA 10.2
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
// download opendr
cd opendr/opendr

pip install glfw
python setup.py install

cd frankmocap/docs

// 从requirements.txt删除opendr那一行
pip install -r requirements.txt

// 安装windows版detectron2, 注意官方只支持linux/macos
cd ../..
git clone https://github.com/conansherry/detectron2

pip install git+https://github.com/facebookresearch/fvcore
pip install cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

cd detectron2
python setup.py install

```
