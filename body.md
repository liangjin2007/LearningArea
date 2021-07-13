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
conda create -n frankmocap python=3.7
conda activate frankmocap
//# CUDA 10.1
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch

// download opendr
cd opendr/opendr

pip install glfw
python setup.py install

cd frankmocap/docs

// 从requirements.txt删除opendr那一行
