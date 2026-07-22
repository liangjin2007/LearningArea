- 安装cmake
```
https://cmake.org/download/

```
- 安装cuda toolkit 12.8：
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

- 安装nvidia driver

- 验证是否装成功cuda
```
nvcc --version
nvidia-smi
```


- Download libtorch
```
从pytorch官网找到libtorch的安装指令 https://pytorch.org/get-started/previous-versions/?_gl=1*1bjf1ve*_up*MQ..*_ga*MTU4OTQ1NTI4MS4xNzg0NjE4MTc4*_ga_469Y0W5V62*czE3ODQ2MTgxNzckbzEkZzAkdDE3ODQ2MTgxNzckajYwJGwwJGgw
xxx2.9.0.xxx.zip
```

- 安装ninja
```
sudo apt install ninja-build
```

- 写Setup.sh
```
执行Setup.sh
```

- 





## 代码问题汇总

- glfw找不到
```
linux系统目录区分大小写，不像windows
```
- glfw cmake不过，找不到第三方库
```
sudo apt install libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libxext-dev
```

