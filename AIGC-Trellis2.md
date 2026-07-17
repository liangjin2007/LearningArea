- hugging face
```
https://huggingface.co/microsoft/TRELLIS.2-4B/tree/main
```

- hugging face password
```
Xr1
```

- Download models
```
# Make sure git-xet is installed (https://hf.co/docs/hub/git-xet)
winget install git-xet

git clone https://huggingface.co/microsoft/TRELLIS.2-4B

# If you want to clone without large files - just their pointers
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/microsoft/TRELLIS.2-4B

# Make sure the hf CLI is installed
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"

# Download the model
hf download microsoft/TRELLIS.2-4B
```

- Use Ubuntu to setup
```
git config --global user.name "xxx"
git config --global user.email "xxx"
ssh-keygen -t rsa

# clone download
git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive

# create env trellis2
./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

# Setup hf
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
# Download the model
hf download microsoft/TRELLIS.2-4B
```
