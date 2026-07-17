- hugging face
```
https://huggingface.co/microsoft/TRELLIS.2-4B/tree/main
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
