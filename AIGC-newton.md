```
conda create -n newton python=3.10
conda activate newton

python.exe -m pip install --no-build-isolation --no-cache-dir "imgui_bundle>=1.92.0"

pip install "newton[examples]"
git clone https://github.com/newton-physics/newton.git

or

cd newton
python.exe -m pip install -e ".[examples]"

python -m newton.examples
```
