```
conda update -n base -c defaults conda
conda env list
conda create -n xxx python=xxx
conda create --file xxx.yml
conda remove -n ct2hair --all
powershell: conda init powershell

// setup packages directory to another path.
// setup env to another path
conda config --add pkgs_dirs <Drive>/<pkg_path>
conda env create --file environment.yml --prefix <Drive>/<env_path>/gaussian_splatting
conda activate <Drive>/<env_path>/gaussian_splatting


```





