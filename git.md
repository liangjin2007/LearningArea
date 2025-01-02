```
git status
git config --global user.name "xxx"
git config --global user.email "xxx"
ssh-keygen -t rsa
copy the content of .ssh/xxx.pub to add ssh key
git clone xxx
git submodule update --init --recursive
cd xxx
git checkout xxx 放弃修改，
git branch xxx 创建本地branch
git push origin xxx 上传本地的branch
git push origin -d xxx 删除服务器上的branch

git add . 添加所有本地修改
git reset . revert 'git add'


git checkout -- file
git checkout . #放弃所有
git clean -xdf  #删除未被跟踪untracked的文件，这个测试了，确实有用。

git stash #把所有没有提交的修改暂存到stash里面。可用git stash pop回复。

git lfs track *.fbx大文件使用lfs文件系统
```

