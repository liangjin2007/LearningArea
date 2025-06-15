```
git status
git config --global user.name "xxx"
git config --global user.email "xxx"
git config lfs.https://git.feeling-ai.info/liang.jin/ue-feelingai.git/info/lfs.locksverify true

ssh-keygen -t rsa
copy the content of .ssh/xxx.pub to add ssh key
git clone xxx
git submodule update --init --recursive
cd xxx
git checkout xxx 放弃修改，
git branch xxx 创建本地branch
git push origin xxx 上传本地的branch
git push origin -d xxx 删除服务器上的branch

git add . 暂存所有本地修改
git reset . revert 'git add'

放弃本地新添加的目录和文件 git clean -fd

放弃本地的git add
如果你已经使用git add暂存了更改，但尚未提交，你可以使用git reset命令来取消暂存：
git reset HEAD <file>
例如，要取消暂存特定文件：
git reset HEAD myfile.txt

要取消暂存所有文件：
git reset HEAD .

如果你已经提交了更改，但希望撤销本次提交并保留文件的修改，可以使用git reset命令：
git reset --soft HEAD~1

如果你想撤销最近的提交并丢弃更改，可以使用git reset --hard：
git reset --hard HEAD~1

回退到某个版本
  git reset --hard <sha>
  git config lfs.https://git.feeling-ai.info/liang.jin/ue-feelingai.git/info/lfs.locksverify true
  git push -f

git checkout -- file
git checkout . #放弃所有
git clean -xdf  #删除未被跟踪untracked的文件，这个测试了，确实有用。

git stash #把所有没有提交的修改暂存到stash里面。可用git stash pop回复。

git lfs track *.fbx大文件使用lfs文件系统

git log xxxfile  显示提交记录

撤销单个文件的更改到某个sha
git checkout <sha> xxxfile

切换仓库到某个tag
git checkout tags/1.1.4

autocrlf相关
UE本身的代码换行为LF
在Windows上设置
  // 提交时转换为LF，检出时转换为CRLF
  git config --global core.autocrlf true  
在Mac上设置
  // 提交时转换为LF，检出时不转换
  git config --global core.autocrlf input

换行的安全检查
git config --global core.safecrlf true

创建仓库
git checkout -b backup-2025-03-31
更新到服务器
git push -u origin backup-2025-03-31

放弃 merge：如果你确定要放弃当前的 merge，可以使用 `git merge –abort` 命令


放弃一个已经Push的提交，并保留历史
  git log查看hash
  git revert <hash>
  git push


```

