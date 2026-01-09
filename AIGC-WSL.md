**Windows 11能很方便地安装一个Ubuntu子系统，它自己提供了一个服务。**

### Windows 11安装Ubuntu linux操作系统
- https://blog.csdn.net/bule_shake/article/details/135992375
```
在搜索栏搜索Windows，点击“启动或关闭Windows功能”。

勾上适用于windows的linux子系统

完成后需要重启电脑

打开Microsoft Store，商店内直接搜索Ubuntu, 注意千万别装Ubuntu20.04

选择第一个下载安装

下载完成后，在桌面搜索栏搜索“Ubuntu”并打开

会报错。

以管理员身份打开PowerShell或CMD，依次执行：
    wsl --update
    重启系统
    
    # 启用适用于Linux的Windows子系统
    dism.exe  /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

    # 启用虚拟机平台

    dism.exe  /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

    # 设置WSL2为默认版本
    wsl --set-default-version 2

在桌面搜索栏搜Ubuntu，此次提示正在安装，过一会会提示设置UNIX username和密码，成功。
    
```

```
错误地将C:\Users\liang\AppData\Local\Packages\ 删除解决办法：
https://blog.csdn.net/m0_69593912/article/details/143580502
```


- 安装图形化界面及支持远程桌面连接
```
安装必要组件（在WSL终端中执行）：

sudo apt update && sudo apt upgrade -y
sudo apt install xorg xfce4 xrdp -y


xorg：基础显示服务
xfce4：轻量级桌面环境
xrdp：远程桌面协议服务
配置Xrdp服务：

sudo echo xfce4-session > ~/.xsession  # 设置默认会话为Xfce4

修改xrdp端口为3390
    sudo vim /etc/xrdp/xrdp.ini

sudo service xrdp restart              # 重启服务 

linux桌面黑屏的问题：
    一、修改XRDP启动脚本（推荐方案）
    在Ubuntu服务器上操作：
        sudo vim /etc/xrdp/startwm.sh 
    在文件顶部添加以下内容：
        unset DBUS_SESSION_BUS_ADDRESS
        unset XDG_RUNTIME_DIR 
        . $HOME/.profile
    保存后重启xrdp服务：
        sudo systemctl restart xrdp 


从Windows连接图形界面：目前没啥用
    打开Windows “远程桌面连接” 应用
    输入地址：localhost:3390
    登录WSL的用户名和密码
    成功后将进入Ubuntu的Xfce4桌面环境
    远程连接可看到桌面

```

### SSH连接WSL
```
# 安装OpenSSH服务
sudo apt update && sudo apt install openssh-server -y

# 修改SSH配置文件
需要修改/取消注释的核心配置：
    Port 22（默认端口，若冲突可改为其他端口如2222）
    PasswordAuthentication yes（允许密码登录，新手推荐先开启）
    PermitRootLogin no（禁用root直接登录，可选但更安全）
保存后重启SSH服务：
    sudo service ssh restart

测试连接
powershell中输入 ssh liangjin@127.0.0.1
press yes, then 输入密码，连接成功

```
