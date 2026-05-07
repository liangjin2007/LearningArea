## First reference [AIGC-WSL.md](https://github.com/liangjin2007/LearningArea/blob/master/AIGC-WSL.md) to setup WSL.
```
启动或关闭Windows功能 打开wsl linux subsystem, 打开虚拟机平台

cmd: 
wsl --update
# 启用适用于Linux的Windows子系统
dism.exe  /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

# 启用虚拟机平台
dism.exe  /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# 设置WSL2为默认版本
wsl --set-default-version 2

wsl：
sudo apt update && sudo apt upgrade -y
sudo apt install xorg xfce4 xrdp -y
sudo echo xfce4-session > ~/.xsession
修改xrdp端口为3390 sudo vim /etc/xrdp/xrdp.ini
重启服务 sudo service xrdp restart

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

## WSL上安装c++开发工具
sudo apt install gcc g++ gdb cmake python3-dev

## python bind 11
https://blog.csdn.net/weixin_43953700/article/details/123772022




