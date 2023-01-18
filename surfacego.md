## No Bootable Device Problem
- 第一步，进入Surface Go UEFI界面（BIOS设置界面），查询序列号
```
首先关机
长按高音量键，按一下电源键进入, 看到Suface标志继续长按， 直到进入Suface UEFI界面。 在PC Information Tab可看到Serial Number
```

- 第二步，搜索 Surface Recovery 映像下载， 找到微软官方下载恢复镜像的页面， https://support.microsoft.com/zh-cn/surface-recovery-image


- 第三步，选择Suface型号，输入序列号， 可以得到两个镜像zip文件，使用迅雷下载
```
下载 Surface 恢复映像
Important: Don't download the files directly to your USB drive.

Surface 恢复映像

下载所需的图像文件并将其保存到 Surface 或其他电脑。该文件将下载为 zip 文件。
产品	下载链接
Surface Go Y/4/64 - Windows 10 Home in S Mode Version 1809	Download image
https://surface.downloads.prss.microsoft.com/dbazure/SurfaceGo_BMR_41_64_2.001.2.zip?t=153f2d79-553f-4d01-9031-0ffdba63376f&e=1674065832&h=7c1ac07525a2b540d4f56bc8339fc201b75df8481d28f37dc567af1588236035

Surface Go Y/4/64 - Windows 10 Home in S Mode Version 1803	Download image
https://surface.downloads.prss.microsoft.com/dbazure/SurfaceGo_BMR_41_64_1.011.2.zip?t=153f2d79-553f-4d01-9031-0ffdba63376f&e=1674065832&h=821b8c253308e17048903dd6133d573efbf3bfa7c1ac7e6ca6485a461b52563d
```

- 第四步，创建 USB 恢复驱动器 
```
如果使用 Surface Hub 2，请参阅 Surface Hub 2S 重置和恢复。

如果使用 Surface Duo，请参阅如果 Surface Duo 无法启动，请进行恢复。

如果使用其他 Surface 设备，以下是创建 USB 恢复驱动器的方法： 

重要: 创建恢复驱动器将清除 U 盘中存储的所有内容。 请确保使用的是空白 U 盘，或者先将 U 盘中的所有重要数据传输到其他存储设备，然后再使用 U 盘来创建恢复驱动器。 

请确保将恢复映像下载到的电脑已打开并接通电源，然后将 USB 恢复驱动器插入 USB 端口。 （尽可能使用 USB 3.0 驱动器。）

在任务栏上的搜索框中，输入“恢复驱动器”，然后从结果中选择“创建恢复驱动器”或“恢复驱动器”。 系统可能会要求你输入管理员密码或确认你的选择。

在“用户帐户控制”框中，选择“是”。

确保清除“将系统文件备份到恢复驱动器”复选框，然后选择“下一步”。

选择你的 U 盘，然后选择“下一步”>“创建”。  需要将一些实用程序复制到恢复驱动器，因此这可能需要几分钟时间。

恢复驱动器准备就绪后，选择“完成”。

双击以前下载的恢复映像 .zip 文件以打开它。

从恢复映像文件夹中选择所有文件，将它们复制到你创建的 USB 恢复驱动器中，然后选择“选择替换目标位置的文件”。

复制完文件后，选择任务栏上的“安全删除硬件并弹出媒体”图标，并拔出 U 盘。

有关如何使用映像的详细信息，请参阅创建和使用 Surface 的 USB 恢复驱动器。

```
