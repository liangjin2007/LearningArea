# 练习UE5中如何制作布料

- 布料制作工具 VRoid Studio 1.29
- UE5.4 Chaos Cloth插件 https://www.youtube.com/watch?v=QnHOjRDQtfo
- KawaiiPhysics插件 https://pafuhana1213.hatenablog.com/entry/2019/07/26/171046
- VRM格式导入插件 https://www.3dcat.live/share/post-id-226/

## 下载Cloth.7z 
Learn https://dev.epicgames.com/community/unreal-engine/learning?

从UE官网->学习->学习资料库 搜索"Cloth" 找到“Panel Cloth Example Files (5.4)”， 下载 UE-5-4-ChaosCloth.7z （ https://d1iv7db44yhgxn.cloudfront.net/post-static-files/UE-5-4-ChaosCloth.7z ）

重命名为Cloth.7z， 解压，以这个项目作为起始项目。

以这个项目作为基础。

## 下载VRM4U插件

从https://github.com/ruyo/VRM4U/releases下载VRM4U插件，放到Cloth文件夹的Plugins目录中，修改Cloth.uproject，添加VRM4U插件
```
{
  "Name": "VRM4U",
  "Enabled": true,
  "SupportedTargetPlatforms": [
    "Win64"
  ]
}
```
