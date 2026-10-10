- [2024]UniHair
  - https://github.com/PAULYZHENG/UniHair， star 78 
  - Gaussian Hair, just for visualization

- [2025]difflocks 
  - https://github.com/Meshcapade/difflocks，190star

- [2025]CHARM
  - 清华/腾讯
  - 动漫头发发卡生成
  - https://github.com/hyz317/CHARM, 53 star

- [2025]Im2Haircut
  - https://github.com/Vanessik/Im2Haircut, 62 star
  - 62 star 

- [2025][sig]Strands2Cards: Automatic Generation of Hair Cards from Strands
  - https://github.com/kenji-tojo/strns2cards

- [2025]TANGLED
  - 无代码 任意视角/风格 + lineart + MultiHair 数据集


- [2026]HairGPT: Strand-as-Language Autoregressive Modeling for Realistic 3D Hairstyle Synthesis
  - Deemos Technology

- [2026]CGHair
  - 多视角，发丝聚类为 hair cards，重建快4×、显存降200×
  - https://humansensinglab.github.io/CGHair/， 无代码
 
- [2026]

## 头发和衣服可实现如下版本
```
输入头发ai 3d资产 + 衣服3d资产

-> 处理头发： 转密度场 -> trace 发丝(CTHair?) -> 转卡片mesh(strns2cards)

-> 处理衣服:  四边形网格化 -> 使用类LightDrape方法（通过骨架对齐和模拟, 用libuipc改进穿插效果）穿到身体上 -> 绑定权重自动生成（https://github.com/rin-23/RobustSkinWeightsTransferCode）
```

## 可模拟身体Setup及模拟器方向
```
Real-Time Simulation of Deformable Tactile Sensors in Robotic Grasping using Graph Neural Networks with Inductive Biases， https://tacgraspnets.github.io/

类似于UE MLDeformer做

```







