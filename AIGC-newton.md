- Hello World
```
conda create -n newton python=3.12
conda activate newton

pip install scikit-learn
python.exe -m pip install --no-build-isolation --no-cache-dir "imgui_bundle>=1.92.0"

pip install "newton[examples]"
git clone https://github.com/newton-physics/newton.git

or

cd newton
python.exe -m pip install -e ".[examples]"

python -m newton.examples
```

- Transmission
```
机器人中的传动transmission是否对应某种硬件还是一个抽象概念？
是抽象概念，但它抽象的正是一类真实硬件。二者不矛盾：transmission 是"把促动器的力/位移映射到关节空间"的数学模型，而不同类型的 transmission 对应不同的物理机构。

抽象层面
在 MuJoCo 里，一个 actuator 被拆成两部分：

  actuator（力生成）：gaintype/biastype/dyntype 决定产生多大的标量力 F（电机、肌肉、气动…）。
  transmission（力传递/传动）：trntype/trnid/gear 决定这个标量力如何作用到机构的多个自由度上。
数学上，transmission 就是一个广义力臂 / 雅可比：

  actuator_length  L(q)      # 促动器的"长度"
  moment  m(q) = ∂L/∂q       # 力臂（对每个 DOF 的偏导）
  velocity v = m(q)·q̇
关节广义力  qfrc_actuator = m(q)ᵀ · F_actuator
对直连关节：m = gear（常数，即减速比/杠杆比）。
对肌腱/缆绳：L(q) 是沿滑轮/绕线点走线的长度，m(q) 随姿态变化——这就是"力臂随位形变化"。
对曲柄滑块：m(q) 由连杆几何决定。
所以 transmission 本身不是某一件硬件，而是"任意机构 → 关节空间"的统一描述方式。

硬件层面
每种 transmission 类型背后都对应真实机构：

transmission 类型	对应硬件
joint / jointinparent	电机直驱或减速箱（谐波/RV/行星），gear 即减速比
tendon	腱/缆绳传动（Shadow Hand、Optimus 手、人体肌肉-肌腱）经滑轮/鞘管走线
slider-crank	曲柄连杆 / 丝杠（旋转↔直线）
site	力作用在某个 site 上（如推力点）
body	对刚体施加力/扭矩（如 adhesion）
so3	直接对相对姿态做伺服
同一个 motor actuator 配不同 transmission，就变成完全不同的物理系统：配 joint 是关节电机，配 tendon 就是"拉缆绳的绞盘"。这正是抽象的价值。

与本仓库/计划的关系
Genesis 目前没有 actuator/transmission 抽象：每个关节都可直接控制，解析 MJCF 时把 gear 折算进 dofs_act_gain/dofs_act_bias（genesis/utils/mjcf.py:464-477）。也就是说 Genesis 隐式假设了"常数 gear 的关节传动"。

myofullbody 的 muscle 用的是 tendon transmission：L(q) 沿 site/圆柱绕线变化，m(q)=∂L/∂q 必须每步算——这正是 Genesis 缺的那层。
因此计划里的 Phase 1 本质就是：补上 tendon 这层 transmission 抽象（算 L 和 ∂L/∂q），muscle 则是它的 actuator（FLV + 激活）。
一句话：transmission 是抽象，但它抽象的是齿轮、腱绳、连杆这些真实传动机构；实现 tendon 支持，就是把这层抽象（力臂 ∂L/∂q）补进刚体求解器。
```
