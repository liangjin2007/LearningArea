# Latex公式编辑
- 在线公式编辑器 https://www.codecogs.com/latex/eqneditor.php
### 常用Latex语法
#### 上下标
```
\begin{aligned}
\dot{x} & = \sigma(y-x) \\
\dot{y} & = \rho x - y - xz \\
\dot{z} & = -\beta z + xy
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\dot{x}&space;&&space;=&space;\sigma(y-x)&space;\\&space;\dot{y}&space;&&space;=&space;\rho&space;x&space;-&space;y&space;-&space;xz&space;\\&space;\dot{z}&space;&&space;=&space;-\beta&space;z&space;&plus;&space;xy&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\dot{x}&space;&&space;=&space;\sigma(y-x)&space;\\&space;\dot{y}&space;&&space;=&space;\rho&space;x&space;-&space;y&space;-&space;xz&space;\\&space;\dot{z}&space;&&space;=&space;-\beta&space;z&space;&plus;&space;xy&space;\end{aligned}" title="\begin{aligned} \dot{x} & = \sigma(y-x) \\ \dot{y} & = \rho x - y - xz \\ \dot{z} & = -\beta z + xy \end{aligned}" /></a>

#### 求和和积分
```
\begin{aligned}
\sum_{i=1}^{n}x_{i}=\int_{0}^{1}f(x)\mathrm{d}x
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\sum_{i=1}^{n}x_{i}=\int_{0}^{1}f(x)\mathrm{d}x&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\sum_{i=1}^{n}x_{i}=\int_{0}^{1}f(x)\mathrm{d}x&space;\end{aligned}" title="\begin{aligned} \sum_{i=1}^{n}x_{i}=\int_{0}^{1}f(x)\mathrm{d}x \end{aligned}" /></a>

```
\begin{aligned}
\sum_{ 1\leqslant i\leq n \atop 1\leqslant j\leq n }a_{ij}
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\sum_{&space;1\leqslant&space;i\leq&space;n&space;\atop&space;1\leqslant&space;j\leq&space;n&space;}a_{ij}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\sum_{&space;1\leqslant&space;i\leq&space;n&space;\atop&space;1\leqslant&space;j\leq&space;n&space;}a_{ij}&space;\end{aligned}" title="\begin{aligned} \sum_{ 1\leqslant i\leq n \atop 1\leqslant j\leq n }a_{ij} \end{aligned}" /></a>

#### 极限
```
\begin{aligned}
\lim_{n \to \infty }\sin x_{n}=0
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\lim_{n&space;\to&space;\infty&space;}\sin&space;x_{n}=0&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\lim_{n&space;\to&space;\infty&space;}\sin&space;x_{n}=0&space;\end{aligned}" title="\begin{aligned} \lim_{n \to \infty }\sin x_{n}=0 \end{aligned}" /></a>

#### 根式
```
\begin{aligned}
x=\sqrt[m]{1+x^{p}}
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;x=\sqrt[m]{1&plus;x^{p}}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;x=\sqrt[m]{1&plus;x^{p}}&space;\end{aligned}" title="\begin{aligned} x=\sqrt[m]{1+x^{p}} \end{aligned}" /></a>

#### 绝对值
```
\begin{aligned}
\left | a+b \right |=\coprod_{m}^{n} \frac{e}{f}
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\left&space;|&space;a&plus;b&space;\right&space;|=\coprod_{m}^{n}&space;\frac{e}{f}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\left&space;|&space;a&plus;b&space;\right&space;|=\coprod_{m}^{n}&space;\frac{e}{f}&space;\end{aligned}" title="\begin{aligned} \left | a+b \right |=\coprod_{m}^{n} \frac{e}{f} \end{aligned}" /></a>

#### 矩阵
```
\begin{aligned}
\begin{pmatrix} 1 & 3 & 5 \\
2 & 4 & 6
\end{pmatrix}
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\begin{pmatrix}&space;1&space;&&space;3&space;&&space;5&space;\\&space;2&space;&&space;4&space;&&space;6&space;\end{pmatrix}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\begin{pmatrix}&space;1&space;&&space;3&space;&&space;5&space;\\&space;2&space;&&space;4&space;&&space;6&space;\end{pmatrix}&space;\end{aligned}" title="\begin{aligned} \begin{pmatrix} 1 & 3 & 5 \\ 2 & 4 & 6 \end{pmatrix} \end{aligned}" /></a>

#### 花括号
```
\begin{aligned}
\overbrace{a+b+\cdots +y+z}^{26}_{=\alpha +\beta}
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\overbrace{a&plus;b&plus;\cdots&space;&plus;y&plus;z}^{26}_{=\alpha&space;&plus;\beta}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\overbrace{a&plus;b&plus;\cdots&space;&plus;y&plus;z}^{26}_{=\alpha&space;&plus;\beta}&space;\end{aligned}" title="\begin{aligned} \overbrace{a+b+\cdots +y+z}^{26}_{=\alpha +\beta} \end{aligned}" /></a>

```
\begin{aligned}
a+\underbrace{b+\cdots +y}_{24}+z 
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;a&plus;\underbrace{b&plus;\cdots&space;&plus;y}_{24}&plus;z&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;a&plus;\underbrace{b&plus;\cdots&space;&plus;y}_{24}&plus;z&space;\end{aligned}" title="\begin{aligned} a+\underbrace{b+\cdots +y}_{24}+z \end{aligned}" /></a>

#### 堆砌
```
\begin{aligned}
y\stackrel{\rm def}{=} f(x) \stackrel{x\rightarrow 0}{\rightarrow} A
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;y\stackrel{\rm&space;def}{=}&space;f(x)&space;\stackrel{x\rightarrow&space;0}{\rightarrow}&space;A&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;y\stackrel{\rm&space;def}{=}&space;f(x)&space;\stackrel{x\rightarrow&space;0}{\rightarrow}&space;A&space;\end{aligned}" title="\begin{aligned} y\stackrel{\rm def}{=} f(x) \stackrel{x\rightarrow 0}{\rightarrow} A \end{aligned}" /></a>

#### 乘/除/点击
```
\begin{aligned}
a \cdot b \\
a  \times b \\
a   \div   b
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;a&space;\cdot&space;b&space;\\&space;a&space;\times&space;b&space;\\&space;a&space;\div&space;b&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;a&space;\cdot&space;b&space;\\&space;a&space;\times&space;b&space;\\&space;a&space;\div&space;b&space;\end{aligned}" title="\begin{aligned} a \cdot b \\ a \times b \\ a \div b \end{aligned}" /></a>

#### 连乘/连加
```
\begin{aligned}
\prod  _{a}^{b} \\
\prod \limits_{i=1}^{n} \\
\sum _{a}^{b} \\
\sum \limits_{i=1} ^{n}
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\prod&space;_{a}^{b}&space;\\&space;\prod&space;\limits_{i=1}^{n}&space;\\&space;\sum&space;_{a}^{b}&space;\\&space;\sum&space;\limits_{i=1}&space;^{n}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\prod&space;_{a}^{b}&space;\\&space;\prod&space;\limits_{i=1}^{n}&space;\\&space;\sum&space;_{a}^{b}&space;\\&space;\sum&space;\limits_{i=1}&space;^{n}&space;\end{aligned}" title="\begin{aligned} \prod _{a}^{b} \\ \prod \limits_{i=1}^{n} \\ \sum _{a}^{b} \\ \sum \limits_{i=1} ^{n} \end{aligned}" /></a>

#### 大于等于 小于等于 不等于
```
\begin{aligned}
a \geq b \\
a \leq b \\
a \neq b
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;a&space;\geq&space;b&space;\\&space;a&space;\leq&space;b&space;\\&space;a&space;\neq&space;b&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;a&space;\geq&space;b&space;\\&space;a&space;\leq&space;b&space;\\&space;a&space;\neq&space;b&space;\end{aligned}" title="\begin{aligned} a \geq b \\ a \leq b \\ a \neq b \end{aligned}" /></a>

#### 积分，正负无穷
```
\begin{aligned}
\int_a^b \\
\int_{- \infty}^{+ \infty}
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\int_a^b&space;\\&space;\int_{-&space;\infty}^{&plus;&space;\infty}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\int_a^b&space;\\&space;\int_{-&space;\infty}^{&plus;&space;\infty}&space;\end{aligned}" title="\begin{aligned} \int_a^b \\ \int_{- \infty}^{+ \infty} \end{aligned}" /></a>

#### 子集
```
\begin{aligned}
A \subset B \\
A \not \subset B \\
A \subseteq B \\
A \subsetneq B \\
A \subseteqq B \\
A \subsetneqq B \\
A \supset B
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;A&space;\subset&space;B&space;\\&space;A&space;\not&space;\subset&space;B&space;\\&space;A&space;\subseteq&space;B&space;\\&space;A&space;\subsetneq&space;B&space;\\&space;A&space;\subseteqq&space;B&space;\\&space;A&space;\subsetneqq&space;B&space;\\&space;A&space;\supset&space;B&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;A&space;\subset&space;B&space;\\&space;A&space;\not&space;\subset&space;B&space;\\&space;A&space;\subseteq&space;B&space;\\&space;A&space;\subsetneq&space;B&space;\\&space;A&space;\subseteqq&space;B&space;\\&space;A&space;\subsetneqq&space;B&space;\\&space;A&space;\supset&space;B&space;\end{aligned}" title="\begin{aligned} A \subset B \\ A \not \subset B \\ A \subseteq B \\ A \subsetneq B \\ A \subseteqq B \\ A \subsetneqq B \\ A \supset B \end{aligned}" /></a>

#### 上滑线，下滑线
```
\begin{aligned}
\overline {a+b} \\
\underline {a+b}
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\overline&space;{a&plus;b}&space;\\&space;\underline&space;{a&plus;b}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\overline&space;{a&plus;b}&space;\\&space;\underline&space;{a&plus;b}&space;\end{aligned}" title="\begin{aligned} \overline {a+b} \\ \underline {a+b} \end{aligned}" /></a>

#### 矢量
```
\begin{aligned}
\vec {ab} \\
\overrightarrow{ab}
\end{aligned}
```
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{aligned}&space;\vec&space;{ab}&space;\\&space;\overrightarrow{ab}&space;\end{aligned}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}&space;\vec&space;{ab}&space;\\&space;\overrightarrow{ab}&space;\end{aligned}" title="\begin{aligned} \vec {ab} \\ \overrightarrow{ab} \end{aligned}" /></a>
