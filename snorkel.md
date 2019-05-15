### [Snorkel实践](https://github.com/liangjin2007/data_liangjin/blob/master/Snorkel.png?raw=true)

#### Snorkel APIs

#### Snorkel： 数据预处理（candidate定义、提取，LF定义和运用，train/dev/test数据集生成8:1:1, 生成noisy training labels）
Intro_Tutorial_2.ipynb

#### Snorkel: 生成模型

#### Snorkel: 判别式模型

#### Snorkel概念
1. 自然语言相关的一些概念
Context, Corpus, Scapy, TSV, mention, Document, Sentence, Relation, etc

2. 图像可能要用到的概念 Candidate, session, LF, LF_accuracy, Dependency,
   黄金标签(load_external_labels, load_gold_labels)，
   
   2.1.lf， lf类型, pattern-based lf, 模糊监督distant supervision lf, 把所有lf定义成一个list, 如何快速测试打标函数的准确性, lf衡量指标
   具体细节可查看Intro_Tutorial_2.ipynb

3. 如何处理非二分类的问题
Categorical_Classes.ipynb












