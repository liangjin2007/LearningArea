### [Snorkel实践](https://github.com/liangjin2007/data_liangjin/blob/master/Snorkel.png?raw=true)

#### Snorkel APIs

#### Snorkel： 数据预处理（candidate定义、提取，LF定义和运用，train/dev/test数据集生成8:1:1, 生成noisy training labels）
Intro_Tutorial_2.ipynb

#### Snorkel: 生成模型

#### Snorkel: 判别式模型

#### Snorkel问题
1. 自然语言处理中有哪些概念？
Context, Corpus, Scapy, TSV, mention, Document, Sentence, Relation, etc

2. 图像可能要用到哪些概念？ Candidate, session, LF, LF_accuracy, Dependency,
   黄金标签(load_external_labels, load_gold_labels)，
   
   2.1.lf， lf类型, pattern-based lf, 模糊监督distant supervision lf, 把所有lf定义成一个list, 如何快速测试打标函数的准确性, lf衡量指标
   具体细节可查看Intro_Tutorial_2.ipynb

   test_LF
   
   2.2.如何运用lf？
   labeler = LabelAnnotator(lfs=LFs)
   np.random.seed(1701)
   %time L_train = labeler.apply(split=0)
   L_train
   
   2.3.从labeler获取矩阵csr_matrix
   %time L_train = labeler.load_matrix(session, split=0)
   L_train
   
3. 如何处理非二分类的问题，如何统计质量？
Categorical_Classes.ipynb

4. 人工打标数据是否必需？
是的

5. 人工打标数据用在哪些阶段？
生成模型训练完成后调用gen_model.error_analysis(session, L_dev, L_gold_dev)

6. 如何训练生成模型？













