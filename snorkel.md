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

7. 如何理解StableLabel, GoldLabel, GoldLabelKey, Context, Candidate...
7.1.StableLabel靠近用户的打标结果
StableLabel.context_stable_ids
StableLabel.annotator_name打标者名字
StableLabel.split数据分区

7.2.reload_annotator_labels
关键词：GoldLabelKey, GoldLabel

首先，xxxxKey代表的是表中的字段
而GoldLabelKey既是字段又是一张表。GoldLabelKey.name == annotator_name， 从这句可以看出GoldLabelKey记录的是标注者的名字。

GoldLabel表有一个GoldLabel.key字段，存的是GoldLabelKey。
GoldLabel表还有一个字段GoldLabel.candidate存的是candidate.


具体reload_annotator_labels做了什么呢？
对每个StableLabel对象，
   7.2.1.根据StableLabel.context_stable_ids获取Context.stable_id，根据Context.stable_id找到对应的Context对象s
   比如StableLabel对应两个context人名, 会包含两个context.stable_id,就能找到两个context. 
   7.2.2.根据找到的context查找到对应的candidate。如何没找到candidate，创建candidate。
   7.2.3.

7.3.Context
Context.stable_id

7.4.Candidate
Candidate.__argnames__
getattr(Candidate, Candidate.__argnames__[0])对应于别的context对象。
Candidate.get_contexts()

7.5.query多级查询
找对应query.first(), query.all(), query.count()



















