### [Snorkel实践](https://github.com/liangjin2007/data_liangjin/blob/master/Snorkel.png?raw=true)

#### Snorkel APIs
- Candidates如何定义
  - 定义于Sentence中的Span之上
  - Span代表概念类目
  - candidate代表mention, mention是一个关系(span1, span2)
  
- 预处理：
加载文档， 提取candidates，然后把所有数据加载到数据库中。
  - 这一步非常慢，最好做完一次就保存到数据库中（本地或者在线）
  
- 加载candidates
  - cands = session.query(Candidate).filter(Candidate.split == 0).all()
  - 每个candidate长这样 Spouse(Span("Ms Morris", sentence=27839, chars=[0,8], words=[0,1]), Span("Earl", sentence=27839, chars=[80,83], words=[16,16]))

- 获取candidate的parent信息

- 写LF的帮助函数
  - 比如提取两个span之间的文本
  - 比如检查span周围的单词窗口
  - lf_helpers
    - get_left_tokens
    - get_right_tokens
    - get_between_tokens
    - get_text_between
    - get_tagged_text
  - 计算lf度量
    - snorkel.lf_helpers.test_LF
  

#### Snorkel： 提取Spouse关系

#### Snorkel: 生成模型

#### Snorkel: 判别式模型







