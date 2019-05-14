### [Snorkel实践](https://github.com/liangjin2007/data_liangjin/blob/master/Snorkel.png?raw=true)

- Candidates如何定义
  - 定义于Sentence中的Span之上
  - Span代表概念类目
  - candidate代表mention, mention是一个关系(span1, span2)
  - subclass.__mapper_args__['polymorphic_identity']
  - subclass.__argnames__
  
- 预处理：
加载文档， 提取candidates，然后把所有数据加载到数据库中。
  - 这一步非常慢，最好做完一次就保存到数据库中（本地或者在线）
  
- 加载candidates
  - 
  - 每个candidate长这样 Spouse(Span("Ms Morris", sentence=27839, chars=[0,8], words=[0,1]), Span("Earl", sentence=27839, chars=[80,83], words=[16,16]))

