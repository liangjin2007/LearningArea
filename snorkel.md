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
  - cands = session.query(Candidate).filter(Candidate.split == 0).all()
  - 每个candidate长这样 Spouse(Span("Ms Morris", sentence=27839, chars=[0,8], words=[0,1]), Span("Earl", sentence=27839, chars=[80,83], words=[16,16]))

- 获取candidate的parent信息
```
def print_cand(cand):
    print(cand) # Spouse(Span("Ms Morris", sentence=27839, chars=[0,8], words=[0,1]), Span("Earl", sentence=27839, chars=[80,83], words=[16,16]))
    print("id:", cand.id) # 1
    print("type:", cand.type) # spouse
    print("split:", cand.split) # 0
    print("__argnames__:", cand.__argnames__) # (person1, person2)
    print("get_cids():", cand.get_cids()) # (None, None)
    print("candidate[0]:", cand[0])
    print("candidate[0].get_span():", cand[0].get_span())
    print("candidate[1].get_span():", cand[1].get_span())
    print("candidate.person1:", cand.person1)
    print("candidate.person2:", cand.person2)
    # the raw word tokens for the person1 Span
    print("words:", cand.person1.get_attrib_tokens("words"))

    # part of speech tags
    print("pos_tags:", cand.person1.get_attrib_tokens("pos_tags"))

    # named entity recognition tags
    print("ner_tags:", cand.person1.get_attrib_tokens("ner_tags"))

    sentence = cand.get_parent()
    document = sentence.get_parent()

    print("sentence", sentence)
    print("document", document)
```

- 写LF






