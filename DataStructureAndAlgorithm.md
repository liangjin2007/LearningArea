# Data Structure and Algorithm

# 二叉树


AVL树

# 红黑树： 一棵有n个内结点的红黑树的高度至多为2lg(n+1)

二叉树的一种
[参考链接](https://blog.csdn.net/haidao2009/article/details/8076970)

- 红黑树定义
1. 每个结点或是红色的，或是黑色的
2. 根节点是黑色的
3. 每个叶结点（NIL）是黑色的
4. 如果一个节点是红色的，则它的两个儿子都是黑色的。
5. 对于每个结点，从该结点到其子孙结点的所有路径上包含相同数目的黑色结点。

- 红黑树例子
   ![红黑树例子](https://github.com/liangjin2007/data_liangjin/blob/master/redblacktree.png?raw=true "红黑树例子")

- C++例子
   - std::map

# 哈希表（散列表）
- 哈希表
   - 哈希值函数 hash_function(key)
   - 一个大vector存指针
   - 用来解决hash值冲突的链表
- 哈希表例子
   ![哈希表例子](https://github.com/liangjin2007/data_liangjin/blob/master/hashtable.jpg?raw=true "哈希表例子")

- C++例子
   - std::unordered_map
