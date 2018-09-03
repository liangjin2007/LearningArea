# Data Structure and Algorithm

# 二叉搜索树（BST, Binary Search Tree）
- 二叉搜索树例子
![例子](https://github.com/liangjin2007/data_liangjin/blob/master/BST.jpg?raw=true)

- C++例子
```
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 * TreeNode *tree = ***;
 */
```
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

# Two Sum

```
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        std::unordered_map<int, int> finder;
        for(int i = 0; i < nums.size(); ++i){
            int num = nums[i];
            auto f = finder.find(target-num);
            if(f != finder.end()){
                return vector<int>{f->second, i};
            }else{
                finder.insert({num, i});
            }
        }
    }
};
```
# Two Sum of Ordered Vector
```
class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        int i = 0, j = numbers.size()-1;
        while(i < j){
            int sum = numbers[i]+numbers[j];
            if(sum == target)
                return vector<int>{i+1, j+1};
            else if(sum < target)
                i++;
            else
                j--;
        }
    }
};
```

# Two Sum of BST
```
class Solution {
public:
    void middleOrderTraverse(TreeNode* root, vector<TreeNode*>& orderedValues){
        if(root->left)
            middleOrderTraverse(root->left, orderedValues);
        if(root)
            orderedValues.push_back(root);
        if(root->right)
            middleOrderTraverse(root->right, orderedValues);
    }
    
    bool findTarget(TreeNode* root, int k) {
        vector<TreeNode*> orderedValues;
        middleOrderTraverse(root, orderedValues);
        
        int i = 0, j = orderedValues.size()-1;
        while(i < j){
            int sum = orderedValues[i]->val + orderedValues[j]->val;
            if(sum == k) return true;
            else if(sum < k)i++;
            else j--;
        }
        return false;
    }
};
```
