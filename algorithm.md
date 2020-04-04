## 算法
### 二叉搜索树
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
### 红黑树
一棵有n个内结点的红黑树的高度至多为2lg(n+1)
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
### 哈希表（散列表）
- 哈希表
   - 哈希值函数 hash_function(key)
   - 一个大vector存指针
   - 用来解决hash值冲突的链表
- 哈希表例子
   ![哈希表例子](https://github.com/liangjin2007/data_liangjin/blob/master/hashtable.jpg?raw=true "哈希表例子")

- C++例子
   - std::unordered_map

### Two Sum

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
### Two Sum of Ordered Vector
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

### Two Sum of BST
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
### Add Two Numbers By Lists
```
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* addThreeNumbers(ListNode* l1, ListNode* l2, int& increment){
        if(!l1 && !l2 && increment == 0)
            return 0;
        
        int sum = 0;
        if(l1) sum += l1->val;
        if(l2) sum += l2->val;
        sum += increment;
        
        int digit = sum % 10;
        ListNode *res = new ListNode(digit);
        increment = sum/10;
        
        return res;
    }
    
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        int increment = 0;
        vector<ListNode*> nodes;
        while(l1 || l2 || increment){
            auto n = addThreeNumbers(l1, l2, increment);
            if(!n)
                break;
            nodes.push_back(n);
            if(l1) l1 = l1->next;
            if(l2) l2 = l2->next;
        }
        
        if(nodes.size() > 0){
            for(int i = 0; i < nodes.size()-1; i++)
                nodes[i]->next = nodes[i+1];

            nodes.back()->next = 0;
            
            return nodes[0];
        }
        
        return 0;
    }
};
```
![流程图](https://github.com/liangjin2007/data_liangjin/blob/master/workflow.jpg?raw=true)


### 字符串处理之——不重复最长字串
```
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        vector<int> dict(256, -1);
        int maxLen = 0;
        int start = -1;
        for(int i = 0; i < s.length(); i++){
            int c = s[i];
            if(dict[c] > start)
                start = dict[c];
            dict[c] = i;
            maxLen = max(maxLen, i-start);
        }
        return maxLen;
    }
};
```

### 求中位数 Median of Two Sorted Arrays 思路融合两个已排序数组
```
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        vector<int> merged;
        int size1 = nums1.size(), size2 = nums2.size();
        merged.reserve(size1+size2);
        int i = 0, j = 0;
        while(i < size1 && j < size2){
            if(nums1[i] <= nums2[j]){
                merged.push_back(nums1[i]);
                i++;
            }
            else{
                merged.push_back(nums2[j]);
                j++;
            }
        }
        
        for(int k = i; k < size1; k++){
            merged.push_back(nums1[k]);
        }
        for(int k = j; k < size2; k++){
            merged.push_back(nums2[k]);
        }
        
        int middle = (merged.size()-1)/2;
        if(merged.size() % 2 == 0)
            return 0.5*(merged[middle]+merged[middle+1]);
        return merged[middle]*1.0;
    }
};
```

### 模拟退火算法
对于一个自变量为x的目标函数为f(x)的极小化问题， 设fk = f(xk), fk+1 = f(xk+1)，对x随机采样及用一个概率值决定是否决定用它

### 采样方法
- Monte Carlo Integration 蒙特卡洛积分
  - 第一步需要理解f(x)的MC求积分法
- Sampling and Expected Values
  - 第二步要理解f(x)的期望是与x的分布有关的。首先由x的分布投掷一批N个样本出来， 那么f(x)的期望应该等于1/NSum(f(x))
- Inverse Transform Sampling(CDF)
  - 如何采样一个离散的P(x)
    - 首先计算离散密度函数（直方图）的累计密度函数
    - 一致采样[0,1]
    - 画一条水平线与累计密度函数求交，求的对应的x位置,作为采样输出。逐一采样比较慢，还可以高效生成许多样本
- Ancestral Sampling
  - 更进一步，如果有多个随机变量如何根据joint distribution采样？ 从P(x1)开始采样，然后从P(x2|x1)采样，然后从P(x3|x2)采样，最后从P(x4|x3)采样。

- 从没有归一化的分布函数进行采样
  - Rejection Sampling
    - 目标：从P采样生成样本xi
    - P*(x)是个函数
    - 从proposal density Q(x)采样xi, 然后从[0,cQ(xi)]一致采样得到u，如果u <= P*(x) 接受xi 否则拒绝
  - Importance Sampling
    - 目标：计算函数f(x)的期望值，而不是生成样本
    ```
    - 从proposal density Q(x)采样xi, 估计期望Eq(f(x)) = 1/N Sum(f(xi))
    - 但是我们想要的是Ep(f(x)), 也就是f(x)在分布P下面的期望。
    ```
    - 从proposal density Q(x)采样xi
    - 生成重要性权重 wi = P*(xi)/Q(xi)
    - 计算经验估计Ep(f(x)) = Sum(wi f(xi))/Sum(wi)
    
  - Resampling
    - 可以使用逆变换采样
    
- Markov Chain Monte Carlo
  - f(x)的期望可以这么用MonteCarlo方法求

### 模拟退火补洞算法
