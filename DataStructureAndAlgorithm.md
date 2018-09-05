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
# Add Two Numbers By Lists
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


# 字符串处理之——不重复最长字串
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

# 求中位数 Median of Two Sorted Arrays 思路融合两个已排序数组
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

# 
