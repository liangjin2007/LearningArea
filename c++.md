## c++ 11
- https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

## c++ 14
- https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP14.md


## c++ 17
- https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP17.md


## c++ 20
- https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP20.md


## Unknown

= default; // you want to use the compiler-generated version of that function, so you don't need to specify a body.

= delete; //  specify that you don't want the compiler to generate that function automatically.

void func() noexcept; // this function don't throw, equal to "void func() noexcept(true);"

noexcept(func) // true, noexcept(expression) return true or false 

explicit xxx;

Copy Constructor

Move Constructor:   ClassName(ClassName &&rhs)

{x, y, z} corresponding to std::initializer_list<T>

variable template parameters and parameter pack expansion feature of C++
```
  template<typename... Args>
  void myFunction(Args... args)
  {
      // ...
      ((std::cout << args << " "), ...); // parameter pack expansion
  }
```

using Base = xxx;
using Base::Base;

## leetcode
- https://programmercarl.com/
- 00.代码随想录-最强八股文-第3版-无密版本.pdf
```
二分查找
哈希表
移除元素： 快慢指针
有序数组的平法： 双指针法
长度最小的子数组： 滑动窗口（也是一种双指针法）

```
