## c++ 11
- https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP11.md

## c++ 14
- https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP14.md


## c++ 17
- https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP17.md


## c++ 20
- https://github.com/AnthonyCalandra/modern-cpp-features/blob/master/CPP20.md

## c++ 知识
- 重载new
```
class C
{
public:
        void* operator new(size_t){ return ::malloc(size_t); }
        // 第一个参数必须是size_t, 额外加一些参数
        void* operator new(size_t, const char*handle, const char* file, const char* line) { }
        void* operator new[](size_t, const char*handle, const char* file, const char* line){ }
};
```

- Allocator
```

```

- 线程锁  
```

```

## c++ 新知识
```
5. C++特性
5.1
// 如何用一行生成[0, 1, ..., n]
std::vector<int> keys(n);
std::generate(keys.begin(), keys.end(), [n = 0]() mutable { return n++; }); // 外部都不需要定义int n
or
std::iota(keys.begin(), keys.end(), 0);

5.2
[[nodiscard]] c++17
        [[nodiscard]]是一个属性，用于指示函数的返回值应该被使用而不是被忽略。如果开发者忽略了这样的返回值，编译器可能会发出警告。
        当一个函数被标记为[[nodiscard]]时，它意味着这个函数的返回值是有意义的，并且程序员在使用这个函数时应该处理这个返回值。这通常用于那些返回错误代码、计算结果或其他有用信息的函数。
        下面是一个简单的例子：
        [[nodiscard]] int calculateSum(int a, int b) {
            return a + b;
        }
        void doSomething() {
            calculateSum(1, 2); // 这里编译器可能会发出警告，因为忽略了返回值
        }
        int main() {
            int result = calculateSum(1, 2); // 这是正确的使用方式，因为返回值被赋值给了一个变量
            return 0;
        }

5.3 tinyply:
        std::ifstream f(file_path, std::ios::binary);
        std::unique_ptr<std::istream> file_stream;
        if (f.fail()) {
            throw std::runtime_error("Failed to open file: " + file_path.string());
        }
        // preload
        std::vector<uint8_t> buffer(std::istreambuf_iterator<char>(f), {});
        file_stream = std::make_unique<std::stringstream>(std::string(buffer.begin(), buffer.end()));
        //
        tinyply::PlyFile file;
        std::shared_ptr<tinyply::PlyData> vertices, normals, colors;
        file.parse_header(*file_stream);
        try {
            vertices = file.request_properties_from_element("vertex", {"x", "y", "z"});
        } catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }
        try {
            normals = file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
        } catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }
    
        try {
            colors = file.request_properties_from_element("vertex", {"red", "green", "blue"});
        } catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }
    
        file.read(*ply_stream_buffer);
    
        PointCloud point_cloud;
        if (vertices) {
            std::cout << "\tRead " << vertices->count << " total vertices " << std::endl;
            try {
                point_cloud._points.resize(vertices->count);
                std::memcpy(point_cloud._points.data(), vertices->buffer.get(), vertices->buffer.size_bytes());
            } catch (const std::exception& e) {
                std::cerr << "tinyply exception: " << e.what() << std::endl;
            }
        } else {
            std::cerr << "Error: vertices not found" << std::endl;
            exit(0);
        }



5.4 std::future<void>
但是我们想要从线程中返回异步任务结果，一般需要依靠全局变量；从安全角度看，有些不妥；为此C++11提供了std::future类模板，future对象提供访问异步操作结果的机制，很轻松解决从异步任务中返回结果。
            futures.push_back(std::async(
            std::launch::async, [resolution](const std::filesystem::path& file_path, const Image* image, CameraInfo* camera_info) {
                // Make a copy of the image object to avoid accessing the shared resource

                auto [img_data, width, height, channels] = read_image(file_path / image->_name, resolution);
                camera_info->_img_w = width;
                camera_info->_img_h = height;
                camera_info->_channels = channels;
                camera_info->_img_data = img_data;

                camera_info->_R = qvec2rotmat(image->_qvec).transpose();
                camera_info->_T = image->_tvec;

                camera_info->_image_name = image->_name;
                camera_info->_image_path = file_path / image->_name;

                ...
            },
            file_path, image, camera_infos.data() + image_ID));
            for (auto& f : futures) {
               f.get(); // Wait for this task to complete
            }




5.5 多个返回值 std::tuple<unsigned char*, int, int, int> read_image(std::filesystem::path image_path, int resolution)
          std::tuple<unsigned char*, int, int, int> read_image(std::filesystem::path image_path, int resolution) {
              int width, height, channels;
              unsigned char* img = stbi_load(image_path.string().c_str(), &width, &height, &channels, 0);
              ...
              return {img, width, height, channels};
          }
          auto [img_data, width, height, channels] = read_image(file_path / image->_name, resolution);



5.6 浮点数中使用单引号
const float image_mpixels = cam0._img_w * cam0._img_h / 1'000'000.0f;


5.7 json
std::vector<nlohmann::json> json_data;
...
nlohmann::json camera_entry = {
        {"id", id},
        {"img_name", cam._image_name},
        {"width", cam._width},
        {"height", cam._height},
        {"position", std::vector<float>(pos.data(), pos.data() + pos.size())},
        {"rotation", serializable_array_2d},
        {"fy", fov2focal(cam._fov_y, cam._height)},
        {"fx", fov2focal(cam._fov_x, cam._width)}};

json_data.erase(std::remove_if(json_data.begin(), json_data.end(),
                               [](const nlohmann::json& entry) { return entry.is_null(); }),
                json_data.end());
nlohmann::json json = json_data;
std::ofstream file(file_path.string());
if (file.is_open())
{
    file << json.dump(4); // Write the JSON data with indentation of 4 spaces
    file.close();
}


5.8 clock
auto start_time = std::chrono::steady_clock::now();


5.9. accumulate / sum
std::accumulate(_rate_of_change_buffer.begin(), _rate_of_change_buffer.end(), 0.f);

5.10. random indices permutation
std::shuffle(indices.begin(), indices.end(), std::default_random_engine());
std::reverse(indices.begin(), indices.end());

```
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
```
c++编译过程： 预编译 -> 编译 -> 汇编 ->链接
  编译预处理。 对#开头的指令和特殊符号进行处理
    宏定义指令#define
    条件编译指令#ifdef, #ifndef, #else, #elif, #endif
    头文件包含指令#include
    特殊符号： __LINE__, __FILE__
  编译优化阶段：
    此法分析和语法分析，翻译为中间代码表示或汇编代码
    优化处理：对中间代码优化，及针对目标代码的生成而进行的优化
  汇编：
    将汇编语言代码翻译成目标机器指令的过程， 得到目标文件。目标文件由段组成，包括代码段和数据段。目标文件有：可重定位文件、共享的目标文件、可执行文件。
c++的链接过程：
  将有关的目标文件彼此相连接，链接处理可分为两种：
  静态链接： 函数代码会从静态链接库中拷贝到最终的可执行程序中。
  动态链接： 在可执行文件被执行时将动态链接库中的全部内容映射到运行时相应进程的虚地址空间。


数组：
  二分查找: 循环不变量原则
  移除元素： 快慢指针
  有序数组的平法： 双指针法
  长度最小的子数组： 滑动窗口（也是一种双指针法）
  螺旋矩阵II: 循环不变量原则

链表：
  移除链表元素： dummyHead
  设计链表: dummyHead, size
  翻转链表： prev, cur
  两两交换链表中的节点： dymmyHead, offset, prev, cur
  删除链表的倒数第N个节点: dummyHead, 双指针fast, slow, 及倒数n的处理方式。
  链表相交：获取size， 尾部对齐
  环形链表II: 快慢指针， 快指针每次走2步，慢指针每次走1步， 会在环内相遇；然后需要一定的数学运算（理解起来需要一定的技巧）

哈希表：
  两数之和
  有效的字母异位词： counter[s[i]-'a']++, counter[t[i]-'a']--;
  两个数组的交集： 两个unordered_set
  快乐数：
  两数之和
  四数相加II
  赎金信
  三数之和： 排序， 循环nums[i], 去重。
  四数之和： 

双指针法：
  
栈与队列：
  std::stack<>
  std::queue<>
  用栈实现队列
  用队列实现栈

回溯算法：
  


贪心算法：
  步骤：
    将问题分解为若干个子问题
    找出适合的贪心策略
    求解每一个子问题的最优解
    将局部最优解堆叠成全局最优解
  分发饼干

动态规划问题：
  如果某一问题有很多重叠子问题，使用动态规划是最有效的。
  每一个状态一定是由上一个状态推导出来的。
  动态规划中dp[j]是由dp[j-weight[i]]推导出来的，然后取max(dp[j], dp[j - weight[i]] + value[i])。
  解题步骤：
    确定dp数组（dp table）以及下标的含义
    确定递推公式
    dp数组如何初始化
    确定遍历顺序
    举例推导dp数组
  动态规划应该如何debug
    找问题的最好方式就是把dp数组打印出来，看看究竟是不是按照自己思路推导的！
    做动规的题目，写代码之前一定要把状态转移在dp数组的上具体情况模拟一遍，心中有数，确定最后推出的是想要的结果。
    这道题目我举例推导状态转移公式了么？
    我打印dp数组的日志了么？
    打印出来了dp数组和我想的一样么？
  1. 斐波那契数: 递归函数
  2. 爬楼梯： 实际还是斐波那契数列， F(n) = F(n-1) + F(n-2); 不过用递归会超时，需要用动态规划法设计成一个表。
  3. 使用最小花费爬楼梯： xxx dpt(cost.size()+1); dpt[0] = 0; dpt[1] = 0; dpt[i] = min(dpt[i-1] + cost[i-1], dpt[i-2] + cost[i-2]);
  4.不同路径 : TODO
  5.不同路径II : TODO
  6.整数拆分 


```

## 00.代码随想录-最强八股文-第3版-无密版本.pdf
- 关键字与运算符：
```
  指针与引用：
    变量： 命了名的对象
    指针存放某个对象的地址，其本身就是变量。本身就有地址。
    引用是变量的别名，从一而终，不可变，必须初始化
    不存在指向空值的引用，但是存在指向空值的指针
  const关键字
    指不能改变，是只读变量，必须在定义的时候就给它赋初值
    常量指针： 强调指针对其所指对象的不可改变性。靠近变量类型。int a; const int* p = &a; 或者int const* p = &a;  *p = 9; // 错误
    指针常量： 强调只针对不可改变性。靠近变量名。int* const p = &a;  p = &temp; // 错误
  define
    只是简单的字符串替换，没有类型检查。
    是在编译的预处理阶段起作用。
    可以用来防止头文件重复引用。
    不分配内存，给出的是立即数，有多少次使用就进行多少次替换
  typedef
    有对应的数据类型，
    是在编译、运行的时候起作用。
    在静态存储区中分配空间，在程序运行过程中内存中只有一个拷贝。
  inline
    先将内联函数编译完成生成了函数体直接插入被调用的地方，减少了普通函数的压栈、跳转和返回的操作，没有普通函数调用时的额外开销。
    内联函数是一种特殊的函数，会进行类型检查。
    对编译器的一种请求，编译器有可能拒绝这种请求。
  C++中inline编译限制：
    不能存在任何形式的循环语句。
    不能存在过多的条件判断语句。
    函数体不能过于庞大。
    内联函数声明必须在调用语句之前。
  override
    重写方法的参数列表、返回值、所抛出的异常与被重写方法一致。
    被重写的方法不能为private
    静态方法不能被重写为非静态的方法。
    重写方法的访问修饰符一定要大于被重写方法的访问修饰符 public > protected > default > private
  overload
    ...方法名相同，而参数形式不同

  new和malloc
    new失败时抛出bac_alloc异常，不会返回NULL；malloc分配内存失败时返回NULL。
    new无需指定内存大小
    operator new/operator delete可以被重载，而malloc/free不允许重载。
    new/delete会调用对象的构造函数和析构函数，而malloc不会。
    malloc/free是标准库函数，而new/delete是运算符。
    new封装了malloc
  constexpr
    const表示只读的语义
    constexpr表示常量的语义。
    constexpr只能定义编译器常量，而const 可以定义编译器常量，也可以定义运行期常量。
    定义了constexpr -> 顺带标记为了const。
  constexpr变量
    constexpr int n = 20;
    constexpr int m = n + 1;
    static constexpr int MOD = 1000000007;
    相比宏来说，没有额外的开销，但更安全可靠。
    编译器可以在编译期对constexpr的diamagnetic进行优化，提高效率。
  constexpr函数
    函数的返回类型和所有形参类型都市字面值类型，函数体有且只有一条return语句。constexpr int new() { return 42; }
  constexpr构造函数
  mutable
    类的常量函数中需要修改与类的状态无关的数据成员。
  volatile
    表示该变量随时可能发生变化，与该变量有关的运算，不要进行编译优化，会从内存中重新装载内容，而不是直接从寄存期拷贝内容。
    保证对特殊地址的稳定访问
    比如for(volatile int i = 0; i < 100000; i++); // 它会执行，不会被优化掉
  extern
    声明外部变量，在函数或文件外部定义的全局变量。
  static
    实现多个对象之间的数据共享+隐藏。
  前置++和后置++
    前置++返回引用。
    后置++返回旧值。 不是引用。
    const self operator++(int){
      self tmp = *this;
      ++*this;
      return tmp;
    }
    self& operator++(){
      node = (linktype)((node).next);
      return *this;
    }
    最好使用前置++。 因为它不会创建临时对象。

std::atomic
    a++和int a = b;是否是线程安全的。
    从汇编指令来看对应三条指令， 首先将变量a对应的内存值搬运到某个寄存器(如eax)，然后将寄存器中的值自增1， 再将该寄存器中的值搬运回a代表的内存中。
```

- C++三大特性
```
访问权限：
  私有成员不能被派生类访问。
  公有和保护成员被派生类访问。

继承：
  实现集成
  接口继承
  可视集成

封装：
  数据和代码捆绑再一起，避免外界干扰和不确定性访问。
  把客观事物封装成抽象的类，并且类可以把自己的数据和方法只让可信的类或者对象操作。对不可信的进行信息隐藏。

多态：
  同一事物表现出不同事物的能力。
  重载实现编译时多态。
  虚函数实现运行时多态。
  实现方式：
    override覆盖，子类重新定义父类的虚函数的做法。
    overload重载： 指允许存在多个同名函数，这些函数的参数表不同。
虚函数：
  虚函数表工作。
  虚表指针指向派生类的虚函数表。
  1.虚函数是动态绑定的
  2.多态
    调用函数的对象必须是指针或引用。
    被调用的函数必须是虚函数，且完成了虚函数的重写。
  3.动态绑定
    指针和引用能够找到实际类的对应函数，而不是执行定义类的函数。
  4.构造函数不能是虚函数。
    构造函数中调用虚函数，实际执行的是父类的对应函数。因为自己还没构造好，多态是被禁掉的。
  5.虚函数的工作方式
    每个类一个虚函数表。
  6.析构函数可以是虚函数，而且再一个复杂类结构中，这往往是必须的。
  7.将一个函数定义为纯虚函数。
    将一个类定义为抽象类，不能实例化对象；纯虚函数通常没有定义体，但也完全可以拥有。
  8.inline, static, constructor三种函数都不能带有virtual 关键字。
    virtual 函数一定要通过对象来调用，有隐藏的this指针，实例相关。
  9.析构函数可以是纯虚的。
    必须有定义体，因为析构函数的调用是在子类中隐含的。
  10.派生类的override虚函数定义必须和父类完全一致。
  11.虚继承。
  为什么需要虚继承
    用来解决多继承的二义性。
空类：
    空类的大小是1个字节。为了确保两个对象的地址不同。
    带虚函数的类不是空类，sizeof是4（32位机器上）。在64位系统上是8（64位机器上）。
    class A{}; class B: public virtual A{}; B不是空类，B包含指向虚基类的指针，所以sizeof是4或者8.
    多重继承 class Child : public Father1, public Father2{}; sizeof是1.

抽象类与接口的实现
    
```

## 难搞的右值引用 https://blog.csdn.net/qq_43331089/article/details/126486313


