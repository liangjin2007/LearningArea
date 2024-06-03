# C#文档 https://learn.microsoft.com/zh-cn/dotnet/csharp/tour-of-csharp/

## 基础
```
1.程序结构
Main方法 class A { static void Main(string[] args){}}
顶级语句：

  类似于python。一个应用程序只能有一个入口点。 一个项目只能有一个包含顶级语句的文件。 在项目中的多个文件中放置顶级语句会导致以下编译器错误：
  在具有顶级语句的项目中，不能使用 -main 编译器选项来选择入口点，即使该项目具有一个或多个 Main 方法。

  仅能有一个顶级文件
  没有其他入口
  using指令必须在开头
  全局命名空间： 有一个隐藏的全局命名空间
  args
  await:调用异步方法
    Console.WriteLine("Hello ");
    await Task.Delay(5000);
    Console.WriteLine("World");
  进程的退出代码
    return 0;
  隐式入口点方法：
    await 和 return	static async Task<int> Main(string[] args)
    await	static async Task Main(string[] args)
    return	static int Main(string[] args)
    否 await 或 return	static void Main(string[] args)

2.类型系统
  在变量声明中指定类型
    // Declaration only:
    float temperature;
    string name;
    MyClass myClass;
    
    // Declaration with initializers (four examples):
    char firstLetter = 'C';
    var limit = 3;
    int[] source = { 0, 1, 2, 3, 4, 5 };
    var query = from item in source
                where item <= limit
                select item;
      bool不能转为int
  内置类型
    bool, int, float, string, char, object
  自定义类型
    struct, class, interface, enum, record
  通用类型系统
    支持继承原则
    每种类型被定义为值类型或引用类型。struct定义的类型为值类型，所有内置数据类型都是structs。使用class定义的类型是引用类型。
    值类型已密封，不能继承。


3.面向对象的编程
4.功能技术
5.异常和错误
6.编码样式
7.教程
```

## C#中的新功能
## 教程
## 语言集成查询（LINQ）
## 异步编程
## C#概念
## C#编程指南
