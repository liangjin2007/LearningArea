# leetcode
https://github.com/soulmachine/leetcode

https://github.com/qiyuangong/leetcode

**已知一个无序数组 array，元素均为正整数。给定一个目标值 target，输出数组中是否存在若干元素的组合，相加为目标值。**
解读
这一题本质上就是硬币凑整类型的题（换个说法而已）
即：判断一个钱数能否通过当前币值的硬币凑出。（属于硬币凑整里面的简单题）

思路：
发现递推关系：存在target-nums[i]可以表示，则target可以表示
发现边界值：任意nums[i]都可以单独被表示
故可以采用动态规划解决！

复杂度
时间复杂度：O(target * nums.length)
空间复杂度：O(target)

实现

/**
 * @author QL
 * @date 2021年5月11日
 * @description 已知一个无序数组 array，元素均为正整数。
 * 给定一个目标值 target，输出数组中是否存在若干元素的组合，相加为目标值。
 * @thought 
 * @input nums = [3,9,7,17]  target = 11
 * @output false
 */
public class Test01 {
	

	public static boolean process(int[] nums, int target) {
		
		//处理空指针情况
		if(nums == null || nums.length == 0) {
			return false;
		}
		
		//声明动态规划数组
		int[] dp = new int[target+1];
		dp[0] = 1;
		
		//初始化动态规划边界
		for(int i = 0; i < nums.length; i++) {
			if(nums[i] < dp.length) {
				dp[nums[i]] = 1;
			}
		}
		
		//执行动态规划递推
		for(int i = 0; i < dp.length; i++) {
			if(dp[i] == 1) {
				for(int j = 0 ; j < nums.length; j++) {
					if(nums[j] + i < dp.length) {
						dp[nums[j] + i] = 1;
					}
				}
			}
		}
		
		//System.out.println(Arrays.toString(dp));  //测试
		
		//返回结果
		return dp[target] == 1;
		
	}
	
	//测试方法
	public static void main(String[] args) {
		int[] nums = {3,9,7,17};
		System.out.println(process(nums, 11));
	}

}

# Game Coding
https://github.com/CharlesPikachu/Games

# IEEE754
https://en.wikipedia.org/wiki/IEEE_754

# Hash表
https://blog.csdn.net/weixin_43867940/article/details/105775739
```
struct Pair
{
	Pair(int v0 = 0, int v1 = 0) : pt0(v0), pt1(v1)
	{
	}
	int pt0;
	int pt1;
};

struct HashFunc {
	size_t operator()(const StitchPair& key) const {
		return  hash<int>()(key.pt0) ^ hash<int>()(key.pt1);
	}
};

struct EqualFunc {
	bool operator()(const StitchPair& lhs, const StitchPair& rhs) const {
		return (lhs.pt0 == rhs.pt0 && lhs.pt1 == rhs.pt1) || (lhs.pt1 == rhs.pt0 && lhs.pt0 == rhs.pt1);
	}
};
unordered_map<Pair, float, HashFunc, EqualFunc> HashTable;
```

# UML图说明
### https://www.cnblogs.com/duanxz/archive/2012/06/13/2547801.html
- 依赖
  - 虚线箭头表示
  - 在C++中体现为局部变量，方法参数，或者对静态函数的调用
- 泛化
  - 带空心箭头的实线线表示
  - 表示继承
- 实现
  - 空心箭头和虚线表示
  - 例如java中的implements关键字
  - 表示接口和实现的关系
- 关联
  - 实线箭头表示
  - 关联关系是类与类之间的联结，它使一个类知道另一个类的属性和方法。
  - c++ 中，关联关系是通过使用成员变量来实现的。
- 聚合
  - 带空心菱形头表示
  - 聚合关系是关联关系的一种，是强的关联关系
  - 关联关系所涉及的两个类处在同一个层次上，而聚合关系中，两个类处于不同的层次上，一个代表整体，一个代表部分。
  - 仅仅从 Java 或 C++ 语法上是无法分辨的，必须考察所涉及的类之间的逻辑关系。
  - 属于是关联的特殊情况，体现部分-整体关系，是一种弱拥有关系；整体和部分可以有不一样的生命周期；是一种弱关联
- 组合
  - 带实心菱形头的实线表示
  - 它要求普通的聚合关系中代表整体的对象负责代表部分的对象的生命周期。
  - 属于是关联的特殊情况，也体现了体现部分-整体关系，是一种强“拥有关系”；整体与部分有相同的生命周期，是一种强关联

# 设计模式
### 创建型模式
- 工厂模式
```
class Factory
{
  public:
  Product* createProduct(string product_type_name)
}
```
- 抽象工厂模式
创建大的复杂对象
```
class AbstractFactory
{
  public:
  virtual void createProductA()
  virtual void createProductB()
}
class Factory1 public AbstractFactory{}
class Factory2 public AbstractFactory{}
```
- 原型模式
```
class Prototype
{
public:
  virtual Clone()
}
class ConcretePrototype
{
public:
virtual Clone()
}
```
- 构建者模式
```
class Builder
{
  public:
    BuildPartA()
    BuildPartB()
    BuildPartC()
    GetProduct()
}
```
- 单例模式
```
class Singleton
{
public:
  Object *getInstance()
}
```
### 结构型模式
- 组合模式

- 过滤器模式
- 桥接模式
```
class Abstraction
{
public:
virtual operation()
private:
AbstractionImp *pImp;
}

class RefinedAbstraction public Abstraction
{
public:
virtual operation()
}

class AbstractionImp
{
public:
virtual operation()
}
class ConcreteAbstractionImp public AbstractionImp
{
public:
virtual operation()
}
```
- 适配器模式
```
将一个类的接口转换成另一个接口。 Adapter 模式使得原本由于接口不兼容而不能一起工作的那些类可以一起工作。

类模式的Adapter： 采用继承的方式复用Adaptee 的接口。

对象模式的Adapter： 采用组合的方式实现Adaptee 的复用。
```
- 装饰者模式
```
class Component{
virtual operation()
}

class Decorator public Component{
public: 
virtual operation()
private:
vector<Component *> components;
}

class ConcreteDecorator public Decorator{
public:
virtual operation()
AddedBehavior()
}
又称包装器 Wrapper。 把所有功能按正确的顺序串联起来进行控制， 动态地给一个对象添加一些额外的职责。与生成子类相比， 更为灵活。

Decorator 提供了一种给类增加职责的方法，不是通过继承实现的，而是通过组合。当系统需要新功能时，向旧的类中添加新的代码。把每个要装饰的功能放在单独的类中，并让这个类包装所要装饰的对象。因此，当需要执行特殊行为时，客户代码就可以在运行时根据需要有选择地、按顺序地使用装饰功能包装对象了。
```

- 门面模式

- 享元模式

- 代理模式
### 行为型模式
- 责任链模式
- 命令模式
- 解释器模式
- 迭代器模式
- 中介者模式
- 备忘录模式
- 观察者模式
- 状态模式
- 空对象模式
- 策略模式
- 模版模式
- 访问者模式

# 设计模式六大原则
- 单一职责原则
```
定义：不要存在多于一个导致类变更的原因。通俗的说，即一个类只负责一项职责。
问题由来：类T负责两个不同的职责：职责P1，职责P2。当由于职责P1需求发生改变而需要修改类T时，有可能会导致原本运行正常的职责P2功能发生故障。

解决方案：遵循单一职责原则。分别建立两个类T1、T2，使T1完成职责P1功能，T2完成职责P2功能。这样，当修改类T1时，不会使职责P2发生故障风险；同理，当修改T2时，也不会使职责P1发生故障风险
```
- 里氏替换原则
```
定义1：如果对每一个类型为 T1的对象 o1，都有类型为 T2 的对象o2，使得以 T1定义的所有程序 P 在所有的对象 o1 都代换成 o2 时，程序 P 的行为没有发生变化，那么类型 T2 是类型 T1 的子类型。

定义2：所有引用基类的地方必须能透明地使用其子类的对象。

问题由来：有一功能P1，由类A完成。现需要将功能P1进行扩展，扩展后的功能为P，其中P由原有功能P1与新功能P2组成。新功能P由类A的子类B来完成，则子类B在完成新功能P2的同时，有可能会导致原有功能P1发生故障。

解决方案：当使用继承时，遵循里氏替换原则。类B继承类A时，除添加新的方法完成新增功能P2外，尽量不要重写父类A的方法，也尽量不要重载父类A的方法。
```
- 依赖倒置原则
```
定义：高层模块不应该依赖低层模块，二者都应该依赖其抽象；抽象不应该依赖细节；细节应该依赖抽象。
问题由来：类A直接依赖类B，假如要将类A改为依赖类C，则必须通过修改类A的代码来达成。这种场景下，类A一般是高层模块，负责复杂的业务逻辑；类B和类C是低层模块，负责基本的原子操作；假如修改类A，会给程序带来不必要的风险。

解决方案：将类A修改为依赖接口I，类B和类C各自实现接口I，类A通过接口I间接与类B或者类C发生联系，则会大大降低修改类A的几率。

依赖倒置原则基于这样一个事实：相对于细节的多变性，抽象的东西要稳定的多。以抽象为基础搭建起来的架构比以细节为基础搭建起来的架构要稳定的多。在java中，抽象指的是接口或者抽象类，细节就是具体的实现类，使用接口或者抽象类的目的是制定好规范和契约，而不去涉及任何具体的操作，把展现细节的任务交给他们的实现类去完成。

依赖倒置原则的核心思想是面向接口编程
```
- 接口隔离原则
```
定义：客户端不应该依赖它不需要的接口；一个类对另一个类的依赖应该建立在最小的接口上。

问题由来：类A通过接口I依赖类B，类C通过接口I依赖类D，如果接口I对于类A和类B来说不是最小接口，则类B和类D必须去实现他们不需要的方法。

解决方案：将臃肿的接口I拆分为独立的几个接口，类A和类C分别与他们需要的接口建立依赖关系。也就是采用接口隔离原则。
```
- 迪米特法则
```
定义：一个对象应该对其他对象保持最少的了解。
问题由来：类与类之间的关系越密切，耦合度越大，当一个类发生改变时，对另一个类的影响也越大。

解决方案：尽量降低类与类之间的耦合。

自从我们接触编程开始，就知道了软件编程的总的原则：低耦合，高内聚。无论是面向过程编程还是面向对象编程，只有使各个模块之间的耦合尽量的低，才能提高代码的复用率。低耦合的优点不言而喻，但是怎么样编程才能做到低耦合呢？那正是迪米特法则要去完成的。

迪米特法则又叫最少知道原则，最早是在1987年由美国Northeastern University的Ian Holland提出。通俗的来讲，就是一个类对自己依赖的类知道的越少越好。
```
- 开闭原则
```
定义：一个软件实体如类、模块和函数应该对扩展开放，对修改关闭。
问题由来：在软件的生命周期内，因为变化、升级和维护等原因需要对软件原有代码进行修改时，可能会给旧代码中引入错误，也可能会使我们不得不对整个功能进行重构，并且需要原有代码经过重新测试。

解决方案：当软件需要变化时，尽量通过扩展软件实体的行为来实现变化，而不是通过修改已有的代码来实现变化。
```

# 编程语言
- C++结构体 https://www.jianshu.com/p/d63efcd8390f
- C++ std::function
  - typedef std::function<bool(param&)> CallbackFunction;

