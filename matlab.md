## C++ Interop with Matlab.

- 类似于用c++ socket向maya发送命令，让maya执行相关命令。

```
第一步 需要在matlab里敲命令 enableservice('AutomationServer', true);

第二步 用matlab c++库向matlab发送命令

vector2matlab(name, std::vector<float> array);

```
