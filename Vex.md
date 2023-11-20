## Point Wrangle geometry node 会被移除
```
0. snippet
1. 修改point attributes using code
2. 对应于VOP SOP
3. 在每个input geometry point上执行代码片段
4. 能通过attributes和VEX函数 从其他几何访问信息
5. 在Point Wrangle node上按中键可以看error 输出
6. VEX 函数 ch ， ch("param")会evalulate the parameter parm on this node
7. ??
does not use local variables
backtick expressions and $F variables will be evaluated at frame 1 not the current frame.




```
## Attribute Wrangle node


