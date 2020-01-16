# Maya5编程全攻略

## Maya MEL
类c语言
变量：类型，数组，vector，boolean， 字符串， int, float
操作符：关系，逻辑，算数
控制:条件，循环
脚本：存储脚本，自定义脚本目录 MAYA_SCRIPT_PATH环境变量

模式:创建，查询，编辑
节点：

获取顶点世界坐标位置：
MEL
pointPosition Louise_Anim_with_Marker_center:Louise_Anim_with_Marker_left:Louise_Anim_With_Marker:Louise.vtx[200];
Python cmds.pointPosition("Louise_Anim_with_Marker_center:Louise_Anim_with_Marker_left:Louise_Anim_With_Marker:Louise.vtx[116]");
获取局部坐标位置
getAttr  Louise_Anim_with_Marker_center:Louise_Anim_with_Marker_left:Louise_Anim_With_Marker:Louise.vtx[200]

## Maya SDK
博客 https://blog.csdn.net/whwst/article/details/81604853
https://www.autodesk.com/developer-network/platform-technologies/maya


