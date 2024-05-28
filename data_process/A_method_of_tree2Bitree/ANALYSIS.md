# 需求分析

## 明确需求

文件结构：

根据编号，一个all文件夹对应一个ast、cfg、pdg文件夹

需求：

待处理数据要先做一个按结点编号升序排序；大致格式：{"父节点编号":{"1*NODEFIXn": {"node": "1*NODEFIX1", "children": ["1*NODEFIX2", "1*NODEFIX50"], "parent":"1*NODEFIX1"},...}；结点多与两个就split-tree递归拆除出去，然后merge，最终就得到结果。所以就是将split和merge搞懂然后移植到我这个任务中

## 执行步骤

- [x] 文件路径与批量处理
  - [x] 先解决一个文件
- [x] 结点重新升须排序
- [x] 改java2tree等函数
  - [x] getChildCount改为children，children改为value
  - [x] id要改一下：从0一直标到最后
  - [x] 解决问题：序号不能连续
  - [x] 核心差异：java2tree函数/traverse_java_tree函数。他处理完traverse_java_tree就相当于我当前排好序号的json普通树。tree的children的编号没改过来
  - [x] 解决children的编号与id相等的问题
  - [x] 解决问题：为什么前几个结点不是二叉树？
- [x] 将结果用json.load写入到json文件
  - [x] 文件名获取的问题
  - [x] python文件读写：以适当的命名新建json文件
- [ ] 代码结构优化
  - [ ] 代码简化
  - [ ] 将json、graphviz等基础语法补充到python知识体系
  - [ ] 将ast、cpg等概念补充到深度学习知识体系



