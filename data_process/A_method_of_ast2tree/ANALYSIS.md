# 需求分析

## 明确需求

文件结构：

根据编号，一个all文件夹对应一个ast、cfg、pdg文件夹

需求：

借助json文件中的结点的信息（结点的value），利用ast.dot、cfg.dot、pdg.dot文件的结点的信息（结点的id）和边的信息（结点的children）生成原代码ast（有指定格式）

## 执行步骤

- [ ] 处理dot文件，获取结点的id和结点的children
  - [x] dot文件筛选：只处理带main函数的（发现main函数的都是第一个文件的）
  - [x] dot文件格式化：去掉空格生成新的dot文件
  - [x] dot文件读写：从string中获取
  - [x] dot文件读写：从.dot文件中获取
  - [x] 图的获取：networkx.DiGraph
  - [x] 最终输出格式处理
  - [x] networkx库学习
  - [x] networkx创建图、获取结点id、child、value
  - [ ] 代码优化：遍历、列表生成式等
- [x] 处理json文件，获取结点的value
  - [x] json文件读写
  - [x] 获取结点的value
- [x] 在遍历json文件时，根据结点的id获取结点的value，并写入到output的字典中
  - [x] 遍历中筛选，先选label_value合适的，再看int或out的id合适的，取出该合适结点的value
  - [x] 遍历过程中，将取出的value添加到该结点的字典中
- [x] 将结果用json.load写入到json文件
  - [x] 文件名获取的问题
  - [x] python文件读写：以适当的命名新建json文件
  - [x] 用json.load写入到json文件的单双引号问题
- [ ] 生成CPG文件（又不用了，跑偏了）
  - [x] joern安装
  - [x] 文件导入与生成
  - [ ] 生成全部json的cpg文件
- [ ] 代码结构优化
  - [ ] 代码简化
  - [ ] 将json、graphviz等基础语法补充到python知识体系
  - [ ] 将ast、cpg等概念补充到深度学习知识体系



