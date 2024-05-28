import json
import os.path
import re

from nltk import word_tokenize as wt
import networkx as nx   # https://networkx.org/

from .find_json import find_from_json
# from find_json import find_from_json


# 将有向图转换为列表
def create_tree_list(json_dir, node, G, type):
    node = node.strip()
    # 检索前对type进行处理
    if type == 'ast':
        type = 'AST'
    elif type == 'cfg':
        type = 'CFG'
    elif type == 'pdg':
        type = 'REACHING_DEF'
    else:
        print("Wrong type")
        return
    # 检索value
    dot_value, dot_label = find_from_json(json_dir, type, int(node))
    # 检索children
    children = list(G.successors(node))
    if len(children) > 0:
        # 对每个child，使用列表生成器生成int类型的child结点的id
        return {"id": int(node), "type": dot_label, "children": [int(child) for child in children], "value": dot_value}
    else:
        return {"id": int(node), "type": dot_label, "value": dot_value}


# 生成ast，写入json
def write_ast(json_dir, save_dot_path, ast_path, type):
    # 读取dot文件中的内容，生成有向图
    try:
        if not os.path.getsize(save_dot_path):
            return 0
        graph = nx.drawing.nx_pydot.read_dot(save_dot_path)


        # if len(graph.nodes) > 180:
        #     return -1


        # a = graph.nodes.data()
        # for node, attributes in a:
        #     label = attributes.get('label')
        #     if label:
        #         basic_word = label.split(')<SUB>')[0].split(',')[-1]
        #         process_word = basic_word.replace('...', '').replace('===', ' ').replace('!==', ' ').replace(
        #             '{', ' ').replace('}', ' ').replace('(', ' ').replace(')', ' ').replace('>', ' ').replace(
        #             '<', ' ').replace('-', ' ').replace('*', ' ').replace('=', ' ').replace('_', ' ').replace(
        #             '.', ' ').replace(',', ' ').replace(':', ' ').replace('#', ' ')
        #         core_word = wt(process_word)
        #         print(f"Node {node}: {core_word}")
        if not graph:
            return 0
        G = nx.DiGraph(graph)
    except:
        return 0
    # 生成ast列表
    # tree_list = [create_tree_list(json_dir, str(node), G, type) for node in G.nodes() if node != '\\n']
    tree_list = []
    for node in G.nodes():
        if node != '\\n':
            create_node = create_tree_list(json_dir, str(node), G, type)
            tree_list.append(create_node)
    # 转为json
    j = json.dumps(tree_list)
    # 写入json
    with open(ast_path, 'w+', encoding='utf-8') as wf:
        wf.write(j)
        wf.write('\n')
    return 1


# print(tree_list)
json_dir = r'D:/Coding/PY_project/任务/数据处理任务/dot和json转树/source_data/Result_Java250mini_high_joern_version/p00002/result_java/s003798551_all/'
# write_ast(json_dir, r"save_dot/0-ast.dot", r'save_ast/a.json', 'ast')

# tree_list[0]["value"] = 'MethodDeclaration'
# tree_list[1]["value"] = 'Modifier'

