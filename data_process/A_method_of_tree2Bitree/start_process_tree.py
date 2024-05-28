import json
import copy
import os
import re
import pdb  # c 变运行；q 退出调试；n 执行下一行；s 进入函数；

from tqdm import tqdm

NODE_FIX = '1*NODEFIX'  #'1*NODEFIX'

traveled_node = []
def traverse_java_tree(big_tree, tree, node_json, time=1, idx=1, global_idx=1):
    current_idx = idx
    traveled_node.append(tree['id'])
    if node_json is None:
        node_json = {}
    if 'children' not in tree:   # 如果当前节点没有孩子节点（即是叶子节点），则将该节点的信息存储在 `node_json` 字典中
        node_json['%s%s'%(NODE_FIX, current_idx)] = {
        'node': '%s%s'%(NODE_FIX, current_idx),
        'children': [tree["value"]],
        'parent': None}     # 节点信息包括节点名称 `'node'`、孩子节点列表 `'children'` 和父节点信息 `'parent'`。
    else:  # 如果当前节点有孩子节点，则进入循环
        node_json['%s%s'%(NODE_FIX, current_idx)] = { # 将当前节点的信息存储在 `node_json` 字典中
        'node': '%s%s'%(NODE_FIX, current_idx),
        'children': [], 'parent': None}
        cx = global_idx     # 将全局索引 `global_idx` 赋值给 `cx`
        clean_list = []
        for dot in tree['children']:
            if dot not in traveled_node:
                clean_list.append(dot)
        tree['children'] = clean_list
        if len(tree['children']) == 0:
            node_json['%s%s' % (NODE_FIX, current_idx)] = {
                'node': '%s%s' % (NODE_FIX, current_idx),
                'children': [tree["value"]],
                'parent': None}
            return node_json, global_idx
        for c in tree['children']:      # 遍历当前节点的孩子节点，对每个孩子节点
            idx += 1                    # 递增 `idx` 和 `global_idx` 的值
            global_idx += 1
            # 将该孩子节点的索引存储在父节点的孩子节点列表中
            node_json['%s%s'%(NODE_FIX, current_idx)]['children'].append('%s%s'%(NODE_FIX, global_idx))
        for c in tree['children']:      # 遍历当前节点的孩子节点，对每个孩子节点
            cx += 1                     # 递增 `cx` 的值，递归调用 `traverse_java_tree` 函数
            time += 1
            # 筛选
            # print('c:',c)
            matching_node = find_matching_node(big_tree, c)
            # print('matching_node:', matching_node)
            # for node in big_tree:
            #     if node["id"] == c:
            #         break
            # node_json, global_idx = traverse_java_tree(big_tree, node, node_json, time, cx, global_idx)
            if matching_node:
                # print('node_id:', matching_node["id"])
                # print('time:', time)
                if time >= 600 and 'children' in matching_node:
                    for child in matching_node['children']:
                        if child == c:
                            # print('removing child: ', c)
                            matching_node['children'].remove(c)
                        else:
                            del matching_node['children']
                            break
                node_json, global_idx = traverse_java_tree(big_tree, matching_node, node_json, time, cx, global_idx)
            else: # may have some problems : big_tree[-1]
                node_json, global_idx = traverse_java_tree(big_tree, big_tree[-1], node_json, time, cx, global_idx)
                print('could not find:', c)
                # print('replaced by node:', big_tree[-1].id)
                # print('time:', time)
    return node_json, global_idx


def find_matching_node(big_tree, node_id):
    for node in big_tree:
        if node["id"] == node_id:
            return node
    return None


def split_tree(tree_json, idx_upper, time=1):
    time+=1
    if time >= 200:
        return {}
    # 使用 `copy.deepcopy()` 函数创建 `tree_json_splited` 的副本，以免修改原始树
    tree_json_splited = copy.deepcopy(tree_json)

    if tree_json is None:
        tree_json = {}
    if tree_json_splited is None:
        tree_json_splited = {}

    # 遍历树中的每个节点，对每个节点：
    for k, node in tree_json.items():
        # 检查节点的孩子节点数量是否大于2，如果大于，则将当前节点拆分为两个节点。将新生成的节点添加到 `tree_json_splited` 中
        if len(node['children']) > 2:
            tree_json_splited['%s%s'%(NODE_FIX, idx_upper + 1)] = {'node': 'Tmp', 'children': node['children'][1:], 'parent': k} # idx_upper + 2
            # 更新新节点的孩子节点的父节点为新节点的名称。
            for ch in node['children'][1:]:
                if ch in tree_json_splited:
                    tree_json_splited[ch]['parent'] = '%s%s'%(NODE_FIX, idx_upper + 1)
                # print('ch:',ch)
                # print('%s' % (NODE_FIX))
                # print('%s' % (idx_upper + 1))
                # print('tree_json_splited[ch]["parent"]:',tree_json_splited[ch]['parent'])
            # 更新原节点的孩子节点列表为原节点的第1个孩子节点和新节点的名称。
            tree_json_splited[k]['children'] =  [tree_json_splited[k]['children'][0],'%s%s'%(NODE_FIX, idx_upper + 1)]# idx_upper + 2
            # tree_json_splited[node['children'][0]]['parent'] = '%s%s'%(NODE_FIX, k)
            tree_json_splited[node['children'][0]]['parent'] =  k   # 更新原节点的第1个孩子节点的父节点为原节点的名称。

            idx_upper += 1
    for k, node in tree_json_splited.items():
        if 'children' not in node:
            break
        children = []
        for c in node['children']:
            if c and c.startswith(NODE_FIX):
                children.append(c)
        children_length = len(children)
        # children_length = len([c for c in node['children'] if c.startswith(NODE_FIX)])
        if children_length > 2:
            tree_json_splited = split_tree(tree_json_splited, idx_upper, time+1)
    return tree_json_splited


def _removekey(d, key):
    r = dict(d)
    del r[key]
    return r


def merge_tree(tree_json, time=1):
    time += 1
    if time >= 200:
        return {}
    if tree_json:
        for k, node in tree_json.items():
            children_length = len([c for c in node['children'] if c and c.startswith(NODE_FIX)])
            if children_length == 1:
                del_key = node['children'][0]
                node['children'] = tree_json[node['children'][0]]['children']

                for ch in tree_json[del_key]['children']:
                    if ch and ch.startswith(NODE_FIX) and ch in tree_json:
                        tree_json[ch]['parent'] = k

                tree_json = _removekey(tree_json, del_key)
                break

        for k, node in tree_json.items():
            children_length = len([c for c in node['children'] if c and c.startswith(NODE_FIX)])
            if children_length == 1:
                tree_json = merge_tree(tree_json, time+1)
    return tree_json


def is_folder_exists(folder_path):
    exist = os.path.isdir(folder_path)
    if exist == 1:
        files = os.listdir(folder_path)
        if len(files) == 3:
            print(f"{folder_path} is finished.")
            return 1
        else:
            return 0
    else:
        return 0


def examine_file(path, type):
    # 获取文件夹下的全部文件名
    files = os.listdir(path)
    if len(files)<3:
        return 1
    file = files[0]
    file_ext = os.path.splitext(file)[1]
    filename = os.path.join(path, file)
    if not os.path.getsize(filename):
        print(f"{filename} is empty file!")
        return 1
    return 0


def start(source_path, output_path, temp_tree_path, temp_sorted_tree_path):
    with open(source_path, 'r', encoding='utf-8') as f:
        # print(source_path)
        tree = json.load(f)
        sorted_tree = sorted(tree, key=lambda x: x["id"])
        b_tree = json.dumps(sorted_tree)
        with open(temp_tree_path, 'w') as temp:
            temp.write(b_tree)

        sorted_tree = copy.deepcopy(tree)
        for i, d in enumerate(sorted_tree):
            d["id"] = i
            if 'children' in d:
                children = []
                for child_id in d["children"]:
                    for j, child in enumerate(sorted_tree):  # 使用 sorted_tree 进行索引查找
                        if child["id"] == child_id:
                            children.append(j)
                            break
                d["children"] = children
        b_sorted_tree = json.dumps(sorted_tree)
        with open(temp_sorted_tree_path, 'w') as temp:
            temp.write(b_sorted_tree)

        tree_json = {}
        for i in range(len(sorted_tree)):
            traveled_node.clear()
            tree_json, _ = traverse_java_tree(sorted_tree, sorted_tree[i], tree_json)
            tree_json = split_tree(tree_json, len(tree_json))
            tree_json = merge_tree(tree_json)
            if not tree_json:
                continue
            sorted_keys = sorted(tree_json.keys(), key=get_number)
            new_tree_json = {}
            for sorted_key in sorted_keys:
                new_tree_json[sorted_key] = tree_json[sorted_key]
            tree_dict_str = json.dumps(new_tree_json)
            break
    with open(output_path, 'w') as ast_json_file:
        ast_json_file.write(tree_dict_str)


def get_number(key):
    pattern = r'NODEFIX(\d+)'
    match = re.search(pattern, key)
    if match:
        return int(match.group(1))
    else:
        print('ERROR!')


def joint_result_path(root):
    # 获取文件夹下的全部文件名
    files = os.listdir(root)
    filenames = []
    for file in files:
        # 拼接
        filename = os.path.join(root, file)
        filenames.append(filename)
    return filenames


def joint_root_file_path(result):
    # 获取文件夹下的全部文件名
    files = os.listdir(result)
    filenames = []
    save_nums = []
    for file in files:
        # 拼接
        filename = os.path.join(result, file)
        save_nums.append(file)
        filenames.append(filename)
    return filenames, save_nums


def get_num(s):
    trigger_num = s.split('___')[0].split('__')[1]
    return int(trigger_num)


def get_tree(root_dir, save_root_dir, temp_tree_dir, temp_tree_path, temp_sorted_tree_path):
    os.makedirs(temp_tree_dir, exist_ok=True)
    # 用根目录拼接result目录
    result_dir = joint_result_path(root_dir)
    # print(result_dir)
    # 循环遍历每个文件夹
    for id, r in enumerate(result_dir):
        if not r.endswith('train'):
            continue
        # 获取文件夹的名称，例如p00002
        files = os.listdir(root_dir)
        p_name = files[id]
        # 用result目录拼接json目录
        json_dir, save_num_names = joint_root_file_path(r)

        json_dir = sorted(json_dir, key=get_num)
        save_num_names = sorted(save_num_names, key=get_num)

        # json_dir:['D:\\Coding\\PY_project\\任务\\数据处理任务\\dot和json转树\\output_data\\save_ast\\p00002\\s003798551\\',...]
        for file_id in tqdm(range(len(json_dir))):
            if file_id == 61:
                pass
            # 保存路径拼接
            save_root = os.path.join(save_root_dir, p_name, save_num_names[file_id])
            # # 判断目标文件夹是否存
            # if is_folder_exists(save_root):
            #     continue
            # 检查文件是否为空
            if examine_file(json_dir[file_id], "json"):
                print(f'ERROR in {r}')
                continue
            # 根据save_dot文件查询all生成对应的json文件，放到指定目录
            os.makedirs(save_root, exist_ok=True)
            json_files = []
            all_files = os.listdir(json_dir[file_id])
            for item in all_files:  # all_files:['0-ast.dot.json', '0-cfg.dot.json', '0-pdg.dot.json']
                if item.endswith('_ast.json'):
                    json_files.append(item)     # json_files:['0-ast.dot.json']
            for i in json_files:
                source = os.path.join(json_dir[file_id], i)
                #source = os.path.join(json_dir[file_id], i)
                save = os.path.join(save_root, i)
                #save = os.path.join(save_root, i)
                start(source, save, temp_tree_path, temp_sorted_tree_path)


if __name__ == '__main__':
    file = r'/home/user/PY_Projects/0_MUTI/multilanguage_py/速通/核心代码/2_8/A_method_of_tree2Bitree/save_ast/1-ast.json'
    out = r'/home/user/PY_Projects/0_MUTI/multilanguage_py/速通/核心代码/2_8/A_method_of_tree2Bitree/save_ast/1-ast-Bitree.json'
    # file = r'/media/zhangfanlong/c7e5b1f8-c17b-4f55-b700-bbd6d16c1e04/lmc/codenet/Project_CodeNet/derived/benchmarks/py_tree'
    # out = r'/media/zhangfanlong/c7e5b1f8-c17b-4f55-b700-bbd6d16c1e04/lmc/codenet/Project_CodeNet/derived/benchmarks/python_Bitree'
    temp_tree_path = 'save_ast/tree.json'
    temp_sorted_tree_path = 'save_ast/tree_sorted.json'

    start(file, out, temp_tree_path, temp_sorted_tree_path)