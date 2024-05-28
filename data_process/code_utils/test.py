import json
import logging
import os
import re
import shutil

from tqdm import tqdm


def delete(f_dir):
    code_list = os.listdir(f_dir)
    delete_list = []
    for file in code_list:
        code_path = os.path.join(f_dir, file)
        if len(os.listdir(code_path)) < 3:
            delete_list.append(code_path)
    for code_path in delete_list:
        shutil.rmtree(code_path)
        logging.basicConfig(level=logging.INFO)
        logging.info(f'Trying to delete {code_path}')

f_dir = r'/home/user/PY_Projects/0_MUTI/multilanguage_json/base/data_2_10/mid_output/tree/train'
# delete(f_dir)


def count_Bitree_len(f_path):
    len_80 = 0
    len_80_100 = 0
    lem_100_120 = 0
    len_120_150 = 0
    len_150 = 0
    tree_len_count = 0
    tree_count = 0
    with open(f_path, 'r') as f:
        trees = json.load(f)
    for tree in trees:
        tree_count += 1
        tree_len = len(tree)
        tree_len_count += tree_len
        if tree_len < 80:
            len_80 += 1
        elif tree_len < 100:
            len_80_100 += 1
        elif tree_len < 120:
            lem_100_120 += 1
        elif tree_len < 150:
            len_120_150 += 1
        else:
            len_150 += 1
    print(f'average len:{tree_len_count/tree_count}')
    print(f'<80:{len_80}')
    print(f'80-100:{len_80_100}')
    print(f'100-120:{lem_100_120}')
    print(f'120-150:{len_120_150}')
    print(f'150-:{len_150}')


code = r"'def:\nstr=@absdaf'"
pattern = r'([^\'"])([^\'"]*?)(@)([^\'"]*?)([^\'"])'
# 使用findall()方法找到所有匹配的字符串
matches = re.findall(pattern, code)
# 判断是否存在每对引号之外的内容包含字符'@'
a_flag = 0
# for match in matches:
#     if '@' in match[1] and len(match[0])!= 0 and len(match[2])!= 0:
#         a_flag = 1
#         print(f"存在引号之外的内容包含'@': {match[1]}{match[2]}{match[3]}")

    # if '@' in origin_line or '#' in origin_line:
    #     if 'def' in source_line:
    #         def_matches = re.findall(def_pattern, source_line)
    #         if len(def_matches) >= 2:
    #             pattern = re.compile(r"(?=(" + re.escape(def_string) + r"))")
    #             matches = pattern.finditer(source_line)
    #             def_matches_indexes = []
    #             while True:
    #                 try:
    #                     match = next(matches)
    #                     index = match.start()
    #                     def_matches_indexes.append(index)
    #                 except StopIteration:
    #                     break
    #             match_len = len(def_matches_indexes)
    #             check_success = 0
    #             if match_len >= 2:
    #                 for d_idx, d_match in enumerate(def_matches_indexes):
    #                     first_index = def_matches_indexes[d_idx]
    #                     if d_idx + 1 < match_len:
    #                         next_index = def_matches_indexes[d_idx + 1]
    #                         source_code = source_line[first_index:next_index]
    #                     else:
    #                         source_code = source_line[first_index:]
    #                     origin_line = clean_code(source_code, 'py')
    #                     if py_code_check(origin_line):
    #                         check_success = 1
    #                         print(source_code)
    #                         print(origin_line)
    #                         break
    #             if check_success == 0:
    #                 origin_line = code_summary_pycode_string_repair(origin_line)
    #                 print(origin_line)
    #                 return origin_line
    #         if len(def_matches) <= 1:
    #             origin_line = code_summary_pycode_string_repair(origin_line)
    #             print(origin_line)
    #             return origin_line


def remove_newline(sign, match):
    source_code = match.group(0)
    if sign == 'single':
        code = source_code.replace("'", '')
    elif sign == 'double':
        code = source_code.replace('"', '')
    else:
        code = source_code
    code = code.replace('\n\t', '\n')
    code = code.replace('\n ', '\n')
    code = code.replace('\n', '')
    code = code.replace('\t\t', '\t')
    code = code.replace('\t\t', '\t')
    code = code.replace('\t\t', '\t')
    code = code.replace('\t\t', '\t')
    code = code.replace('\t\t', '\t')
    code = code.replace('\t\t', '\t')
    code = code.replace('  ', ' ')
    code = code.replace('  ', ' ')
    code = code.replace('  ', ' ')
    return code


def process_dot(d_dir, d_save, d_type, d_mn):
    # 获取文件夹下的全部文件名
    files = os.listdir(d_dir)
    # length_file = []
    # for filename in files:
    #     read_filename = os.path.join(d_dir, filename)
    #     file_length = os.path.getsize(read_filename)
    #     length_file.append(file_length)
    # max_len_value = max(length_file)
    # max_len_index = length_file.index(max_len_value)
    # check_filename = files[max_len_index]
    # check_filename_index = check_filename.split('-')[0]
    # if int(check_filename_index) < 2:
    #     read_filename = os.path.join(d_dir, check_filename)
    #     with open(read_filename, "r") as file:
    #         content = file.read()
    #         # if 'io.joern' in content or 'ErrorStatement' in content:
    #         #     return 0
    #     if 'io.joern' not in content and 'ErrorStatement' not in content:
    #         content_without_spaces = content.replace(" ", "")
    #         # 拼接
    #         save_filename = os.path.join(d_save, check_filename)
    #         # 写入到新文件
    #         target_file = open(save_filename, "w")
    #         target_file.write(content_without_spaces)
    #         target_file.write('\n')
    #         target_file.close()
    #         file.close()
    #         return 1
    # print(f'Index error: save_dot file not in [0, 1], in{d_dir}')
    # print(f'Index error')
    # 处理main函数
    for i, filename in enumerate(files):
        if len(d_mn) == 0:
            break
        # 拼接
        read_filename = os.path.join(d_dir, filename)
        file = open(read_filename, "r")
        content = file.read()
        # 读取原文件的文本信息
        file = open(read_filename, "r")
        first_line = file.readline()
        second_line = file.readline()
        file.close()
        if len(second_line) == 0:
            return 0
        # d_name = f'digraph(\s+)?"{d_mn}"'
        d_name = f'module&gt"'
        if d_name in first_line:
        # if re.match(d_name, first_line) and file_len > 50:
            with open(read_filename, "r") as file:
                content = file.read()
            if 'io.joern' not in content and 'ErrorStatement' not in content:
                content_without_spaces = content.replace(" ", "")
                # 拼接
                save_filename = os.path.join(d_save, filename)
                # 写入到新文件
                target_file = open(save_filename, "w")
                target_file.write(content_without_spaces)
                target_file.write('\n')
                target_file.close()
                file.close()
                return 1
    # 最后
    # print(f'Index error: something wrong, in{d_dir}')
    print(f'Index error')
    dot_match = r'^\d\d?\d?-(\w\w\w)(.dot)$'
    process_match = re.match(dot_match, files[0])
    if not process_match:
        return 0
    else:
        process_type = process_match.group(1)
    process_name = f'0-{process_type}.dot'
    read_filename = os.path.join(d_dir, process_name)
    # 读取原文件的文本信息，写入到新文件
    with open(read_filename, "r") as file:
        content = file.read()
    if 'io.joern' in content and 'ErrorStatement' in content:
        return 0
    content_without_spaces = content.replace(" ", "")
    save_filename = os.path.join(d_save, process_name)
    with open(save_filename, "w") as target_file:
        target_file.write(content_without_spaces)
        target_file.write('\n')
    # 返回
    return 1


def generate_ast(source_dir, output):
    # total_dataset = os.listdir(args.clean_Bitree_dir)
    total_dataset = os.listdir(source_dir)
    for main_data in total_dataset:
        if main_data != 'train':
            continue
        total_dict = {}
        # Bitree_folder_path = os.path.join(args.clean_Bitree_dir, main_data)
        Bitree_folder_path = os.path.join(source_dir, main_data)
        Bitree_list = os.listdir(Bitree_folder_path)
        Bitree_list_sort = sorted(Bitree_list, key=lambda x: int(re.search(r'__(\d+)___', x).group(1)))
        tr_tq = tqdm(enumerate(Bitree_list_sort))
        for tree_index, Bit in tr_tq:
            tr_tq.update(1)
            ast_dict_folder = os.path.join(Bitree_folder_path, Bit, Bit+'_ast.json')
            with open(ast_dict_folder, 'r', encoding='utf-8') as inf:
                one_tree = json.load(inf)
                total_dict[str(tree_index)] = one_tree
        ast_name = main_data + '_ast.json'
        out_ast_path = os.path.join(output, ast_name)
        with open(out_ast_path, 'w') as fw:
            json.dump(total_dict, fw)


so = r'/home/user/PY_Projects/lmc/code_summary/data/mid_output/clean_tree'
out_dir = r'/home/user/PY_Projects/回收站'
# generate_ast(so, out_dir)


tree_data = [
    {"id": 22, "type": "RETURN", "children": [18], "value": "return value[0]"},
    {"id": 21, "type": "CALL", "children": [22, 18, 24], "value": "value[0]"},
    {"id": 24, "type": "RETURN", "children": [18], "value": "return u'"},
    {"id": 14, "type": "IDENTIFIER", "children": [21], "value": "<empty>"},
    {"id": 18, "type": "METHOD_RETURN", "value": "RET"}
]

# 将树的列表形式转换为字典形式，以便更容易地通过id访问节点
tree_dict = {node['id']: node for node in tree_data}

def preorder_traversal(node_id, tree_dict):
    node = tree_dict[node_id]
    tokens = [node['value']]  # 首先获取当前节点的value作为token
    # 如果节点有子节点，则递归访问每个子节点
    if 'children' in node:
        for child_id in node['children']:
            tokens.extend(preorder_traversal(child_id, tree_dict))
    return tokens

# 找到根节点，即没有在其他节点的children中出现的节点
root_id = None
all_children_ids = {child_id for node in tree_data if 'children' in node for child_id in node['children']}
for node in tree_data:
    if node['id'] not in all_children_ids:
        root_id = node['id']
        break

# 使用先序遍历函数，并将结果tokens拼接成一个字符串
result_tokens = preorder_traversal(root_id, tree_dict)
result_string = ' '.join(result_tokens)

print(result_string)


