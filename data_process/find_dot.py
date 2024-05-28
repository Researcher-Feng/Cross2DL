import os
import re

import pydot


def process_dot(d_dir, d_save, d_type, d_mn):
    # 获取文件夹下的全部文件名
    files = os.listdir(d_dir)
    '''
    length_file = []
    for filename in files:
        read_filename = os.path.join(d_dir, filename)
        file_length = os.path.getsize(read_filename)
        length_file.append(file_length)
    max_len_value = max(length_file)
    max_len_index = length_file.index(max_len_value)
    check_filename = files[max_len_index]
    check_filename_index = check_filename.split('-')[0]
    if int(check_filename_index) < 2:
        read_filename = os.path.join(d_dir, check_filename)
        file = open(read_filename, "r")
        content = file.read()
        if 'io.joern' in content or 'ErrorStatement' in content:
            return 0
        content_without_spaces = content.replace(" ", "")
        # 拼接
        save_filename = os.path.join(d_save, check_filename)
        # 写入到新文件
        target_file = open(save_filename, "w")
        target_file.write(content_without_spaces)
        target_file.write('\n')
        target_file.close()
        file.close()
        return 1
    print(f'Index error: save_dot file not in [0, 1], in{d_dir}')
    '''
    # 处理main函数
    for i, filename in enumerate(files):
        if len(d_mn) == 0:
            break
        # 拼接
        read_filename = os.path.join(d_dir, filename)
        file = open(read_filename, "r")
        content = file.read()
        file_len = len(content)
        # 读取原文件的文本信息
        file = open(read_filename, "r")
        first_line = file.readline()
        second_line = file.readline()
        file.close()
        if len(second_line) == 0:
            return 0
        d_name = f'digraph(\s+)?"{d_mn}"'
        # if d_name in first_line:
        # if re.match(d_name, first_line) and file_len > 50:
        if 'module' in first_line and ('0' in filename or '1' in filename) or file_len > 50:
            file = open(read_filename, "r")
            content = file.read()
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
    print(f'Index error: something wrong, in{d_dir}')
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
    content_without_spaces = content.replace(" ", "")
    save_filename = os.path.join(d_save, process_name)
    with open(save_filename, "w") as target_file:
        target_file.write(content_without_spaces)
        target_file.write('\n')
    # 返回
    return 1


def find_note_from_dot(d_dir):
    # 参数设置
    count = 0
    # 获取文件夹下的全部文件名
    files = os.listdir(d_dir)
    # for file in files:
        # 拼接
        # filename = d_dir + file

    filename = d_dir + files[0]
    # 读取 .dot 文件
    Graph = pydot.graph_from_dot_file(filename)

    graph = Graph[0]
    # print(graph)

    # 获取节点和边的信息
    nodes = {}
    edges = []

    for node in graph.get_nodes():
        node_name = node.get_name().strip('"')
        node_label = node.get_label().strip('"')
        nodes[node_name] = node_label

    for edge in graph.get_edges():
        edge_start = edge.get_source().strip('"')
        edge_end = edge.get_destination().strip('"')
        edges.append((edge_start, edge_end))

    # 打印节点和边的信息
    print("节点：")
    for node_name, node_label in nodes.items():
        print(f"{node_name}: {node_label}")

    print("\n边的关系：")
    for edge_start, edge_end in edges:
        print(f"{edge_start} -> {edge_end}")

    print("\n\n")


if __name__ == '__main__':
    # linux
    # json_dir = r'/home/user/cv_linux/dot2json/source_data/Result_Java250mini_high_joern_version/p00002/result_java/s003798551_all'
    # dot_dir = r'/home/user/cv_linux/dot2json/source_data/Result_Java250mini_high_joern_version/p00002/result_java/s003798551_ast'
    # save_java_root = r'/home/user/cv_linux/dot2json/related_methods/GetData(1)/GetData/myPratices/save_dot'
    # windows
    json_dir = r'D:/Coding/PY_project/任务/数据处理任务/dot和json转树/source_data/Result_Java250mini_high_joern_version/p00002/result_java/s003798551_all/'
    dot_dir = r'D:\Coding\PY_project\任务\数据处理任务\dot和json转树\source_data\Result_Python800mini_high_joern_version\p00000\result_py\s002191454_ast/'
    save_java_root = r'D:/Coding/PY_project/0_My/Pratices/save_dot/'

    process_dot(dot_dir, save_java_root)
    # find_note_from_dot(save_java_root)