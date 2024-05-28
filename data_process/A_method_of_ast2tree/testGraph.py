import os


def joint_result_path(root):
    # 获取文件夹下的全部文件名
    files = os.listdir(root)
    filenames = []
    for file in files:
        # 拼接
        filename = os.path.join(root, file)
        filenames.append(filename)
    return filenames
def get_tree(root_dir, save_root_dir, temp_tree_path, temp_sorted_tree_path):
    # 用根目录拼接result目录
    result_dir = joint_result_path(root_dir)
    print(result_dir)
    # print(result_dir)
    # 循环遍历每个文件夹
    # for id, r in enumerate(result_dir):
    #     # 获取文件夹的名称，例如p00002
    #     files = os.listdir(root_dir)


if __name__ == '__main__':
    # file = r'D:\Coding\PY_project\任务\数据处理任务\dot和json转树\output_data\py_tree'
    # out = r'D:\Coding\PY_project\任务\数据处理任务\dot和json转树\output_data\python_out\save_tree'
    file = r'/media/zhangfanlong/c7e5b1f8-c17b-4f55-b700-bbd6d16c1e04/lmc/codenet/Project_CodeNet/derived/benchmarks/py_tree'
    out = r'/media/zhangfanlong/c7e5b1f8-c17b-4f55-b700-bbd6d16c1e04/lmc/codenet/Project_CodeNet/derived/benchmarks/python_Bitree'
    temp_tree_path = 'save_ast/tree.json'
    temp_sorted_tree_path = 'save_ast/tree_sorted.json'

    get_tree(file, out, temp_tree_path, temp_sorted_tree_path)