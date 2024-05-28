import json
import logging
import os
import pickle
import re
import shutil

from tqdm import tqdm

from .find_dot import process_dot
from .dfs_tree import write_ast
# from find_dot import process_dot
# from dfs_tree import write_ast



def joint_result_path(root):
    # 获取文件夹下的全部文件名
    files = os.listdir(root)
    filenames = []
    for file in files:
        # 拼接
        filename = os.path.join(root, file)
        # filename = os.path.join(root, file, 'result_js')
        filenames.append(filename)
    return filenames


def joint_tree_path(result, match):
    # 获取文件夹下的全部文件名
    files = os.listdir(result)
    filenames = []
    for file in files:
        # 拼接
        filename = os.path.join(result, file)
        if re.match(match, filename):
            filenames.append(filename)
            break
    return filenames


def is_folder_exists(folder_path):
    exist = os.path.isdir(folder_path)
    if exist:
        file_list = os.listdir(folder_path)
        if len(file_list) == 3:
            for file in file_list:
                file_path = os.path.join(folder_path, file)
                if os.path.getsize(file_path):
                    pass
                else:
                    exist = 0
                    return exist
        else:
            exist = 0
        # print(f"文件夹 {folder_path} 已存在")
    return exist


def examine_file(path):
    # 获取文件夹下的全部文件名
    if not os.path.exists(path):
        return 1
    files = os.listdir(path)
    if len(files) < 1:
        return 1
    return 0


def to_delete_file(root_dir, save_root_dir, code_type, code_name):
    dot_path = os.path.join(root_dir, code_type, code_name)
    tree_path = os.path.join(save_root_dir, code_type, code_name)
    if os.path.exists(dot_path):
        to_remove(dot_path)
    if os.path.exists(tree_path):
        to_remove(tree_path)


def to_remove(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    # logging.basicConfig(level=logging.INFO)
    # logging.info(f'Trying to delete {path}')


def double_check(path):
    with open(path, "r") as f:
      first_line = f.readline()
      second_line = f.readline()
      if len(second_line) == 0:
          return 0
    return 1


def dot_process_entrance(r, dot_root, language, code_mn):
    flag = 1
    ast_match = r'.*ast'
    cfg_match = r'.*cfg'
    pdg_match = r'.*pdg'
    ast_dir = joint_tree_path(r, ast_match)
    cfg_dir = joint_tree_path(r, cfg_match)
    pdg_dir = joint_tree_path(r, pdg_match)
    # 检查文件是否为空
    if len(ast_dir) == 0 or len(cfg_dir) == 0 or len(pdg_dir) == 0:
        return 0
    if examine_file(ast_dir[0]) or examine_file(cfg_dir[0]) or examine_file(pdg_dir[0]):
        print(f'Dot file needed in {r}.')
        flag = 0
        return flag

    # 分别处理ast.dot/cfg.dot/pdg.dot
    for root_type in [ast_dir, cfg_dir, pdg_dir]:
        success = process_dot(root_type[0], dot_root, language, code_mn)
        if success == 0:
            print(f'Error in process_dot {r}')
            flag = 0
            break
    return flag


def get_json_ast(root_dir, save_root_dir, language, pkl_dir):
    # 正则表达式
    json_match = r'.*all'
    dot_match = r'^(\d\d?\d?)-(\w\w\w)(.dot)$'
    # 用根目录拼接
    mid_file = os.listdir(root_dir)
    # 获取文件夹的名称mid_folder，例如test
    for mid_folder in mid_file:
        error_file_list = {'Dot_processing_error': [], 'process_type_error': [], 'dot_exported_file_error': [], 'ast_writing_error': []}
        # if mid_folder != 'fix':
        #     continue
        # pkl
        pkl_json_name = mid_folder + '.json'
        pkl_json_path = os.path.join(pkl_dir, pkl_json_name)
        with open(pkl_json_path, 'r') as f:
            method_name_data = json.load(f)
        # 子文件夹名
        mid_dir = os.path.join(root_dir, mid_folder)
        file_dir = os.listdir(mid_dir)
        file_path_dir = joint_result_path(mid_dir)
        t = tqdm(enumerate(file_path_dir))
        total_len = len(file_path_dir)
        for file_id, r in t:
            code_name = file_dir[file_id]
            # 最终dot文件是否处理好
            dot_root = os.path.join(r, code_name + '_dot')
            save_root = os.path.join(save_root_dir, mid_folder, code_name)
            # 没有就创建save_dot
            os.makedirs(save_root, exist_ok=True)
            # 从pkl文件读取
            index = method_name_data['name'].index(code_name)
            code_mn = method_name_data['methodName'][index]
            # if not is_folder_exists(dot_root):
            to_remove(dot_root)
            os.makedirs(dot_root)
            dot_result = dot_process_entrance(r, dot_root, language, code_mn)
            if dot_result == 0:
                error_file_list['Dot_processing_error'].append(r)
                if os.path.exists(save_root):
                    to_remove(save_root)
                continue

            # 判断目标文件夹是否存在
            save_check = 1
            if is_folder_exists(save_root):
                # 使用json查询结点value，生成ast
                dot_files = os.listdir(dot_root)
                if len(dot_files) != 3:
                    error_file_list['Dot_processing_error'].append(r)
                    print(f'Error in process_dot {r}')
                    if os.path.exists(save_root):
                        to_remove(save_root)
                    continue
                for dot_file in dot_files:  # dot_file:  0-ast.dot
                    # 找出当前类型(ast/cfg/pdg)
                    process_type = re.match(dot_match, dot_file)
                    if process_type:
                        dot_num = process_type.group(1)
                        if dot_num not in ['0', '1']:
                            save_check = 0
                            break
                if save_check == 1:
                    continue

            # 用result目录拼接json目录
            json_dir = joint_tree_path(r, json_match)
            export_file = os.path.join(json_dir[0], 'export.json')
            if not os.path.exists(export_file):
                error_file_list['dot_exported_file_error'].append(r)
                print(f'Exported file in {r} does not exist.')
                if os.path.exists(save_root):
                    to_remove(save_root)
                continue

            # 使用json查询结点value，生成ast
            dot_files = os.listdir(dot_root)
            if len(dot_files) != 3:
                if os.path.exists(save_root):
                    to_remove(save_root)
                error_file_list['dot_exported_file_error'].append(r)
                print(f'Error in process_dot {r}')
                continue
            dot_success = 1
            for dot_file in dot_files:     # dot_file:  0-ast.dot
                # 找出当前类型(ast/cfg/pdg)
                process_type = re.match(dot_match, dot_file)
                if process_type is None:
                    dot_success = 0
                    print(f'Error in process type: {r}')
                    if os.path.exists(save_root):
                        to_remove(save_root)
                    break
                dot_num = process_type.group(1)
                if dot_num not in ['0', '1']:
                    to_remove(dot_root)
                    os.makedirs(dot_root)
                    dot_result = dot_process_entrance(r, dot_root, language, code_mn)
                    if dot_result == 0:
                        dot_success = 0
                        error_file_list['Dot_processing_error'].append(r)
                        if os.path.exists(save_root):
                            to_remove(save_root)
                        break
                    break
            if dot_success == 0:
                continue
            for dot_file in dot_files:  # dot_file:  0-ast.dot
                # 找出当前类型(ast/cfg/pdg)
                process_type = re.match(dot_match, dot_file)
                if process_type is None:
                    error_file_list['process_type_error'].append(r)
                    print(f'Error in process type: {r}')
                    if os.path.exists(save_root):
                        to_remove(save_root)
                    break
                processed_type = process_type.group(2)
                # 拼接
                save_dot_path = os.path.join(dot_root, dot_file)
                file_tail = code_name + '_' + processed_type + '.json'
                save_ast_json = os.path.join(save_root, file_tail)
                if os.path.exists(save_ast_json):
                    continue
                # 完成文件的读写生成ast
                if os.path.exists(save_dot_path) and os.path.getsize(save_dot_path):
                    success = write_ast(json_dir[0], save_dot_path, save_ast_json, processed_type)
                    if success == 1:
                        # print(f'Success in {r}, {file_id} in {total_len}')
                        pass
                    else:
                        print(f'Error in write_ast: {r}')
                        error_file_list['ast_writing_error'].append(r)
                        if os.path.exists(save_root):
                            to_remove(save_root)
                        break

        total_error_len = 0
        for error_type in error_file_list:
            total_error_len += len(error_file_list[error_type])
        error_file_json = os.path.join(save_root_dir, mid_folder+f'_{total_error_len}.json')
        with open(error_file_json, 'w') as f:
            json.dump(error_file_list, f)






if __name__ == '__main__':
    # 文件根目录
    root_dir = r'/home/user/PY_Projects/0_MUTI/multilanguage_json/base/data_2_14(test)/mid_output/dot'
    save_root_dir = r'/home/user/PY_Projects/0_MUTI/multilanguage_json/base/data_2_14(test)/mid_output/tree'

    get_json_ast(root_dir, save_root_dir, 'py', '')



