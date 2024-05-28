import os
import shutil

def remove_empty_folders(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for dir in dirs:
            folder_path = os.path.join(root, dir)
            if not os.listdir(folder_path):
                try:
                    os.rmdir(folder_path)
                    print(f"{folder_path} 为空，已删除.")
                except OSError:
                    print(f"删除失败： {folder_path}")
            else:
                remove_empty_folders(folder_path)


def delete_directory(directory):
    try:
        shutil.rmtree(directory)
        print("目录文件存在错误，已删除：", directory)
    except OSError as e:
        print("删除含错误文件目录失败：", directory, e)

# 读取原文件的文本信息
def examine_file(path, type):
    # pdg存在，已被检索的标志
    flag = -1  # 不存在
    # 获取文件夹下的全部文件名
    files = os.listdir(path)
    if len(files) == 0:
        return 1

    for file in files:
        file_list = os.path.join(path, file)
        # print(file_list)
        json_files = os.listdir(file_list)
        if len(json_files) < 3:
            delete_directory(file_list)
            continue
        for json_file in json_files:
            file_ext = json_file[-8:-5]
            # print(json_file)
            # print(file_ext)
            if file_ext == type:
                flag = 0
                filename = os.path.join(file_list, json_file)
                # print(os.path.getsize(filename))
                if os.path.getsize(filename) < 100:
                    # print("size < 100")
                    delete_directory(file_list)

        if flag == -1:
            # print("pdg does not exist")
            delete_directory(file_list)
    return flag

path = r"/media/zhangfanlong/c7e5b1f8-c17b-4f55-b700-bbd6d16c1e04/lmc/codenet/Project_CodeNet/derived/benchmarks/python_Bitree/test"
remove_empty_folders(path)
examine_file(path, 'pdg')
