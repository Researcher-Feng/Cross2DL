import os
import json

root_path = r'/media/zhangfanlong/c7e5b1f8-c17b-4f55-b700-bbd6d16c1e04/lmc/codenet/Project_CodeNet/derived/benchmarks/python_Bitree/test'
result_file_path = r'/media/zhangfanlong/c7e5b1f8-c17b-4f55-b700-bbd6d16c1e04/lmc/codenet/Project_CodeNet/derived/benchmarks/py_final/test.pdg.json'

def read_json_file(path):
    with open(os.path.join(path, '0-pdg.json'), 'r') as f:
        json_str = f.read()
        return json.loads(json_str)


def add_dict_to_new_dict(old_dict, new_dict, key):
    new_dict[key] = old_dict
    return new_dict


basic_count = 0
basic_dict = {}
with open(result_file_path, 'r') as f:
    json_str = f.read()
    if json_str:
        basic_dict = json.loads(json_str)
        basic_count = len(basic_dict)

if basic_count == 0:
    new_dict = {}
    count = 0
else:
    count = basic_count
    new_dict = basic_dict

total_files = 40  # 读取的文件夹数量

for folder_name in os.listdir(root_path):
    if count >= total_files:
        break

    folder_path = os.path.join(root_path, folder_name)
    if os.path.isdir(folder_path):
        old_dict = read_json_file(folder_path)
        new_dict = add_dict_to_new_dict(old_dict, new_dict, str(count))
        count += 1

    print(f'Processed {count} folders.')

# 统计字典的长度
dict_length = len(new_dict)
print(f'The length of the dictionary is: {dict_length}')

# 保存所有文件夹的结果到一个文件中
with open(result_file_path, 'w') as f:
    json.dump(new_dict, f)

print(f'Processed {count} folders in total.')