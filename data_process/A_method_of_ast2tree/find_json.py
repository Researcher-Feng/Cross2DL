import json
import os

# 根据dot文件中结点的id在json文件找label和value
def find_from_json(j_dir, process_type, dot_id):
    target_value = None
    target_edge = None
    # 获取文件夹下的全部文件名
    files = os.listdir(j_dir)
    for file in files:
        filename = os.path.join(j_dir, file)
        with open(filename, 'r', encoding='utf-8') as f:
            f_read = json.load(f)
            # 找value
            values = f_read["@value"]["vertices"]
            for value in values:
                if value["id"]["@value"] == dot_id:
                    # target_value = value["properties"]["CODE"]["@value"]
                    if 'CODE' in value["properties"]:
                        # print(j_dir)
                        target_value = value["properties"]["CODE"]["@value"]
                    else :
                        print('value["properties"] has not key :["CODE"]')
                    break
            # 找label
            edges = f_read["@value"]["edges"]
            for edge in edges:
                dot_label = edge["label"]
                if dot_label == process_type:
                    if edge['inV']['@value'] == dot_id or edge['outV']['@value'] == dot_id:
                        # print(f"边：{edge['id']} 入点：{edge['inV']['@value']} 出点：{edge['outV']['@value']}")
                        target_edge = edge['inVLabel']
                        break
            return target_value, target_edge

# 根据dot文件中结点的id在json文件找value
def find_value_from_json(j_dir, dot_id):
    # 获取文件夹下的全部文件名
    files = os.listdir(j_dir)
    for file in files:
        filename = j_dir + file
        with open(filename, 'r', encoding='utf-8') as f:
            f_read = json.load(f)
            values = f_read["@value"]["vertices"]
            # 筛选
            for value in values:
                if value["id"]["@value"] == 2:
                    print(value["properties"]["CODE"]["@value"])

# if __name__ == '__main__':
#     # linux
#     # json_dir = r'/home/user/cv_linux/dot2json/source_data/Result_Java250mini_high_joern_version/p00002/result_java/s003798551_all'
#     # dot_dir = r'/home/user/cv_linux/dot2json/source_data/Result_Java250mini_high_joern_version/p00002/result_java/s003798551_ast'
#     # save_java_root = r'/home/user/cv_linux/dot2json/related_methods/GetData(1)/GetData/myPratices/save_dot'
#
#     # windows
#     json_dir = r'D:/Coding/PY_project/任务/数据处理任务/dot和json转树/source_data/Result_Java250mini_high_joern_version/p00002/result_java/s003798551_all/'
#     dot_dir = r'D:/Coding/PY_project/任务/数据处理任务/dot和json转树/source_data/Result_Java250mini_high_joern_version/p00002/result_java/s003798551_ast/'
#     save_java_root = r'D:/Coding/PY_project/0_My/Pratices/save_dot/'
#
#     v = find_label_from_json(json_dir, 'AST', 92)
#     print(v)
