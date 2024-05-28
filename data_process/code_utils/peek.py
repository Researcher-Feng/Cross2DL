import json
import pickle

def peek_pkl(pkl_path, i):
    f = open(pkl_path, 'rb')
    s = pickle.load(f)
    print('Print an example:')
    print(f's.shape:{s.shape}')
    print('name:', s.loc[i][['name'][0]])
    print('code_tokens:')
    print(s.loc[i][['code_tokens'][0]])
    print('desc_tokens:')
    print(s.loc[i][['desc_tokens'][0]])
    print('methodName:')
    print(s.loc[i][['methodName'][0]])


def peek_json(file_path, num):
    j = open(file_path, 'r')
    cfg_dict = json.load(j)
    print(cfg_dict)
    # for i, key in enumerate(cfg_dict):
    #     print(f'{key}:{cfg_dict[key]}', end=', ')
    #     if i == num:
    #         print('...len(cfg_dict):', len(cfg_dict))
    #         break


if __name__ == '__main__':
    # gz_path = r'/home/user/PY_Projects/参考方法/python/process_data/pythondata/final/source_jsonl/test'
    # merge_all_jsonl_to_total(gz_path, 1)

    # pkl_path = r'/home/user/PY_Projects/核心代码/data/mid_output/pkl/valid.pkl'
    # peek_pkl(pkl_path, 0)

    # json_path = r'/home/user/PY_Projects/0_MUTI/速通/data/final_output/vocab/vocab_2_code_tokens.json'
    json_path = r'/home/user/PY_Projects/参考方法/0_process_order/4_pkl4/data_source/1-cfg.json'
    peek_json(json_path, 10)

