import argparse
import gzip
import os.path
import re

import tables
import h5py
import jsonlines
from code_utils.clean import *
from A_method_of_ast2tree.run import *
from A_method_of_tree2Bitree.start_process_tree import *

PAD_ID, UNK_ID = [0, 1]


def generate_mix_json(args):
    # 1.打开目录下的cfg和pdg文件
    source_files = os.listdir(args.temp_tree_dir)
    graph_tree_files = []
    for filename in source_files:
        if filename.split('_')[0] in ['train', 'dev', 'test'] and 'cfg' in filename:
            graph_tree_files.append(filename)
    for t_file_1 in graph_tree_files:
        print('Processing file {}'.format(t_file_1))
        mix_trees = {}
        mix_trees_out = {}
        folder_name = t_file_1.split('_')[0]
        model_type_1 = t_file_1.split('_')[1]
        if model_type_1 == 'cfg':
            model_type_2 = 'pdg'
        else:
            model_type_2 = 'cfg'
        t_file_2 = f"{folder_name}_{model_type_2}_temp.json"
        t_path_1 = os.path.join(args.temp_tree_dir, t_file_1)
        t_path_2 = os.path.join(args.temp_tree_dir, t_file_2)
        with open(t_path_1, 'r', encoding='utf-8') as inf:
            trees_1 = json.load(inf)
        with open(t_path_2, 'r', encoding='utf-8') as inf:
            trees_2 = json.load(inf)
        # 2.合并
        for out_key, out_value in trees_2.items():
            mix_trees_list = []
            if len(trees_2[out_key]) > len(trees_1[out_key]):
                mix_trees[out_key] = trees_2[out_key]
                temp_tree = trees_1[out_key]
                if model_type_2 == 'cfg':
                    special_key = 'snode_pdg'
                else:
                    special_key = 'snode_cfg'
            else:
                mix_trees[out_key] = trees_1[out_key]
                temp_tree = trees_2[out_key]
                if model_type_1 == 'cfg':
                    special_key = 'snode_pdg'
                else:
                    special_key = 'snode_cfg'
            for mix_key in mix_trees[out_key].keys():
                for temp_key, temp_value in temp_tree.items():
                    if temp_value['old_id'] == mix_trees[out_key][mix_key]['old_id']:
                        if special_key in temp_value:
                            mix_trees[out_key][mix_key][special_key] = temp_value[special_key]
                mix_trees_list.append(mix_trees[out_key][mix_key])
            # 3.排序
            sorted_trees = sort_mix_method(mix_trees_list)
            sort_trees_dict = {}
            for dot_num, small_tree in enumerate(sorted_trees):
                sort_trees_dict[dot_num] = small_tree
            mix_trees_out[out_key] = sort_trees_dict

        # 4.写入
        out_name = folder_name + f'_mix.json'
        out_path = os.path.join(args.output_dir, out_name)
        final_dict_str = json.dumps(mix_trees)
        with open(out_path, 'w') as fw:
            fw.write(final_dict_str)


def generate_json(args, model_type):
    # 1.建立一个字典，存放同一个目录下所有的最终的cfg/pdg
    temp_dict = {}

    # 2.打开词典
    vocab_file_name = f'vocab_mix.json'
    vocab_file_path = os.path.join(args.vocab_dir, vocab_file_name)
    with open(vocab_file_path, 'r', encoding='utf-8') as v:
        voca = json.load(v)

    # 3.打开目录下的每个待处理文件,执行排序
    # source_files = os.listdir(args.clean_tree_dir)
    source_files = os.listdir(args.clean_graph_dir)
    for t_folder in source_files:
        final_dict = {}
        # graph_folders = os.path.join(args.clean_tree_dir, t_folder)
        graph_folders = os.path.join(args.clean_graph_dir, t_folder)
        cfg_folder_list = os.listdir(graph_folders)
        cfg_folder_list_sort = sorted(cfg_folder_list, key=lambda x: int(re.search(r'__(\d+)___', x).group(1)))
        t = tqdm(enumerate(cfg_folder_list_sort))
        for i, source_file in t:
            json_file_name = source_file + f'_{model_type}.json'
            json_file_name_second = f'train_{model_type}.json'
            file_path = os.path.join(graph_folders, source_file, json_file_name)
            file_path_second = os.path.join(graph_folders, source_file, json_file_name_second)
            if os.path.exists(file_path_second):
                with open(file_path_second, 'r', encoding='utf-8') as inf:
                    trees = json.load(inf)
                    sort_trees = sort_method(trees, model_type)
            else:
                with open(file_path, 'r', encoding='utf-8') as inf:
                    trees = json.load(inf)
                    sort_trees = sort_method(trees, model_type)
            sort_trees_dict = {}
            for dot_num, small_tree in enumerate(trees):
                sort_trees_dict[dot_num] = small_tree
            temp_dict[i] = sort_trees_dict
            # 4.每个文件的cfg，和创建字典相同的方式进行清洗，清洗完立刻查字典，对'wordid'进行序列化，查不到就给1
            for sort_tree in sort_trees:
                rep = check_voca(sort_tree['wordid'], voca)
                # # 补空
                # if len(rep) <= 0:
                #     rep.append(0)
                sort_tree['wordid'] = rep
                del (sort_tree['id'])
            new_dict = {}
            for dot_num, small_tree in enumerate(sort_trees):
                new_dict[dot_num] = small_tree
            final_dict[i] = new_dict
            t.update(1)

        # 5.写入
        temp_name = t_folder + f'_{model_type}_temp.json'
        temp_path = os.path.join(args.temp_tree_dir, temp_name)
        temp_dict_str = json.dumps(temp_dict)
        with open(temp_path, 'w') as fw:
            fw.write(temp_dict_str)
        out_name = t_folder + f'_{model_type}.json'
        out_path = os.path.join(args.output_dir, out_name)
        final_dict_str = json.dumps(final_dict)
        with open(out_path, 'w') as fw:
            fw.write(final_dict_str)


def to_remove(path):
    shutil.rmtree(path)
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Trying to delete {path}')


def get_clean_graph(args):
    clean_tree_dir = args.clean_tree_dir
    clean_graph_dir = args.clean_graph_dir
    source_files = os.listdir(clean_tree_dir)
    model_type_list = ['cfg', 'pdg']
    for i, json_file in enumerate(source_files):
        if json_file != 'train':
            continue
        the_json_path = os.path.join(clean_tree_dir, json_file)
        the_save_path = os.path.join(clean_graph_dir, json_file)
        json_folder_list = os.listdir(the_json_path)
        for json_folder in tqdm(json_folder_list):
            save_dir = os.path.join(the_save_path, json_folder)
            os.makedirs(save_dir, exist_ok=True)
            for model_type in model_type_list:
                json_path = os.path.join(the_json_path, json_folder, json_folder + f'_{model_type}.json')
                if not os.path.exists(json_path):
                    print(f'Processing {json_folder} file in {json_file} ERROR!!!')
                    break
                with open(json_path, 'r') as j:
                    cfg_dict = json.load(j)

                    # 2.24type
                    children_node = []
                    total_info = []
                    for single_dict in cfg_dict:
                        # single_dict['value'].insert(0, single_dict['type'])
                        for i_word in single_dict['value']:
                            if i_word not in total_info:
                                total_info.append(i_word)
                        if 'children' in single_dict:
                            children_node.extend(single_dict['children'])

                # 2.24type
                for single_dict in cfg_dict:
                    if single_dict['id'] not in children_node:
                        single_dict['value'] = total_info

                save_json = os.path.join(save_dir, json_folder + f'_{model_type}.json')
                with open(save_json, 'w') as fw:
                    json.dump(cfg_dict, fw)


def get_clean_tree(args, del_ws):
    tree_dir = args.tree_dir
    clean_tree_dir = args.clean_tree_dir
    source_files = os.listdir(tree_dir)
    model_type_list = ['ast', 'cfg', 'pdg']
    for i, json_file in enumerate(source_files):
        if json_file != 'train':
            continue
        the_json_path = os.path.join(tree_dir, json_file)
        the_save_path = os.path.join(clean_tree_dir, json_file)
        # the_save_path = os.path.join(clean_tree_dir, 'train')

        json_folder_list = os.listdir(the_json_path)
        for json_folder in tqdm(json_folder_list):
            # print(f'Processing {json_folder} file in {json_file}')
            json_folder_path = os.path.join(the_json_path, json_folder)
            save_dir = os.path.join(the_save_path, json_folder)
            # if os.path.exists(save_dir):
            #     continue
            os.makedirs(save_dir, exist_ok=True)
            for model_type in model_type_list:
                json_path = os.path.join(the_json_path, json_folder, json_folder + f'_{model_type}.json')
                if not os.path.exists(json_path):
                    print(f'Processing {json_folder} file in {json_file} ERROR!!!')
                    break
                with open(json_path, 'r') as j:
                    cfg_dict = json.load(j)

                    for single_dict in cfg_dict:
                        words = wt(single_dict['value'])
                        new_token_words = clean_token(words, 'code')
                        if model_type == 'ast':
                            if len(new_token_words) != 0:
                                single_dict['value'] = get_one_words(new_token_words)
                        else:
                            single_dict['value'] = get_several_words(new_token_words, 20, 'def', del_ws)
                save_json = os.path.join(save_dir, json_folder + f'_{model_type}.json')
                # save_json = os.path.join(save_dir, 'train' + f'_{model_type}.json')

                with open(save_json, 'w') as fw:
                    json.dump(cfg_dict, fw)


def generate_vocab(args, model_type, add_voca_in_string, add_voca_in_list):
    print("Generating vocab...")
    # 1.准备一个列表，存放词表单词，最后传给vocab_info作为词表
    token_words = []
    vocab_info = Counter(token_words)

    # 2.先在词典添加基础词汇，包括最重要的<unk>
    basic_vocabulary = []
    vocab_ast_index = {'<pad>': 0, '<unk>': 1}
    if len(add_voca_in_string) == 0 and len(add_voca_in_list) != 0:
        basic_vocabulary = add_voca_in_list
    elif len(add_voca_in_string) != 0:
        basic_vocabulary = list(add_voca_in_string) + add_voca_in_list
    current_len = len(basic_vocabulary) + 2
    vocab_info.update(basic_vocabulary)

    # 3.文件夹下所有pkl文件的code_token，经过筛选后，录入词表
    if model_type == 'mix':
        # vocab_info = read_vocab_from_json(args.clean_tree_dir, 'cfg', vocab_info)
        # vocab_info = read_vocab_from_json(args.clean_tree_dir, 'pdg', vocab_info)
        vocab_info = read_vocab_from_json(args.clean_graph_dir, 'cfg', vocab_info)
        vocab_info = read_vocab_from_json(args.clean_graph_dir, 'pdg', vocab_info)
        vocab_current = vocab_info.most_common()[:args.vocab_max_len_graph_token]
    elif model_type == 'ast':
        vocab_info = read_vocab_from_ast(args.Bitree_dir, vocab_info)
        # vocab_info = read_vocab_from_ast(args.Bitree_dir, args.clean_Bitree_dir, vocab_info)
        vocab_current = vocab_info.most_common()[:args.vocab_max_len_ast_token]
    elif model_type == 'code_tokens':
        vocab_info = read_vocab_from_pkl(args.pkl_token_dir, model_type, vocab_info)
        vocab_current = vocab_info.most_common()[:args.vocab_max_len_code_token]
        # print(vocab_current[100000:100100])
    elif model_type == 'desc_tokens':
        vocab_info = read_vocab_from_pkl(args.pkl_token_dir, model_type, vocab_info)
        vocab_current = vocab_info.most_common()[:args.vocab_max_len_desc_token]
        # print(vocab_current[40000:50000])
    else:
        print('WRONG!')
        return

    # 4.从词表到映射词典：统计词表中每个词的词频，词频高的放在词典前面
    for i, item in enumerate(vocab_current):
        vocab_ast_index[item[0]] = i + current_len
    print('Finish Indexing!')

    # 5.词典输出为json文件
    token_dic_str = json.dumps(vocab_ast_index)
    vocab_file_name = 'vocab_' + model_type + '.json'
    vocab_file_path = os.path.join(args.vocab_dir, vocab_file_name)
    print(f'With {current_len} basic words, {vocab_file_name} has a total vocabulary of {len(vocab_ast_index)} !')
    with open(vocab_file_path, 'w') as vocab_ast_file:
        vocab_ast_file.write(token_dic_str)
    print('Vocabulary File is ready!')
    if model_type == 'mix':
        vocab_cfg_file_path = os.path.join(args.vocab_dir, 'vocab_cfg.json')
        vocab_pdg_file_path = os.path.join(args.vocab_dir, 'vocab_pdg.json')
        shutil.copy(vocab_file_path, vocab_cfg_file_path)
        shutil.copy(vocab_file_path, vocab_pdg_file_path)


def generate_h5(args, type):
    # 1. 从 pkl 获取 code_tokens
    pkl_names = os.listdir(args.pkl_token_dir)
    for pkl_name in pkl_names:
        print(f'Reading pkl_file {pkl_name}...')
        pkl_out_path = os.path.join(args.pkl_token_dir, pkl_name)
        source = pd.read_pickle(pkl_out_path)
        sents_lists = source[type].values.tolist()
        # 2. output_phrases 为 token 的索引列表，output_indices 为每个 token 长度和起始索引的元组组成的列表
        output_phrases = []
        output_indices = []
        temp_phrases = []
        # 3. 获取 code_tokens 词典 vocab_token_file_path
        vocab_file_name = 'vocab_' + type + '.json'
        vocab_file_path = os.path.join(args.vocab_dir, vocab_file_name)
        out_file_name = pkl_name.split('.')[0] + '_' + type + '.h5'
        output_file_path = os.path.join(args.output_dir, out_file_name)
        print(f'Reading vocab_file {vocab_file_name}...')
        vocab = json.loads(open(vocab_file_path, 'r').readline())
        # 4. 开始序列化，并在适当时候按批次写入h5文件中
        start_index = 0
        print('Indexing...')
        print('args.index_batch: ', args.index_batch)
        pbar = tqdm(sents_lists)
        len_0 = len(sents_lists)
        if type == 'code_tokens':
            token_len = args.max_code_token_len
        else:
            token_len = args.max_desc_token_len
        batch_total_len = 0
        for i, sent in enumerate(pbar):
            real_sent_len = len(sent)
            batch_total_len += real_sent_len
            # print('real_sent_len:', real_sent_len)
            sent_len = min(real_sent_len, token_len)
            output_indices.append((token_len, start_index))
            for j in range(0, sent_len):
                word = sent[j]
                temp_phrases.append(vocab.get(word, UNK_ID))
            # 补全
            while len(temp_phrases) < token_len:
                temp_phrases.append(0)
            output_phrases.extend(temp_phrases)
            temp_phrases = []
            start_index += token_len
            if i % args.index_batch == 0:
                pbar.update(1)
                # print(f'Indexing sentence {i}...')
                if i == 0:
                    with h5py.File(output_file_path, 'w') as output_file:
                        output_file.create_dataset('phrases', data=output_phrases, maxshape=(None,), chunks=True)
                        output_file.create_dataset('indices', data=output_indices, maxshape=(None, None), chunks=True)
                        original_phrases_size = output_file['phrases'].shape[0]
                        original_indices_size = output_file['indices'].shape[0]
                else:
                    with h5py.File(output_file_path, 'a') as output_file:
                        original_phrases_size = output_file['phrases'].shape[0]
                        original_indices_size = output_file['indices'].shape[0]
                        new_phrases_maxshape = (original_phrases_size + len(output_phrases),)
                        new_indices_maxshape = (original_indices_size + len(output_indices), 2)
                        output_file['phrases'].resize(new_phrases_maxshape)
                        output_file['indices'].resize(new_indices_maxshape)
                        output_file['phrases'][
                        original_phrases_size:original_phrases_size + len(output_phrases)] = output_phrases
                        output_file['indices'][
                        original_indices_size:original_indices_size + len(output_indices)] = output_indices
                if i < len_0 - 1:
                    output_phrases = []
                    output_indices = []
                else:
                    continue

        batch_average_len = batch_total_len / (i + 1)
        print(f'After {i} batches, batch_average_len:', batch_average_len)

        # 5.处理剩余部分的数据
        print('total_len:', start_index)
        print(f'h5_file {out_file_name} will be ready soon!')
        if len(sents_lists[0]) % args.index_batch != 0:
            with h5py.File(output_file_path, 'a') as output_file:
                original_phrases_size = output_file['phrases'].shape[0]
                original_indices_size = output_file['indices'].shape[0]
                new_phrases_maxshape = (original_phrases_size + len(output_phrases),)
                new_indices_maxshape = (original_indices_size + len(output_indices), 2)
                output_file['phrases'].resize(new_phrases_maxshape)
                output_file['indices'].resize(new_indices_maxshape)
                output_file['phrases'][
                original_phrases_size:original_phrases_size + len(output_phrases)] = output_phrases
                output_file['indices'][
                original_indices_size:original_indices_size + len(output_indices)] = output_indices
        # 6.验证格式统一性
        with tables.open_file(output_file_path) as table_tokens:
            idx_tokens = table_tokens.get_node('/indices')[:]
            idx_descs = table_tokens.get_node('/indices')[:]
            assert idx_tokens.shape[0] == idx_descs.shape[0]


def get_clean_token(args):
    # total_dataset = os.listdir(args.Bitree_dir)
    total_dataset = os.listdir(args.clean_tree_dir)
    # total_dataset = os.listdir(args.clean_Bitree_dir)
    to = tqdm(total_dataset)
    for i in to:
        # if i != 'train':
        #     continue
        # dataset_path = os.path.join(args.clean_Bitree_dir, i)
        dataset_path = os.path.join(args.clean_tree_dir, i)
        # dataset_path = os.path.join(args.Bitree_dir, i)
        code_list = os.listdir(dataset_path)
        get_dataset(args, i, code_list)
        to.update(1)


def get_dataset(args, code_list_name, code_list):
    # 1.通过遍历，读取数据
    dataset_out = {'name': [], 'methodName': [], 'code_tokens': [], 'desc_tokens': []}
    dataset_path = os.path.join(args.pkl_dir, code_list_name+'.json')
    with open(dataset_path, 'r') as fj:
        dataset_source = json.load(fj)
    # 2.清洗
    tq = tqdm(code_list)
    for code_name in tq:
        index = dataset_source['name'].index(code_name)
        methodName = dataset_source['methodName'][index]
        desc_tokens = dataset_source['desc_tokens'][index]
        code_tokens = dataset_source['code_tokens'][index]
        if len(methodName) != 0:
            desc_tokens.append(methodName)

        desc_tokens = clean_token(desc_tokens, 'desc')
        code_tokens = clean_token(code_tokens, 'code')
        if len(desc_tokens) == 0:
            desc_tokens = clean_token(code_tokens, 'desc')
        # if 'function' in code_tokens:
        #     code_tokens.remove('function')
        # if 'return' in code_tokens:
        #     code_tokens.remove('return')

        dataset_out['name'].append(code_name)
        dataset_out['methodName'].append(methodName)
        dataset_out['desc_tokens'].append(desc_tokens)
        dataset_out['code_tokens'].append(code_tokens)
        tq.update(1)
    pkl_name = code_list_name + '.pkl'
    pkl_path = os.path.join(args.pkl_token_dir, pkl_name)
    df = pd.DataFrame(dataset_out,
                      columns=['name', 'methodName', 'desc_tokens', 'code_tokens',
                               'cfg_easy', 'pdg', 'mix', 'cfg'])
    pd.to_pickle(df, pkl_path)


def split_jsonl(path, code_desc_dict):
    def get_josnl(gz_files_path):
        gz_name = gz_files_path.split('/')[-1].split('.')[0]
        # 读取gz压缩文件
        with gzip.GzipFile(gz_files_path, 'r') as f:
            f_read = jsonlines.Reader(f)
            # 每个gz中压缩了多段代码片段，循环取出
            for e_idx, items in enumerate(f_read):
                name = f'f_{gz_name} e_{e_idx}'
                code = items['code']
                code_tokens = items['code_tokens']
                desc_tokens = items['docstring_tokens']
                methodName = items['func_name'].split('.')[-1]
                # if len(methodName) == 0:
                #     print(f'Processing {gz_name} ... PASS')
                #     continue
                code_desc_dict['name'].append(name)
                code_desc_dict['code'].append(code)
                code_desc_dict['methodName'].append(methodName)
                code_desc_dict['code_tokens'].append(code_tokens)
                code_desc_dict['desc_tokens'].append(desc_tokens)
            # print(f'Finish processing  {gz_name} ! {e_idx} data loaded !')

    get_josnl(path)
    return code_desc_dict


def get_name_from_dot(args):
    # 读取jsonl
    json_files = os.listdir(args.jsonl_dir)
    pkl_data_dict = {'name': [], 'methodName': []}
    for json_file in tqdm(json_files):
        json_path = os.path.join(args.jsonl_dir, json_file)
        gz_name = json_path.split('/')[-1].split('.')[0]
        # 读取gz压缩文件
        with gzip.GzipFile(json_path, 'r') as f:
            f_read = jsonlines.Reader(f)
            # 每个gz中压缩了多段代码片段，循环取出
            for e_idx, items in enumerate(f_read):
                name = f'f_{gz_name}__e_{e_idx}'
                methodName = items['func_name'].split('.')[-1]
                pkl_data_dict['name'].append(name)
                pkl_data_dict['methodName'].append(methodName)
    total_length = len(pkl_data_dict['name'])
    print(f'Total data length: {total_length}')
    # 读取dot文件夹，创建split_dict
    dot_file_folders = os.listdir(args.dot_dir)
    for file_folder in dot_file_folders:
        split_dict = {}
        file_folder_path = os.path.join(args.dot_dir, file_folder)
        dot_files = os.listdir(file_folder_path)
        for dot_file in tqdm(dot_files):
            index = pkl_data_dict['name'].index(dot_file)
            method_name = pkl_data_dict['methodName'][index]
            split_dict[dot_file] = method_name
        # 写入
        os.makedirs(args.pkl_dir, exist_ok=True)
        json_file_name = file_folder + '.json'
        json_file_path = os.path.join(args.pkl_dir, json_file_name)
        with open(json_file_path, 'w') as fw:
            json.dump(split_dict, fw)


def generate_pkl_json(args):
    # 读取原始数据
    json_files = os.listdir(args.source_dir)
    for dataset_file in json_files:
        source_dir = os.path.join(args.source_dir, dataset_file)
        if len(os.listdir(source_dir)) <= 0:
            continue
        if dataset_file == 'test':
            continue
        source_code_path = os.path.join(source_dir, dataset_file + '_originalcode')
        source_desc_path = os.path.join(source_dir, 'javadoc.original')
        dataset_source = {'name': [], 'code': [], 'methodName': [], 'code_tokens': [], 'desc_tokens': []}
        with open(source_desc_path, 'r') as f_desc:
            desc = f_desc.readlines()
        with open(source_code_path, 'r') as f_code:
            g_lines = f_code.readlines()
        # 初步断言检查
        assert len(desc) == len(g_lines), 'data length error(desc and code)'
        for idx, source_line in tqdm(enumerate(g_lines)):
            if args.language == 'python':
                clean_result = py_code_cleaner(source_line)
                if clean_result == 0:
                    print(f'ERROR in {idx} in {dataset_file}, code: {source_line}')
                    return
                source_line = clean_result
            code_token = tokenize(source_line)
            desc_token = tokenize(desc[idx])
            method_name = ''
            if args.language == 'python':
                match_result = re.findall(r'def\s*(\w+)\(.*?\)', source_line)
                if match_result:
                    method_name = match_result[0]
                else:
                    print('method name not found')
            # 保存
            name = f'{dataset_file}__{idx}___{method_name}'
            dataset_source['methodName'].append(method_name)
            dataset_source['name'].append(name)
            dataset_source['code'].append(source_line)
            dataset_source['code_tokens'].append(code_token)
            dataset_source['desc_tokens'].append(desc_token)
        # 断言检查
        assert len(dataset_source['name']) == len(dataset_source['code']) == len(dataset_source['methodName']) \
               == len(dataset_source['code_tokens']) == len(dataset_source['desc_tokens']), 'data length error'
        # 保存到pkl
        pkl_name = dataset_file + '.pkl'
        pkl_path = os.path.join(args.pkl_dir, pkl_name)
        df = pd.DataFrame(dataset_source, columns=['name', 'code', 'methodName', 'code_tokens', 'desc_tokens'])
        pd.to_pickle(df, pkl_path)
        # 保存到json
        json_file_name = dataset_file + '.json'
        json_path = os.path.join(args.pkl_dir, json_file_name)
        with open(json_path, 'w') as f_json:
            json.dump(dataset_source, f_json)


def rewrite_fix_code(args, fix_code_dir):
    json_files = os.listdir(fix_code_dir)
    for dataset_file in tqdm(json_files):
        fix_code_path = os.path.join(fix_code_dir, dataset_file, f'{dataset_file}.py')
        with open(fix_code_path, 'r') as f_code:
            source_line = f_code.read()
        clean_result = py_code_cleaner(source_line)
        with open(fix_code_path, 'w', encoding='utf-8') as f:
            f.write(clean_result)


def get_code_folder(args):
    if args.language == 'python':
        exe = '.py'
    else:   # args.language == 'java'
        exe = '.java'
    # 文件夹目录
    json_list = []
    for json_files in os.listdir(args.pkl_dir):
        if json_files.endswith('.json'):
            json_list.append(json_files)
    for json_files in json_list:
        dataset_folder = json_files.split('.json')[0]
        json_path = os.path.join(args.pkl_dir, json_files)
        with open(json_path, 'r') as f:
            code_lists = json.load(f)
            # 代码输出到文件夹
            for i, output_filename in tqdm(enumerate(code_lists['name'])):
                code_folder_path = os.path.join(args.code_dir, dataset_folder, output_filename)
                os.makedirs(code_folder_path, exist_ok=True)
                code_path = os.path.join(code_folder_path, output_filename+exe)
                output_code = code_lists['code'][i]
                if args.language == 'java':
                    output_code = 'public class main{' + output_code + '}'
                # 创建并写入函数内容到单独文件
                with open(code_path, 'w', encoding='utf-8') as f:
                    f.write(output_code)


def adjust_tree(args):
    total_dataset = os.listdir(args.Bitree_dir)
    for main_data in total_dataset:
        Bitree_folder_path = os.path.join(args.Bitree_dir, main_data)
        Bitree_list = os.listdir(Bitree_folder_path)
        clean_tree_folder_path = os.path.join(args.clean_tree_dir, main_data)
        clean_tree_list = os.listdir(clean_tree_folder_path)
        if len(clean_tree_list) == len(Bitree_list):
            continue
        tr = tqdm(clean_tree_list)
        for clean_tree in tr:
            tr.update(1)
            if clean_tree not in Bitree_list:
                # pdb.set_trace()
                delete_clean_tree_path = os.path.join(args.clean_tree_dir, main_data, clean_tree)
                to_remove(delete_clean_tree_path)


def generate_ast(args):
    # total_dataset = os.listdir(args.clean_Bitree_dir)
    total_dataset = os.listdir(args.Bitree_dir)
    for main_data in total_dataset:
        # if main_data == 'train':
        #     continue
        total_dict = {}
        # Bitree_folder_path = os.path.join(args.clean_Bitree_dir, main_data)
        Bitree_folder_path = os.path.join(args.Bitree_dir, main_data)
        Bitree_list = os.listdir(Bitree_folder_path)
        Bitree_list_sort = sorted(Bitree_list, key=lambda x: int(re.search(r'__(\d+)___', x).group(1)))
        tr_tq = tqdm(enumerate(Bitree_list_sort))
        for tree_index, Bit in tr_tq:
            # divide
            # if tree_index > 0 and tree_index % args.tree_divide == 0:
            #     tree_batch = int(tree_index / args.tree_divide)
            #     ast_name = main_data + f'_ast_{tree_batch}.json'
            #     out_ast_path = os.path.join(args.output_dir, ast_name)
            #     with open(out_ast_path, 'w') as fw:
            #         json.dump(total_dict, fw)
            #     total_dict = {}
            # not divide
            tr_tq.update(1)
            tree_dict_folder = os.path.join(Bitree_folder_path, Bit)

            tree_dict_files = os.listdir(tree_dict_folder)
            for tree_dict_file in tree_dict_files:
                if 'ast' in tree_dict_file:
                    tree_path = os.path.join(tree_dict_folder, tree_dict_file)
                    # if not os.path.exists(tree_path):
                    #     tree_path = os.path.join(tree_dict_folder, f'train_ast.json')
                    break
                else:
                    print('error')
                    return
            with open(tree_path, 'r', encoding='utf-8') as inf:
                one_tree = json.load(inf)
                total_dict[str(tree_index)] = one_tree
        ast_name = main_data + '_ast.json'
        out_ast_path = os.path.join(args.output_dir, ast_name)
        with open(out_ast_path, 'w') as fw:
            json.dump(total_dict, fw)


def get_small_tree(args):
    total_dataset = os.listdir(args.Bitree_dir)
    save_root = args.small_Bitree_dir
    for main_data in total_dataset:
        length = 0
        num = 0
        if main_data == 'train':
            continue
        Bitree_folder_path = os.path.join(args.Bitree_dir, main_data)
        save_folder_path = os.path.join(save_root, main_data)
        Bitree_list = os.listdir(Bitree_folder_path)
        tr_tq = tqdm(enumerate(Bitree_list))
        for tree_index, Bit in tr_tq:
            tr_tq.update(1)
            tree_dict_folder = os.path.join(Bitree_folder_path, Bit)
            save_dict_folder = os.path.join(save_folder_path, Bit)
            os.makedirs(save_dict_folder, exist_ok=True)
            tree_dict_files = os.listdir(tree_dict_folder)
            for tree_dict_file in tree_dict_files:
                if 'ast' in tree_dict_file:
                    tree_path = os.path.join(tree_dict_folder, tree_dict_file)
                    save_path = os.path.join(save_dict_folder, tree_dict_file)
                    break
            # with open(tree_path, 'r') as inf:
            #     content = inf.read()
            #     content = content.replace('1*NODEFIX', 'N')

            with open(tree_path, 'r') as fr:
                dict = json.load(fr)
                for i, val in dict.items():
                    for node in val.values():
                        if 'children' in node:
                            words = node['children']
                            if len(words) == 1:
                                for ws in words:
                                    single_len = len(wt(ws))
                                    num += 1
                                    length += single_len

            # with open(save_path, 'w') as fw:
            #     fw.write(content)
            print(f'{length / num:.2f}%')


def get_clean_txt(args, code_min_len, sum_min_len, cl_code=True, cl_summary=True, split_com=True):
    source_files_list = os.listdir(args.source_dir)
    for source_files in source_files_list:
        if source_files == 'test':
            continue
        source_code_token = os.path.join(args.source_dir, source_files, 'code.original_subtoken')
        source_summary = os.path.join(args.source_dir, source_files, 'javadoc.original')

        if cl_code:
            new_code_token = []
            with open(source_code_token, 'r') as f:
                token_txt = f.readlines()
                for t_txt in tqdm(token_txt):
                    filtered = easy_clean_txt(t_txt, 'code', split_function='split', minimum_word_len=code_min_len, non_stopwords=True, split_com=split_com)
                    if len(filtered) == 0:
                        new_code_token.append(t_txt)
                        continue
                    new_code_token.append(filtered + '\n')
            output_code_token = os.path.join(args.source_dir, source_files, f'clear_code_split{split_com}_{code_min_len}.original_subtoken')
            new_code_string = ''.join(new_code_token)
            with open(output_code_token, 'w') as f:
                f.write(new_code_string)

        if cl_summary:
            new_summary = []
            with open(source_summary, 'r') as f:
                summary_txt = f.readlines()
                un = 0
                for s_txt in tqdm(summary_txt):
                    filtered = easy_clean_txt(s_txt, 'desc', split_function='split', minimum_word_len=sum_min_len, non_stopwords=True, split_com=split_com)
                    txt_len = len(filtered)
                    if txt_len == 0:
                        s_txt = re.sub('.', '', s_txt)
                        new_summary.append(s_txt)
                        un += 1
                        continue
                    if '.' != filtered[-1]:
                        if filtered[-1] in [',', ';']:
                            filtered = filtered[:-1]
                        new_summary.append(filtered + ' . \n')
                    else:
                        new_summary.append(filtered + '\n')
            # print(un)
            output_summary = os.path.join(args.source_dir, source_files, f'clear_javadoc_split{split_com}_{sum_min_len}_nstop.original')
            new_summary_string = ''.join(new_summary)
            with open(output_summary, 'w') as f:
                f.write(new_summary_string)


traveled_node = []
def preorder_traversal(node_id, tree_dict):
    try:
        if node_id not in tree_dict.keys():
            return ''
        node = tree_dict[node_id]
        traveled_node.append(node['id'])
        tokens = [node['value']]  # 首先获取当前节点的value作为token
        # 如果节点有子节点，则递归访问每个子节点
        if 'children' in node:
            for child_id in node['children']:
                if child_id not in traveled_node:
                    tokens.extend(preorder_traversal(child_id, tree_dict))
        return tokens
    except Exception as e:
        print(tree_dict)


def get_multi_text(model_type, folder, args, check_order, min_len=2, clear_tree=True):
    if model_type in ['cfg', 'pdg', 'ast']:
        clean_graph_dir = args.tree_dir
        source_files = os.listdir(clean_graph_dir)
        for i, json_file in enumerate(source_files):
            new_token = []
            if folder != 'all':
                if json_file != folder:
                    continue
            the_json_path = os.path.join(clean_graph_dir, json_file)
            if not os.path.isdir(the_json_path):
                continue
            json_folder_list = os.listdir(the_json_path)
            sorted_list = sorted(json_folder_list, key=lambda x: int(re.search(r'__(\d+)___', x).group(1)))
            for json_folder in tqdm(sorted_list):
                method_name = json_folder.split('___')[1]
                json_path = os.path.join(the_json_path, json_folder, json_folder + f'_{model_type}.json')
                if not os.path.exists(json_path):
                    print(f'Processing {json_folder} file in {json_file} ERROR!!!')
                    return
                f_string = method_name + ' '
                with open(json_path, 'r') as f:
                    pre_tree = json.load(f)
                    for single_dict in pre_tree:
                        new_token_words = single_dict['value']
                        clean_t = clean_server(new_token_words, 'code', split_function='split', minimum_word_len=min_len, non_stopwords=True, split_com=False)
                        if min_len == 1:
                            clean_t = get_several_words(clean_t, min_len, 'def', '')
                        single_dict['value'] = ' '.join(clean_t)
                    model_tree = []
                    value_tree = []
                    for i_node in pre_tree:
                        if clear_tree:
                            if i_node['value'] == 'tmp':
                                continue
                        i_node['value'] = i_node['value'].replace('\n', ' ').replace('...', ' ')
                        model_tree.append(i_node)
                        value_tree.append(i_node['value'])
                    if check_order == 'for':
                        f_string += ' '.join(value_tree)
                        clean_f_string = easy_clean_txt(f_string, 'code')
                    elif check_order == 'pre_order':
                        tree_dict = {node['id']: node for node in model_tree}
                        root_id = None
                        all_children_ids = {child_id for node in model_tree if 'children' in node
                                            for child_id in node['children']}
                        for node in model_tree:
                            if node['id'] not in all_children_ids:
                                root_id = node['id']
                                break
                        result_tokens = preorder_traversal(root_id, tree_dict)
                        traveled_node.clear()
                        result_string = ' '.join(result_tokens)
                        f_string = f_string + result_string
                        clean_f_string = easy_clean_txt(f_string, 'code', minimum_word_len=3)
                    if len(clean_f_string) == 0:
                        new_token.append(f_string + '\n')
                        continue
                new_token.append(clean_f_string + '\n')
            output_path = os.path.join(args.source_dir, json_file, f'clear_{model_type}_{min_len}.{check_order}_subtoken')
            new_code_string = ''.join(new_token)
            with open(output_path, 'w') as f:
                f.write(new_code_string)
    elif model_type == 'mix':
        data_source_dir = args.source_dir
        source_files = os.listdir(data_source_dir)
        for i, json_file in enumerate(source_files):
            new_token = []
            if folder != 'all':
                if json_file != folder:
                    continue
            the_json_path = os.path.join(data_source_dir, json_file)
            if not os.path.isdir(the_json_path):
                continue
            cfg_path = os.path.join(the_json_path, f'cfg.{check_order}_subtoken')
            pdg_path = os.path.join(the_json_path, f'pdg.{check_order}_subtoken')
            assert os.path.isfile(cfg_path) and os.path.isfile(pdg_path), 'CFG or PDG File does not exist'
            with open(cfg_path, 'r') as f:
                cfg_list = f.readlines()
            with open(cfg_path, 'r') as f:
                pdg_list = f.readlines()
            assert len(cfg_list) == len(pdg_list), 'Length alignment error in CFG or PDG Files'
           # mix_list



def parse_args():
    parser = argparse.ArgumentParser("Parse data...")
    # 语言选择
    lang_choice = 'python'
    parser.add_argument('--language', type=str, default=lang_choice)
    root_dir = f'/home/user/PY_Projects/lmc/code_summary/datas/{lang_choice}/'
    # 根路径
    parser.add_argument('--root_dir', type=str, default=root_dir)
    # 原始数据的路径
    parser.add_argument('--source_dir', type=str, default=root_dir + 'source')
    # 中间结果的路径
    parser.add_argument('--pkl_dir', type=str, default=root_dir + 'mid_output/pkl')
    parser.add_argument('--pkl_token_dir', type=str, default=root_dir + 'mid_output/pkl_token')
    parser.add_argument('--code_dir', type=str, default=root_dir + 'mid_output/code')
    parser.add_argument('--dot_dir', type=str, default=root_dir + 'mid_output/dot')
    parser.add_argument('--tree_dir', type=str, default=root_dir + 'mid_output/tree')
    parser.add_argument('--clean_tree_dir', type=str, default=root_dir + 'mid_output/clean_tree')
    parser.add_argument('--clean_graph_dir', type=str, default=root_dir + 'mid_output/clean_graph')
    parser.add_argument('--Bitree_dir', type=str, default=root_dir + 'mid_output/Bitree')
    parser.add_argument('--clean_Bitree_dir', type=str, default=root_dir + 'mid_output/clean_Bitree')
    parser.add_argument('--small_Bitree_dir', type=str, default=root_dir + 'mid_output/small_Bitree')
    parser.add_argument('--temp_tree_dir', type=str, default=root_dir + 'mid_output/temp_tree')
    parser.add_argument('--temp_tree_path', type=str,
                        default=root_dir + 'mid_output/temp_tree/temp_tree.json')
    parser.add_argument('--temp_sorted_tree_path', type=str,
                        default=root_dir + 'mid_output/temp_tree/temp_sorted_tree.json')
    # 最终结果的路径
    parser.add_argument('--output_dir', type=str, default=root_dir + 'final_output/dataset')
    parser.add_argument('--vocab_dir', type=str, default=root_dir + 'final_output/vocab')
    # 手动划分数据集设置(废除)
    parser.add_argument('--test_num', type=int, default=50)
    parser.add_argument('--valid_num', type=int, default=50)
    parser.add_argument('--train_num', type=int, default=100)
    # 词表长度和序列长度设置
    parser.add_argument('--max_code_token_len', type=int, default=200)  # 目前code_token平均长度67.5，desc_token平均长度9.5
    parser.add_argument('--max_desc_token_len', type=int, default=20)  # 目前code_token平均长度67.5，desc_token平均长度9.5
    parser.add_argument('--vocab_max_len_code_token', type=int, default=80000)  # 目前desc:22575;code:32035
    parser.add_argument('--vocab_max_len_desc_token', type=int, default=80000)  # 目前desc:22575;code:32035
    parser.add_argument('--vocab_max_len_graph_token', type=int, default=200000)
    parser.add_argument('--vocab_max_len_ast_token', type=int, default=400000)
    parser.add_argument('--index_batch', type=int, default=100)
    parser.add_argument('--tree_divide', type=int, default=50000)

    return parser.parse_args()

'''
核心流程：
1.batch
生成code-生成dot-生成fix_dot-生成tree-生成fix_tree-合并tree和fix_tree-生成其他
2.全部
同样流程
'''
if __name__ == '__main__':
    '''TEST'''

    '''RUN'''
    args = parse_args()
    code_token_special_vocab_1 = r'<>=+-*/%^'
    del_key = ['.', ',', '\\', "'", '"', '?', '!', ':', ';', '…', '，', '。', '；', '：', '‘', '“', '{', '}', '[',
               ']', '(', ')', '|', '\\\\', '\n', '\'\'', '\t', ' ', '@', '#', '$', '&', '-', '—', '`', '~', '～']
    del_words = ['var', 'tmp', 'self', 'const', 'function', 'str', 'int', 'from', 'to', 'import']
    del_words = []
    lowered_java_basic_vocab = ['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class',
                                'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global',
                                'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise',
                                'return', 'try', 'while', 'with', 'yield']
    '''clean summary/code_tokens'''
    get_clean_txt(args, code_min_len=2, sum_min_len=2, cl_code=False, cl_summary=True, split_com=False)
    '''generate pkl/code'''
    # generate_pkl_json(args)
    # get_code_folder(args)
    '''generate dot'''
    # parse code
    # check_delete()
    # find_sth()
    fix_dir = r'/home/user/PY_Projects/lmc/code_summary/data/mid_output/code/fix_dev'
    # rewrite_fix_code(args, fix_dir)
    # parse code
    # copy_success_dot()
    # 重命名 unsolved_file_list
    # ...
    '''generate pkl from dot'''
    # get_name_from_dot(args)
    '''generate tree'''
    # get_json_ast(args.dot_dir, args.tree_dir, args.language, args.pkl_dir)  # practise1
    '''generate clean_tree'''
    # get_clean_tree(args, del_words)
    # get_clean_graph(args)
    '''generate multi-model text'''
    # get_multi_text('cfg', 'train', args, 'pre_order')
    # get_multi_text('cfg', 'all', args, 'pre_order', min_len=2)
    # get_multi_text('ast', 'all', args, 'pre_order')
    get_multi_text('ast', 'all', args, 'pre_order', min_len=2)
    # get_multi_text('pdg', 'all', args, 'pre_order')
    # get_multi_text('pdg', 'all', args, 'pre_order', min_len=2)
    '''generate Bitree'''
    # get_tree(args.clean_tree_dir, args.Bitree_dir, args.temp_tree_dir, args.temp_tree_path, args.temp_sorted_tree_path)   # practise2
    # get_tree(args.tree_dir, args.Bitree_dir, args.temp_tree_dir, args.temp_tree_path, args.temp_sorted_tree_path)   # practise2
    '''adjust size'''
    # adjust_tree(args)
    '''generate ast_vocab'''
    # generate_vocab(args, 'ast', [], [])
    '''generate small_Bitree (if necessary)'''
    # get_small_tree(args)
    '''generate ast'''
    # generate_ast(args)

    '''generate useful_token_pkl'''
    # get_clean_token(args)
    '''generate token_vocab'''
    # generate_vocab(args, 'code_tokens', [], '')
    # generate_vocab(args, 'desc_tokens', '', [])
    '''generate token'''
    # generate_h5(args, 'code_tokens')
    # generate_h5(args, 'desc_tokens')

    '''generate graph_vocab'''
    # generate_vocab(args, 'mix', [], [])
    '''generate graph'''
    # generate_json(args, 'cfg')
    # generate_json(args, 'pdg')
    # generate_mix_json(args)
