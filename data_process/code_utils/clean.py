import os
import re
import json
from difflib import SequenceMatcher
import jieba
import string
import urllib.parse, http.client
from collections import Counter
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
# import nltk
# nltk.download('punkt')
import nltk
from nltk import word_tokenize as wt
from nltk.corpus import stopwords
from nltk.corpus import wordnet
wnl = nltk.stem.WordNetLemmatizer()
import spacy
nlp = spacy.load('en_core_web_sm')

MINIMUM_COMPOUND_WORD_LEN = 6
STANDARD_COMPOUND_WORD_LEN = 16
STANDARD_CODE_SEQUENCE_LEN = 3
THRESHOLD = 0.6


def tokenize(input_string):
    return wt(input_string)


# 创建停用词列表
def stopwordslist():
    # stopwords = [line.strip() for line in open('stop_words.txt', encoding='UTF-8').readlines()]
    my_stopwords = ['的', '得', '之', '啊', '某个', '某', '中', '一', '一个', '后', '之后', '前', '之前']
    nltk_stopwords_chinese = stopwords.words('chinese')
    my_stopwords.extend(nltk_stopwords_chinese)
    return my_stopwords
# 对句子进行中文分词
def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    sentence_depart = jieba.cut(sentence.strip())
    # 创建一个停用词列表
    stop_words = stopwordslist()
    # 输出结果为outstr
    out_str = []
    # 去停用词
    for word in sentence_depart:
        if word not in stop_words:
            if word != '\t':
                out_str.append(word)
    return out_str
def ch_to_en(words, species="json"):
    seq = seg_depart(words)
    trans_seq = []
    for q in seq:
        if re.match(r'[a-zA-Z_0-9-+*=/<>&|.!]', q):
            trans_seq.append(q)
            continue
        q = urllib.parse.quote(q)
        myurl = "/api/dictionary.php?w=" + q + "&type=" + species + "&key=54A9DE969E911BC5294B70DA8ED5C9C4"  # 开发者Key
        try:
            httpClient = http.client.HTTPConnection('dict-co.iciba.com')
            httpClient.request('GET', myurl)
            # response是HTTPResponse对象
            response = httpClient.getresponse()  # 获取返回结果
            result = json.loads(response.read().decode("utf-8"))
            if 'parts' in result['symbols'][0]:
                trans_word = result['symbols'][0]['parts'][0]['means'][0]['word_mean']
                trans_word_wt = wt(trans_word)
                if len(trans_word_wt) != 1:
                    for div in trans_word_wt:
                        trans_word = re.sub(' ', '', div)
                        trans_word = re.sub(r'[^a-z]', '', trans_word.lower())
                        if len(trans_word) != 0:
                            trans_seq.append(trans_word)
                else:
                    trans_word = re.sub(' ', '', trans_word)
                    trans_word = re.sub(r'[^a-z]', '', trans_word.lower())
                    if len(trans_word) != 0:
                        trans_seq.append(trans_word)
            else:
                trans_seq.append('error400')
        except Exception as e:
            print(e)
            trans_seq.append('error400')
        finally:
            if httpClient:
                httpClient.close()
    new_trans_seq = []
    for seq in trans_seq:
        if seq != 'error400':
            new_trans_seq.append(seq)
    return new_trans_seq
def is_non_english(words):
    pattern_cn = r'[\u4e00-\u9fff]+'  # 匹配中文的正则表达式模式
    return_trans = []
    for word in words:
        if len(word) > 0:
            char = word[0]
            if (char in string.ascii_letters or char in '0123456789-+*!=/%<>&|,.\'\";:[]{}()`~@#$_\\') and len(return_trans) == 0:
                return words
            else:
                # print('non eng')
                result_cn = re.search(pattern_cn, char)
                if result_cn:
                    trans = ch_to_en(word)
                    return_trans.extend(trans)
                else:
                    return_trans.append(word)
    if len(return_trans) == 0:
        return words
    else:
        return return_trans


def to_lower(value):
    return value.lower()


def camel_case_split_underscore(identifier):
    # 1.长度小于等于1，直接返回
    if len(identifier) <= 1:
        return [to_lower(identifier)]
    # 2.使用正则表达式匹配驼峰命名的单词，如果没有任何匹配，low_word为原词，否则，low_word为匹配词列表；然后，low_word再拿去作下划线分割，返回最终结果
    else:
        words = re.findall(r'[a-z]+|[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', identifier)
        if len(words) <= 0:
            low_word = [identifier]
        else:
            low_word = words
        # 使用下划线分割字符串，并去除首尾空格
        components = []
        for i, j in enumerate(low_word):
            identifiers = j.split('_')
            for identifier in identifiers:
                components.append(to_lower(identifier))
        return components


def clean_token(pre_tokens_list, token_type, minimum_word_len=2, non_stopwords=True, split_com=True):
    cleaned_token_list = []
    output_token_list = []
    if non_stopwords:
        stop_words = stopwords.words('english')
    # 预处理
    tokens_list = []
    for pre_tok in pre_tokens_list:
        if '/' in pre_tok:
            if pre_tok.count('/') >= 3:
                tokens_list.append('url')
        else:
            tokens_list.append(pre_tok)
    # 分类处理
    if token_type == 'desc':
        # 翻译(目前是金山词霸api，中英互译)
        trans_tokens_list = is_non_english(tokens_list)
        if minimum_word_len:
            # 过滤单字符
            for voc_key in trans_tokens_list:  # voc_key 是一个词
                # if len(voc_key) <= 1 and voc_key != 'a':
                if len(voc_key) <= 1 and voc_key not in 'a;{},.':
                    continue
                else:
                    cleaned_token_list.append(voc_key)
        else:
            cleaned_token_list = trans_tokens_list
    else:
        # 过滤特殊字符
        cleaned_tokens = []
        for cleaned_tok in tokens_list:
            if re.match(r'^[a-zA-Z_0-9-=+*/<>%^&|{}()\[\],.!\'\":;]+$', cleaned_tok):
            # if re.match(r'^[a-zA-Z_\-+*<>=/&|!,.]+$', cleaned_tok):
            # if re.match(r'^[a-zA-Z_\-+*<>=/&|!.]+$', cleaned_tok):
                cleaned_tokens.append(cleaned_tok)
        if minimum_word_len:
            for voc_key in cleaned_tokens:  # voc_key 是一个词
                # if len(voc_key) <= 1 and voc_key != 'a':
                if len(voc_key) <= 1 and voc_key not in 'a;{},.':
                    continue
                else:
                    cleaned_token_list.append(voc_key)
        else:
            cleaned_token_list = cleaned_tokens
    # 驼峰和下划线拆分/词干提取
    aft_lem_words = get_lemmatize(cleaned_token_list)
    # 删除停用词
    for aft in aft_lem_words:
        if non_stopwords:
            # if aft not in stop_words or aft == 'a':
            if aft not in stop_words or aft not in 'a;{},.':
                if minimum_word_len:
                    # if len(aft) <= 1 and aft != 'a':
                    if len(aft) <= 1 and aft not in 'a;{},.':
                        continue
                    else:
                        output_token_list.append(aft.replace('===', '==').replace('!==', '!='))
                else:
                    output_token_list.append(aft.replace('===', '==').replace('!==', '!='))
        else:
            output_token_list = [ele.replace('===', '==').replace('!==', '!=') for ele in aft_lem_words]
    # 将所有数字和各种运算符的格式统一
    output_token_list = filter_string_list(output_token_list, token_type)
    output_token_list = clean_number(output_token_list)
    return_list = final_check_string(output_token_list, minimum_word_len, split_com)
    # 允许置空
    return return_list


def final_check_string(output_token_list, minimum_word_len=2, split_com=True):
    result_token_list = []
    for key in output_token_list:
        # if len(key) <= 1 and key != 'a' and minimum_word_len:
        if len(key) <= 1 and key not in 'a;{},.' and minimum_word_len:
            continue
        elif len(key) > STANDARD_COMPOUND_WORD_LEN and key != 'numbercharacteristic':
            if split_com:
                if not wordnet.synsets(key):
                    temp_token_list = []
                    is_composite, prefix, suffix = find_composite_words(key)
                    if is_composite:
                        temp_token_list.append(prefix)
                        temp_token_list.append(suffix)
                    else:
                        if len(prefix) != 0:
                            max_word_list = find_max(prefix)
                            if len(max_word_list) != 0:
                                temp_token_list.extend(max_word_list)
                            else:
                                mul_result, words = find_multi_composite_words(key)
                                if mul_result:
                                    found_key = second_find(key, words)
                                    temp_token_list.extend(found_key)
                                else:
                                    final_keys = find_violent(key)
                                    found_key = second_find(key, final_keys)
                                    temp_token_list.extend(found_key)
                        else:
                            mul_result, words = find_multi_composite_words(key)
                            if mul_result:
                                found_key = second_find(key, words)
                                temp_token_list.extend(found_key)
                            else:
                                final_keys = find_violent(key)
                                found_key = second_find(key, final_keys)
                                temp_token_list.extend(found_key)
                    temp_list = get_lemmatize(temp_token_list)
                    result_list = recognize_the_same(temp_list)
                    result_token_list.extend(result_list)
                else:
                    result_token_list.append(key)
            else:
                result_token_list.append(key)
        else:
            result_token_list.append(key)
    return result_token_list


def recognize_the_same(traverse_list):
    output_list = []
    same_list = []
    if len(traverse_list) <= 1:
        return traverse_list
    for i in range(len(traverse_list)):
        if i != 0:
            word_former = traverse_list[i-1]
            word_latter = traverse_list[i]
            similarity = similar(word_former, word_latter)
            if similarity > THRESHOLD:
                if len(output_list) != 0 and output_list[-1] != '<UNK>':
                    last_ele = output_list.pop()
                    same_list.append(last_ele)
                same_list.append(traverse_list[i])
                if len(output_list) == 0 or output_list[-1] != '<UNK>':
                    output_list.append('<UNK>')
                continue
        output_list.append(traverse_list[i])
    max_same_token = find_max(same_list, strict=True)
    if '<UNK>' in output_list:
        un_index = output_list.index('<UNK>')
        output_list[un_index] = max_same_token
    if '<UNK>' in output_list:
        while '<UNK>' in output_list:
            output_list.remove('<UNK>')
        final_list = [f for f in output_list if len(f) > MINIMUM_COMPOUND_WORD_LEN]
    else:
        final_list = output_list
    return final_list


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def second_find(original_key, first_finding_list):
    if len(first_finding_list) != 0:
        max_word_list = find_max(first_finding_list)
        if len(max_word_list) != 0:
            result_list = max_word_list
        else:
            result_list = [original_key]
    else:
        result_list = [original_key]
    return result_list


def find_max(lst, strict=False):
    if len(lst) <= 1:
        return lst
    max_length = max(len(word) for word in lst)
    if max_length < MINIMUM_COMPOUND_WORD_LEN:
        return []
    if strict:
        longest_words = [word for word in lst if len(word) == max_length][0]
    else:
        longest_words = [word for word in lst if len(word) == max_length or len(word) >= STANDARD_COMPOUND_WORD_LEN]
    return longest_words


def is_valid_word(word):
    if len(word) < MINIMUM_COMPOUND_WORD_LEN:
        return False
    # 检查单词是否在单词列表中
    return wordnet.synsets(word)


def find_composite_words(string):
    fail_save = []
    for i in range(1, len(string)):
        prefix = string[:i]
        suffix = string[i:]

        # 如果当前分割的两部分都是有效的单词
        if is_valid_word(prefix) and is_valid_word(suffix):
            return True, prefix, suffix
        elif is_valid_word(prefix):
            fail_save.append(prefix)
        elif is_valid_word(suffix):
            fail_save.append(suffix)

    return False, fail_save, None


def find_multi_composite_words(string, path=None):
    if path is None:
        path = []
    if is_valid_word(string):
        return True, path + [string]
    for i in range(1, len(string)):
        prefix = string[:i]
        suffix = string[i:]
        if is_valid_word(prefix):
            result, words = find_multi_composite_words(suffix, path + [prefix])
            if result:
                return True, words
    return False, []


def find_violent(code_string):
    violent_list = []
    code_len = len(code_string)
    for i in range(MINIMUM_COMPOUND_WORD_LEN, code_len):
        if code_len - i <= 0:
            continue
        for j in range(0, code_len - i + 1):
            code_snippets = code_string[j:j + i]
            if is_valid_word(code_snippets):
                violent_list.append(code_snippets)
    return violent_list


filter_string_1 = '''------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'''
filter_string_2 = '''================================================================================================================================================================================================================================================================================================'''
filter_string_3 = '''************************************************************************************************************************************************************************************************************************************************************************************************'''
filter_string_4 = '''////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////'''
filter_string_5 = '''................................................................................................................................................................................................................................................................................................'''
filter_string_6 = '''################################################################################################################################################################################################################################################################################################'''
filter_string_7 = '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
filter_string_8 = '''────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────'''
def filter_string_list(lst, t_type):
    result = []
    for item in lst:
        if item.isnumeric():
            result.append('numbercharacteristic')
        elif item in filter_string_1 and (item != '--' and item != '-'):
            continue
        elif item in filter_string_2 and (item != '===' and item != '==' and item != '='):
            continue
        elif item in filter_string_3 and (item != '**' and item != '*'):
            continue
        elif item in filter_string_4 or '//-' in item or '//=' in item:
            continue
        elif item in filter_string_5 and item != '.':
            continue
        elif item in filter_string_6:
            continue
        elif item in filter_string_7:
            continue
        # elif item in ',:;((()))[[[]]]{{{}}}$@\\\\\\\\`' and t_type != 'desc':
        #     continue
        elif re.match(r'/\*.*?\*/', item):
            continue
        else:
            result.append(item)
    return result


def clean_number(code_list):
    result = []
    for i in range(len(code_list)):
        if code_list[i] == 'number':
            if i == 0 or code_list[i - 1] != 'number':
                result.append('number')
        else:
            result.append(code_list[i])
    return result


def get_lemmatize(source_word_list):
    aft_lem = []
    for voc_key in source_word_list:
        temp_tokens = camel_case_split_underscore(voc_key)
        aft_lem.extend(temp_tokens)
    out_aft_lem = spacy_lemmatization(aft_lem)
    return out_aft_lem


def spacy_lemmatization(word_list):
    lemmatized_list = []
    for word in word_list:
        if len(word):
            doc = nlp(word)
            if len(doc):
                lemmatized_word = doc[0].lemma_
                lemmatized_list.append(lemmatized_word)
    return lemmatized_list


def py_code_check(code):
    if 'DCNL' in code or 'DCSP' in code or '@' in code or '#' in code or 'def' not in code:
        return 0
    else:
        return 1


def finding_flag(c_code, tag, return_note):
    # 假设tag为单引号
    start_index = c_code.find(tag)
    # 有单引号
    if start_index != -1:
        flag_find = 1
        next_index = start_index + 1
        while True:
            end_index = c_code.find(tag, next_index)
            if next_index == -1 or end_index == -1:
                break
            if flag_find > 0:
                check_snippet = c_code[next_index: end_index]
                if return_note in check_snippet:
                    return 1
            next_index = end_index + 1
            flag_find = -flag_find
        # 有单引号但没有发现问题
        return -1
    # 无单引号
    else:
        return 0


# 检查引号之间的换行和匹配的问题
def check_newline(check_code):
    find_1 = finding_flag(check_code, "'", '\n')
    if find_1 == 1:
        return 1
    if find_1 == 0:
        find_2 = finding_flag(check_code, '"', '\n')
        if find_2 == 1:
            return 1
    return 0


def py_code_cleaner(source_line):
    origin_line = clean_code(source_line, 'py')
    if 'DC' in origin_line:
        pass
    if 'DCNL' in origin_line or 'DCSP' in origin_line or 'DCTB' in origin_line:
        return 0
    if 'def' not in origin_line:
        return 0
    return origin_line


def check_a_pattern(code):
    if '@' not in code:
        return 0
    # 定义正则表达式匹配模式
    pattern = r'([^\'"])([^\'"]*?)(@)([^\'"]*?)([^\'"])'
    # 使用findall()方法找到所有匹配的字符串
    matches = re.findall(pattern, code)
    # 判断是否存在每对引号之外的内容包含字符'@'
    a_flag = 0
    for match in matches:
        if '@' in match[1] or '@' in match[3]:
            a_flag = 1
            print(f"存在引号之外的内容包含'@': {match[1]}{match[2]}{match[3]}")
    # a_pattern = r'[^\'"]@'
    # # 使用search()方法匹配出第一个引号外的内容中是否包含@字符
    # match_a = re.search(a_pattern, code)
    if a_flag == 0:
        return 0
    else:
        quote_pattern_single = r"([^'])(@.*?\n)([^'])"
        quote_pattern_double = r'([^"])(@.*?\n)([^"])'

    def replace(match):
        match_1 = match.group(1)
        match_3 = match.group(3)
        return match_1 + match_3
    # 使用sub()方法替换引号之外的部分
    code = re.sub(quote_pattern_single, replace, code)
    result = re.sub(quote_pattern_double, replace, code)
    return result


def clean_code(codes, code_type):
    code = codes
    if code_type == 'py':
        code = re.sub(r'DCNL(\s+)DCSP(\s+)', '\n  ', code)
        code = re.sub(r'DCSP(\s+)', '  ', code)
        code = re.sub(r'DCTB(\s+)', '\t', code)
        code = re.sub(r'DCNL(\s+)', '\n', code)
        code = re.sub(r'@', '#', code)
        code = re.sub(r'`', '', code)
        code = re.sub(r'(\d+)L', r'\1', code)
        # 还需要，方法形参列表的括号内不能有括号
        code = code_summary_pycode_repair(code)
    elif code_type == 'json':
        # 删除注释
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'@.*$', '', code, flags=re.MULTILINE)
        # 匹配并删除"/**/"之间的内容
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # 删除换行，多余空格和括号
        code = re.sub(r'\s+', ' ', code, flags=re.MULTILINE)
        # 替换r''引号里面的内容为空字符
        # code = re.sub(r'\'.*?\'', '\'str\'', code)
    elif code_type == 'java':
        # 删除注释
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'@.*$', '', code, flags=re.MULTILINE)
        # 匹配并删除"/**/"之间的内容
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

        if 'public class main{' not in code:
            code = 'public class main{ ' + code + '}'
        code = re.sub(r'Nonnull|Nullable', '', code, flags=re.MULTILINE)
        code = re.sub(r'{(\w+)\s+public', '{public', code, flags=re.MULTILINE)
        code = re.sub(r'{(\w+)\s+private', '{private', code, flags=re.MULTILINE)
        code = re.sub(r'{(\w+)\s+protected', '{protected', code, flags=re.MULTILINE)
        code = re.sub(r'(\w+)\s+final', 'final', code, flags=re.MULTILINE)
        code = re.sub(r'(\w+)\s+default', 'default', code, flags=re.MULTILINE)
        # 替换r''引号里面的内容为空字符
        code = re.sub(r"'", '"', code)
        # 删除换行，多余空格和括号
        code = re.sub(r'\s+', ' ', code, flags=re.MULTILINE)

    code = code.strip()
    return code


def easy_clean_txt(text, process_type, split_function='split', minimum_word_len=2, non_stopwords=True, copy_short=0, split_com=True):
    processing_list = clean_server(text, process_type, split_function, minimum_word_len, non_stopwords, split_com)
    if copy_short:
        if len(processing_list) < copy_short:
            processing_list.extend(processing_list)
    processed_list = []
    for x in processing_list:
        if minimum_word_len:
            # if isinstance(x, str) and (len(x) >= minimum_word_len or x == 'a'):
            if isinstance(x, str) and (len(x) >= minimum_word_len or x in 'a;{},.'):
                processed_list.append(x)
        else:
            if isinstance(x, str) and len(x) > 0:
                processed_list.append(x)
    if len(processed_list) < STANDARD_CODE_SEQUENCE_LEN:
        processed_list = processing_list
    p_precessed = clean_multiple_pun(processed_list)
    processed_text = ' '.join(p_precessed)
    # print(filter_txt)
    # print(processed_text)
    return processed_text


def clean_multiple_pun(text_list):
    new_list = []
    previous_comma = False
    for idx, item in enumerate(text_list):
        if item in [',', ';', '.']:
            if previous_comma:
                continue
            else:
                if idx == len(text_list) - 1:
                    new_list.append('.')
                    continue
                new_list.append(',')
                previous_comma = True
        else:
            new_list.append(item)
            previous_comma = False
    return new_list


def clean_server(text, process_type, split_function, minimum_word_len, non_stopwords, split_com):
    text = re.sub('[\n]+', '', text)
    if split_function == 'sub':
        filter_txt = re.sub(r'[^a-zA-Z0-9\s\t]', ' ', text)
        process_list = wt(filter_txt)
    else:
        process_list = text.split()
    processing_list = clean_token(process_list, process_type, minimum_word_len, non_stopwords, split_com)
    return processing_list


def code_summary_pycode_repair(code):
    # code = code.replace('\\\\', '\\')
    # code = code.replace('\\n', '\n')
    code = code.replace('\\x1b[0m\n")', '')
    # code = re.sub('\x.*\)', '', code)
    return code


def remove_newline(match):
    code = match.group(0)
    code = code.replace('\n\t', '\n')
    code = code.replace('\n ', '\n')
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


def code_summary_pycode_string_repair(code):
    code = re.sub(r"'([^']*)'", remove_newline, code, flags=re.DOTALL)
    return code


def gain_code_mn(no_comment_code, methodName, code_tokens, code_type):
    if code_type == 'json':
        # 提取代码中的方法名
        method_name = ''
        method_content = ''
        pattern = r"function\s*(\w+)?([\s\S]*)"
        matches = re.findall(pattern, no_comment_code)
        for match in matches:
            method_name = match[0]
            method_content = match[1]
            break
        # 如果方法名中methodName为空
        if len(methodName) == 0:
            # 如果代码中method_name不为空
            if method_name:
                methodName = method_name
            # 如果代码中method_name为空
            else:
                # 提取code_token中的方法名
                # 清洗code_tokens
                if 'function' in code_tokens:
                    code_tokens.remove('function')
                if 'return' in code_tokens:
                    code_tokens.remove('return')
                code_tokens = clean_token(code_tokens, 'code')
                code_tokens_info = Counter(code_tokens)
                for i, item in enumerate(code_tokens_info.most_common()):
                    if item[0].isalpha():
                        methodName = item[0]
                        method_name = item[0]
                        no_comment_code = 'function ' + method_name + method_content
                        break
    return methodName, no_comment_code


def is_folder_exists(folder_path):
    exist = os.path.isdir(folder_path)
    if exist:
        print(f"文件夹 {folder_path} 已存在")
    return exist


def sort_mix_method(tree):
    sorted_tree = sorted(tree, key=lambda x: x["old_id"])
    for i, d in enumerate(sorted_tree):
        d['id'] = i
        del (d['old_id'])
    return sorted_tree


def sort_method(tree, model_type):
    sorted_tree = sorted(tree, key=lambda x: x["id"])
    for i, d in enumerate(sorted_tree):
        d['new_id'] = i
    for i, d in enumerate(sorted_tree):
        if 'children' in d:
            children = []
            for child_id in d["children"]:
                for child in sorted_tree:  # 使用 sorted_tree 进行索引查找
                    if child["id"] == child_id:
                        children.append(child['new_id'])
                        break
            d['children'] = children
    for dict in sorted_tree:
        if 'type' in dict:
            del (dict['type'])
        modify_dict_key(dict, 'id', 'old_id')
        modify_dict_key(dict, 'new_id', 'id')
        modify_dict_key(dict, 'value', 'wordid')
        modify_dict_key(dict, 'children', f'snode_{model_type}')
    return sorted_tree


def modify_dict_key(dictionary, old_key, new_key):
    if old_key in dictionary:
        dictionary[new_key] = dictionary.pop(old_key)


def check_voca(word_string, total_vocab):
    # # 1.建立一个新列表，存放序号
    # replace = []
    # # 2.对原封不动的word_string，和建立词汇表时候一样，先用wt分词，此时能比较完整地保留语义信息
    # word_list = wt(word_string)
    # # 3.对new_list进行初筛
    # cleaned_word_list = clean_token(word_list, 'code')
    # # 4.初筛后，进行序列化
    # for lowered_word in cleaned_word_list:
    #     if lowered_word == 'empty':
    #         replace.append(0)
    #     elif lowered_word in total_vocab:
    #         replace.append(total_vocab[lowered_word])
    #     else:
    #         replace.append(1)
    # # 5.返回
    # return replace

    replace = []
    for lowered_word in word_string:
        if lowered_word in total_vocab:
            replace.append(total_vocab[lowered_word])
        else:
            replace.append(1)
    # 5.返回
    return replace


def read_vocab_from_pkl(pkl_dir, model_type, vocab_info):
    pkl_files = os.listdir(pkl_dir)
    vocab_basic_len = 10000
    vocab_rough_total_len = 10000
    vocab_len_count = 1
    for i, pkl_file in enumerate(pkl_files):
        # if pkl_file == 'train.pkl':
        #     continue
        the_pkl_path = os.path.join(pkl_dir, pkl_file)
        folder_name = the_pkl_path.split('/')[-1].split('.')[0]

        source = pd.read_pickle(the_pkl_path)
        for _, data in source.iterrows():
            new_token_words = []
            # for index, c_tok in enumerate(data[model_type]):
            #     检查当前词表，如果有特殊符号，将其删除
            #     if c_tok not in del_keys:
            #         new_token_words.append(c_tok)


            # new_words = clean_token(data[model_type])
            # new_words = clean_token(data[model_type], 'code')
            new_words = data[model_type]


            new_token_words.extend(new_words)
            vocab_info.update(new_token_words)
            vocab_temp_len = len(vocab_info)
            if vocab_temp_len > vocab_rough_total_len:
                vocab_len_count += 1
                vocab_rough_total_len = vocab_basic_len * vocab_len_count
                print('vocab_temp_len: ', vocab_temp_len)

        vocab_temp_len = len(vocab_info)
        if vocab_temp_len > 25100:
            print(vocab_info.most_common()[25000:25100])
        if vocab_temp_len > 30000:
            print(vocab_info.most_common()[29900:30000])
    print('Finish checking!')
    return vocab_info


def read_vocab_from_json(json_dir, model_type, vocab_info):
    source_files = os.listdir(json_dir)
    vocab_basic_len = 10000
    vocab_rough_total_len = 10000
    vocab_len_count = 1
    for i, json_file in enumerate(source_files):
        the_json_path = os.path.join(json_dir, json_file)
        json_folder_list = os.listdir(the_json_path)
        td = tqdm(json_folder_list)
        for json_folder in td:
            td.update(1)
            json_path = os.path.join(the_json_path, json_folder, json_folder + f'_{model_type}.json')
            json_path_second = os.path.join(the_json_path, json_folder, f'train_{model_type}.json')
            if os.path.exists(json_path_second):
                with open(json_path_second, 'r') as j:
                    cfg_dict = json.load(j)
            elif os.path.exists(json_path):
                with open(json_path, 'r') as j:
                    cfg_dict = json.load(j)
            for single_dict in cfg_dict:



                # words = wt(single_dict['value'])
                # words = [single_dict['value']]
                # vocab_info.update(words)

                # new_token_words = clean_token(words, 'code')

                new_token_words = single_dict['value']
                vocab_info.update(new_token_words)

                vocab_temp_len = len(vocab_info)
                if vocab_temp_len > vocab_rough_total_len:
                    vocab_len_count += 1
                    vocab_rough_total_len = vocab_basic_len * vocab_len_count
                    print('vocab_temp_len: ', vocab_temp_len)
                # words = wt(single_dict['value'])
                # for word in words:  # word是一个词
                #     # 检查当前词表，如果有特殊符号，将其删除
                #     new_token_words = []
                #     if not any(d_k in word for d_k in del_keys):
                #         new_token_words.append(word)
                #     vocab_info.update(new_token_words)
                #     vocab_temp_len = len(vocab_info)
                #     if vocab_temp_len > vocab_rough_total_len:
                #         vocab_len_count += 1
                #         vocab_rough_total_len = vocab_basic_len * vocab_len_count
                #         print('vocab_temp_len: ', vocab_temp_len)
        vocab_temp_len = len(vocab_info)
        if vocab_temp_len > 40000:
            print(vocab_info.most_common()[39900:40000])
    print('Finish checking!')
    return vocab_info


# def read_vocab_from_ast(json_dir, save_dir, vocab_info):
#     source_files = os.listdir(json_dir)
#     vocab_basic_len = 10000
#     vocab_rough_total_len = 10000
#     vocab_len_count = 1
#     for i, json_file in enumerate(source_files):
#         the_json_path = os.path.join(json_dir, json_file)
#         json_folder_list = os.listdir(the_json_path)
#         the_save_dir = os.path.join(save_dir, json_file)
#         td = tqdm(json_folder_list)
#         for json_folder in td:
#             td.update(1)
#             the_save_path = os.path.join(the_save_dir, json_folder)
#             os.makedirs(the_save_path, exist_ok=True)
#             json_path = os.path.join(the_json_path, json_folder, json_folder + f'_ast.json')
#             save_path = os.path.join(the_save_path, json_folder + f'_ast.json')
#             j = open(json_path, 'r')
#             cfg_dict = json.load(j)
#             words = []
#             for single_dict in cfg_dict.values():
#                 if 'children' in single_dict and len(single_dict['children']) == 1:
#                     word = single_dict['children'][0]
#                     # filtered_string = re.sub(r'[^a-z0-9]', '', word.lower())
#                     filtered_string = re.sub(r'[^a-z0-9]', '', word.lower())
#                     single_dict['children'][0] = filtered_string
#                     words.append(filtered_string)
#             clean_dict = json.dumps(cfg_dict)
#             with open(save_path, 'w') as fw:
#                 fw.write(clean_dict)
#
#             vocab_info.update(words)
#             vocab_temp_len = len(vocab_info)
#             if vocab_temp_len > vocab_rough_total_len:
#                 vocab_len_count += 1
#                 vocab_rough_total_len = vocab_basic_len * vocab_len_count
#                 print('vocab_temp_len: ', vocab_temp_len)
#         vocab_temp_len = len(vocab_info)
#         if vocab_temp_len > 40000:
#             print(vocab_info.most_common()[39900:40000])
#     print('Finish checking!')
#     return vocab_info


def read_vocab_from_ast(json_dir, vocab_info):
    source_files = os.listdir(json_dir)
    vocab_basic_len = 10000
    vocab_rough_total_len = 10000
    vocab_len_count = 1
    for i, json_file in enumerate(source_files):
        the_json_path = os.path.join(json_dir, json_file)
        json_folder_list = os.listdir(the_json_path)
        td = tqdm(json_folder_list)
        for json_folder in td:
            td.update(1)
            json_path = os.path.join(the_json_path, json_folder, json_folder + f'_ast.json')
            if not os.path.exists(json_path):
                json_path = os.path.join(the_json_path, json_folder, f'train_ast.json')


            j = open(json_path, 'r')
            cfg_dict = json.load(j)
            words = []
            for single_dict in cfg_dict.values():
                if 'children' in single_dict and len(single_dict['children']) == 1:
                    word = single_dict['children'][0]
                    filtered_string = word
                    # filtered_string = re.sub(r'[^a-z0-9]', '', word.lower())
                    words.append(filtered_string)
            vocab_info.update(words)
            vocab_temp_len = len(vocab_info)
            if vocab_temp_len > vocab_rough_total_len:
                vocab_len_count += 1
                vocab_rough_total_len = vocab_basic_len * vocab_len_count
                print('vocab_temp_len: ', vocab_temp_len)
        vocab_temp_len = len(vocab_info)
        if vocab_temp_len > 40000:
            print(vocab_info.most_common()[39900:40000])
    print('Finish checking!')
    return vocab_info


# 获取词汇。频率最高优先，其次长度最大
def get_one_words(my_list):
    # 统计出现次数和长度
    count_dict = {}
    for word in my_list:
        if word in count_dict:
            count_dict[word][0] += 1
        else:
            count_dict[word] = [1, len(word)]

    max_freq = 0  # 最大的频率
    max_len = 0  # 最长的长度
    result = None  # 结果字符串

    for word, (freq, length) in count_dict.items():
        if freq > max_freq or (freq == max_freq and length > max_len):
            max_freq = freq
            max_len = length
            result = word

    return result


# 获取词汇
def get_several_words(my_list, top_num, title_word, del_words):
    result = []
    if title_word in my_list:
        result.append(title_word)
        list_len = len(my_list)
        idx = my_list.index(title_word)
        if idx + 1 < list_len:
            result.append(my_list[idx+1])
        if idx + 2 < list_len:
            result.append(my_list[idx+2])
        if idx + 3 < list_len:
            result.append(my_list[idx+3])
        if idx + 4 < list_len:
            result.append(my_list[idx+4])
        if idx + 5 < list_len:
            result.append(my_list[idx+5])
        if idx + 6 < list_len:
            result.append(my_list[idx+6])
        if idx + 7 < list_len:
            result.append(my_list[idx+7])
        return result

    my_list_2 = []
    for check_word in my_list:
        if check_word not in del_words:
            my_list_2.append(check_word)

    # counter = Counter(my_list_2)
    # # counter = Counter(my_list)
    # freq_values = list(counter.values())
    # all_equal = all(x == freq_values[0] for x in freq_values)
    # if not all_equal:
    #     top_list = counter.most_common(top_num)
    #     for word in top_list:
    #         result.append(word[0])
    # else:
    #     result.extend(my_list[:top_num])
        # top_list = sorted(my_list, key=len, reverse=True)[:top_num]
        # for word in top_list:
        #     result.append(word)

    result = my_list_2[:top_num]

    return result



'''
def py_code_cleaner(source_line):
    def_pattern = r'def\s.*?\('
    def_string = r'def '
    origin_line = clean_code(source_line, 'py')
    if 'DCNL' in origin_line or 'DCSP' in origin_line:
        return 0
    if '@' in origin_line or '#' in origin_line:
        if 'def' in source_line:
            def_matches = re.findall(def_pattern, source_line)
            if len(def_matches) >= 2:
                pattern = re.compile(r"(?=(" + re.escape(def_string) + r"))")
                matches = pattern.finditer(source_line)
                def_matches_indexes = []
                while True:
                    try:
                        match = next(matches)
                        index = match.start()
                        def_matches_indexes.append(index)
                    except StopIteration:
                        break
                match_len = len(def_matches_indexes)
                check_success = 0
                if match_len >= 2:
                    for d_idx, d_match in enumerate(def_matches_indexes):
                        first_index = def_matches_indexes[d_idx]
                        if d_idx + 1 < match_len:
                            next_index = def_matches_indexes[d_idx + 1]
                            source_code = source_line[first_index:next_index]
                        else:
                            source_code = source_line[first_index:]
                        origin_line = clean_code(source_code, 'py')
                        if py_code_check(origin_line):
                            check_success = 1
                            print(source_code)
                            print(origin_line)
                            break
                if check_success == 0:
                    origin_line = code_summary_pycode_string_repair(origin_line)
                    print(origin_line)
                    return origin_line
            if len(def_matches) <= 1:
                origin_line = code_summary_pycode_string_repair(origin_line)
                print(origin_line)
                return origin_line
    if 'def' not in origin_line:
        return 0
    return origin_line
'''

