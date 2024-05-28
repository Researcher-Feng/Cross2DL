import logging
import random
import string
from collections import Counter
from tqdm import tqdm

from c2nl.objects import Code, Summary
from c2nl.inputters.vocabulary import Vocabulary, UnicodeCharsVocabulary
from c2nl.inputters.constants import BOS_WORD, EOS_WORD, PAD_WORD, \
    UNK_WORD, TOKEN_TYPE_MAP, AST_TYPE_MAP, DATA_LANG_MAP, LANG_ID_MAP
from c2nl.utils.misc import count_file_lines

logger = logging.getLogger(__name__)


def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True


def generate_random_string(N=8):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(N))


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------

def process_examples(lang_id,
                     data_index,
                     source,
                     target,
                     max_src_len,
                     max_tgt_len,
                     code_tag_type,
                     uncase=False,
                     test_split=True):
    code_output = []
    # code_tokens_all = []
    # for source_i in source:
    #     code_tokens = source_i.split()
    #     code_tokens = code_tokens[:max_src_len]
    #     if len(code_tokens) == 0:
    #         return None
    #     code_tokens_all.extend(code_tokens)
    # code_all = Code()
    # code_all.language = lang_id
    # code_all.tokens = code_tokens_all

    for source_i in source:
        code_tokens = source_i.split()
        code_tokens = code_tokens[:max_src_len]
        code_type = []
        code_type = code_type[:max_src_len]
        TAG_TYPE_MAP = TOKEN_TYPE_MAP if \
            code_tag_type == 'subtoken' else AST_TYPE_MAP
        code = Code()
        # code.text = code_text_all
        code.text = source_i
        code.language = lang_id
        code.tokens = code_tokens
        # code.src_vocab = code_all.src_vocab
        code.type = [TAG_TYPE_MAP.get(ct, 1) for ct in code_type]
        if code_tag_type != 'subtoken':
            code.mask = [1 if ct == 'N' else 0 for ct in code_type]
        code_output.append(code)

    if target is not None:
        summ = target.lower() if uncase else target
        summ_tokens = summ.split()
        if not test_split:
            summ_tokens = summ_tokens[:max_tgt_len]
        if len(summ_tokens) == 0:
            return None
        summary = Summary()
        summary.text = ' '.join(summ_tokens)
        summary.tokens = summ_tokens
        summary.prepend_token(BOS_WORD)
        summary.append_token(EOS_WORD)
    else:
        summary = None

    example = dict()
    for idx, code_output_i in enumerate(code_output):
        example[f'rep_{idx}'] = code_output_i
    example['summary'] = summary
    return example


def load_data(args, filenames, max_examples=-1, dataset_name='java',
              test_split=False):
    """Load examples from preprocessed file. One example per line, JSON encoded."""
    with open(filenames['tgt']) as f:
        targets = [line.strip() for line in
                   tqdm(f, total=count_file_lines(filenames['tgt']))]

    rep_list = []

    if args.use_src:
        with open(filenames['src']) as f:
            rep_list.append([line.strip() for line in
                            tqdm(f, total=count_file_lines(filenames['src']))])

    if args.use_cfg:
        with open(filenames['cfg']) as f:
            rep_list.append([line.strip() for line in
                            tqdm(f, total=count_file_lines(filenames['cfg']))])

    if args.use_pdg:
        with open(filenames['pdg']) as f:
            rep_list.append([line.strip() for line in
                            tqdm(f, total=count_file_lines(filenames['pdg']))])

    if args.use_mix:
        with open(filenames['mix']) as f:
            rep_list.append([line.strip() for line in
                            tqdm(f, total=count_file_lines(filenames['mix']))])

    if args.use_ast:
        with open(filenames['ast']) as f:
            rep_list.append([line.strip() for line in
                            tqdm(f, total=count_file_lines(filenames['ast']))])

    assert len(rep_list) == 2, 'invalid rep_list'

    assert len(rep_list[0]) == len(rep_list[1]) == len(targets), 'alignment error in rep_list'

    examples = []
    rep_child_list = []
    for i_dx in range(len(rep_list)):
        rep_child_list.append([])

    for idx, tgt in tqdm(enumerate(targets), total=len(targets)):
        for i_dx, i_list in enumerate(rep_list):
            rep_child_list[i_dx] = rep_list[i_dx][idx]
        if dataset_name in ['java', 'python']:
            _ex = process_examples(LANG_ID_MAP[DATA_LANG_MAP[dataset_name]],
                                   idx,
                                   rep_child_list,
                                   tgt,
                                   args.max_src_len,
                                   args.max_tgt_len,
                                   args.code_tag_type,
                                   uncase=args.uncase,
                                   test_split=test_split)
            if _ex is not None:
                examples.append(_ex)

        if max_examples != -1 and len(examples) > max_examples:
            break

    return examples


# ------------------------------------------------------------------------------
# Dictionary building
# ------------------------------------------------------------------------------


def index_embedding_words(embedding_file):
    """Put all the words in embedding_file into a set."""
    words = set()
    with open(embedding_file) as f:
        for line in tqdm(f, total=count_file_lines(embedding_file)):
            w = Vocabulary.normalize(line.rstrip().split(' ')[0])
            words.add(w)

    words.update([BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD])
    return words


def load_words(args, examples, fields, dict_size=None):
    """Iterate and index all the words in examples (documents + questions)."""

    def _insert(iterable):
        words = []
        for w in iterable:
            w = Vocabulary.normalize(w)
            words.append(w)
        word_count.update(words)

    word_count = Counter()
    for ex in tqdm(examples):
        for field in fields:
            _insert(ex[field].tokens)

    # -2 to reserve spots for PAD and UNK token
    dict_size = dict_size - 2 if dict_size and dict_size > 2 else dict_size
    most_common = word_count.most_common(dict_size)
    words = set(word for word, _ in most_common)
    return words


def build_word_dict(args, examples, fields, dict_size=None,
                    no_special_token=False):
    """Return a dictionary from question and document words in
    provided examples.
    """
    word_dict = Vocabulary(no_special_token)
    for w in load_words(args, examples, fields, dict_size):
        word_dict.add(w)
    return word_dict


def build_word_and_char_dict(args, examples, fields, dict_size=None,
                             no_special_token=False):
    """Return a dictionary from question and document words in
    provided examples.
    """
    words = load_words(args, examples, fields, dict_size)
    dictioanry = UnicodeCharsVocabulary(words,
                                        args.max_characters_per_token,
                                        no_special_token)
    return dictioanry


def top_summary_words(args, examples, word_dict):
    """Count and return the most common question words in provided examples."""
    word_count = Counter()
    for ex in examples:
        for w in ex['summary'].tokens:
            w = Vocabulary.normalize(w)
            if w in word_dict:
                word_count.update([w])
    return word_count.most_common(args.tune_partial)
