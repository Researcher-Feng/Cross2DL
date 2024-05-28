def data_setting(language):
    conf = {
        'src': 'clear_code_20.original_subtoken',
        'cfg': 'clear_code_20.original_subtoken',
        'pdg': '',
        'mix': 'mix.for_subtoken',
        'ast': 'ast.for_subtoken',
        'tgt': 'clear_javadoc_20.original',
        'src_tag': '',
        'code_tag_type': 'subtoken',
        'uncase': True,
        'max_characters_per_token': 30
    }
    if language == 'python':
        conf['max_src_len'] = 400
        conf['max_tgt_len'] = 30
        conf['src_vocab_size'] = 50000
        conf['tgt_vocab_size'] = 30000
    elif language == 'java':
        conf['max_src_len'] = 600
        conf['max_tgt_len'] = 50
        conf['src_vocab_size'] = 50000
        conf['tgt_vocab_size'] = 30000
    else:
        raise Exception('Dataset name not supported. Available languages: python, java. ')
    return conf
