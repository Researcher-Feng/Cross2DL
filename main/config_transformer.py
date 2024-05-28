def transformer_setting():
    conf = {
        'model_type': 'transformer',
        'emsize': 512,
        'rnn_type': 'LSTM',
        'nhid': 200,
        'bidirection': True,
        'nlayers': 6,
        'use_all_enc_layers': False,

        'src_pos_emb': False,
        'tgt_pos_emb': True,
        'max_relative_pos': [32],
        'use_neg_dist': True,
        'd_ff': 2048,
        'd_k': 64,
        'd_v': 64,
        'num_head': 8,
        'trans_drop': 0.2,
        'layer_wise_attn': False,

        'use_src_char': False,
        'use_tgt_char': False,
        'use_src_word': True,
        'use_tgt_word': True,
        'n_characters': 260,
        'char_emsize': 16,
        'filter_size': 5,
        'nfilters': 100,

        'copy_attn': True,
        # 'share_decoder_embeddings': True,
        'share_decoder_embeddings': False,
        'split_decoder': False,

        'dropout_emb': 0.2,
        'dropout_rnn': 0.2,
        'dropout': 0.2,
        'fix_embeddings': False,

        'attn_type': 'general',
        'coverage_attn': False,
        'review_attn': False,
        'force_copy': False,
        'reuse_copy_attn': False,
        'reload_decoder_state': None,
        'conditional_decoding': False,
    }
    return conf

