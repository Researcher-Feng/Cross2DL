import torch


def vectorize(ex, model):
    """Vectorize a single example."""
    src_dict = model.src_dict
    tgt_dict = model.tgt_dict

    vectorized_ex = dict()

    for model_type in ex:
        if model_type == "summary":
            continue
        code = ex[model_type]
        vectorized_ex['id'] = code.id
        vectorized_ex['language'] = code.language

        # vectorized_ex[f'{model_type}_code'] = code.text
        vectorized_ex[f'{model_type}_code_tokens'] = code.tokens
        vectorized_ex[f'{model_type}_code_char_rep'] = None
        vectorized_ex[f'{model_type}_code_type_rep'] = None
        vectorized_ex[f'{model_type}_code_mask_rep'] = None
        vectorized_ex['use_code_mask'] = False

        vectorized_ex[f'{model_type}_code_word_rep'] = torch.LongTensor(code.vectorize(word_dict=src_dict))
        if model.args.use_src_char:
            vectorized_ex[f'{model_type}_code_char_rep'] = torch.LongTensor(code.vectorize(word_dict=src_dict, _type='char'))
        if model.args.use_code_type:
            vectorized_ex[f'{model_type}_code_type_rep'] = torch.LongTensor(code.type)
        if code.mask:
            vectorized_ex[f'{model_type}_code_mask_rep'] = torch.LongTensor(code.mask)
            vectorized_ex['use_code_mask'] = True
        vectorized_ex['src_vocab'] = code.src_vocab

    summary = ex['summary']
    vectorized_ex['summ'] = None
    vectorized_ex['summ_tokens'] = None
    vectorized_ex['stype'] = None
    vectorized_ex['summ_word_rep'] = None
    vectorized_ex['summ_char_rep'] = None
    vectorized_ex['target'] = None

    if summary is not None:
        vectorized_ex['summ'] = summary.text
        vectorized_ex['summ_tokens'] = summary.tokens
        vectorized_ex['stype'] = summary.type
        vectorized_ex['summ_word_rep'] = torch.LongTensor(summary.vectorize(word_dict=tgt_dict))
        if model.args.use_tgt_char:
            vectorized_ex['summ_char_rep'] = torch.LongTensor(summary.vectorize(word_dict=tgt_dict, _type='char'))
        # target is only used to compute loss during training
        vectorized_ex['target'] = torch.LongTensor(summary.vectorize(tgt_dict))

    vectorized_ex['use_src_word'] = model.args.use_src_word
    vectorized_ex['use_tgt_word'] = model.args.use_tgt_word
    vectorized_ex['use_src_char'] = model.args.use_src_char
    vectorized_ex['use_tgt_char'] = model.args.use_tgt_char
    vectorized_ex['use_code_type'] = model.args.use_code_type
    vectorized_ex['code'] = ex['rep_0'].text

    return vectorized_ex


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    # batch is a list of vectorized examples
    batch_size = len(batch)
    use_src_word = batch[0]['use_src_word']
    use_tgt_word = batch[0]['use_tgt_word']
    use_src_char = batch[0]['use_src_char']
    use_tgt_char = batch[0]['use_tgt_char']
    use_code_type = batch[0]['use_code_type']
    use_code_mask = batch[0]['use_code_mask']
    ids = [ex['id'] for ex in batch]
    language = [ex['language'] for ex in batch]

    # --------- Prepare Code tensors ---------
    final_src = {}
    for model_type in ['rep_0', 'rep_1', 'rep_2']:
        if f'{model_type}_code_word_rep' in batch[0]:
            final_src[model_type] = {}
    for model_type in ['rep_0', 'rep_1', 'rep_2']:
        src_vocabs = []
        source_maps = []
        if f'{model_type}_code_word_rep' not in batch[0]:
            continue
        code_words = [ex[f'{model_type}_code_word_rep'] for ex in batch]
        code_chars = [ex[f'{model_type}_code_char_rep'] for ex in batch]
        code_type = [ex[f'{model_type}_code_type_rep'] for ex in batch]
        code_mask = [ex[f'{model_type}_code_mask_rep'] for ex in batch]
        code_tokens = [ex[f'{model_type}_code_tokens'] for ex in batch]
        max_code_len = max([d.size(0) for d in code_words])
        if use_src_char:
            max_char_in_code_token = code_chars[0].size(1)

        # Batch Code Representations
        code_len_rep = torch.zeros(batch_size, dtype=torch.long)
        code_word_rep = torch.zeros(batch_size, max_code_len, dtype=torch.long) \
            if use_src_word else None
        code_type_rep = torch.zeros(batch_size, max_code_len, dtype=torch.long) \
            if use_code_type else None
        code_mask_rep = torch.zeros(batch_size, max_code_len, dtype=torch.long) \
            if use_code_mask else None
        code_char_rep = torch.zeros(batch_size, max_code_len, max_char_in_code_token, dtype=torch.long) \
            if use_src_char else None

        for i in range(batch_size):
            code_len_rep[i] = code_words[i].size(0)
            if use_src_word:
                code_word_rep[i, :code_words[i].size(0)].copy_(code_words[i])
            if use_code_type:
                code_type_rep[i, :code_type[i].size(0)].copy_(code_type[i])
            if use_code_mask:
                code_mask_rep[i, :code_mask[i].size(0)].copy_(code_mask[i])
            if use_src_char:
                code_char_rep[i, :code_chars[i].size(0), :].copy_(code_chars[i])

            context = batch[i][f'{model_type}_code_tokens']
            vocab = batch[i]['src_vocab']
            src_vocabs.append(vocab)
            src_map = torch.LongTensor([vocab[w] for w in context])
            source_maps.append(src_map)

        final_src[model_type]['code_word_rep'] = code_word_rep
        final_src[model_type]['code_char_rep'] = code_char_rep
        final_src[model_type]['code_type_rep'] = code_type_rep
        final_src[model_type]['code_mask_rep'] = code_mask_rep
        final_src[model_type]['code_len'] = code_len_rep
        final_src[model_type]['src_vocab'] = src_vocabs
        final_src[model_type]['src_map'] = source_maps
        final_src[model_type]['code_tokens'] = code_tokens
        src_vocabs_copy = src_vocabs

    # --------- Prepare Summary tensors ---------
    no_summary = batch[0]['summ_word_rep'] is None
    if no_summary:
        summ_len_rep = None
        summ_word_rep = None
        summ_char_rep = None
        tgt_tensor = None
        alignments = None
    else:
        summ_words = [ex['summ_word_rep'] for ex in batch]
        summ_chars = [ex['summ_char_rep'] for ex in batch]
        max_sum_len = max([q.size(0) for q in summ_words])
        if use_tgt_char:
            max_char_in_summ_token = summ_chars[0].size(1)

        summ_len_rep = torch.zeros(batch_size, dtype=torch.long)
        summ_word_rep = torch.zeros(batch_size, max_sum_len, dtype=torch.long) \
            if use_tgt_word else None
        summ_char_rep = torch.zeros(batch_size, max_sum_len, max_char_in_summ_token, dtype=torch.long) \
            if use_tgt_char else None

        max_tgt_length = max([ex['target'].size(0) for ex in batch])
        tgt_tensor = torch.zeros(batch_size, max_tgt_length, dtype=torch.long)
        alignments = []
        for i in range(batch_size):
            summ_len_rep[i] = summ_words[i].size(0)
            if use_tgt_word:
                summ_word_rep[i, :summ_words[i].size(0)].copy_(summ_words[i])
            if use_tgt_char:
                summ_char_rep[i, :summ_chars[i].size(0), :].copy_(summ_chars[i])
            #
            tgt_len = batch[i]['target'].size(0)
            tgt_tensor[i, :tgt_len].copy_(batch[i]['target'])
            target = batch[i]['summ_tokens']
            align_mask = torch.LongTensor([src_vocabs_copy[i][w] for w in target])
            alignments.append(align_mask)

    return {
        'ids': ids,
        'language': language,
        'batch_size': batch_size,
        'src_vocab': src_vocabs_copy,
        'final_src': final_src,
        'code_text': [ex['code'] for ex in batch],
        'summ_word_rep': summ_word_rep,
        'summ_char_rep': summ_char_rep,
        'summ_len': summ_len_rep,
        'tgt_seq': tgt_tensor,
        'summ_text': [ex['summ'] for ex in batch],
        'summ_tokens': [ex['summ_tokens'] for ex in batch],
        'alignment': alignments,
        'stype': [ex['stype'] for ex in batch]
    }
