# src: https://github.com/facebookresearch/DrQA/blob/master/scripts/reader/train.py

import sys

sys.path.append(".")
sys.path.append("..")

import os
import json
import torch
import logging
import subprocess
import argparse
import numpy as np

# import c2nl.config as config
import c2nl.inputters.utils as util
from c2nl.inputters import constants

from collections import OrderedDict, Counter
from tqdm import tqdm
from c2nl.inputters.timer import AverageMeter, Timer
import c2nl.inputters.vector as vector
import c2nl.inputters.dataset as data

from main.model import Code2NaturalLanguage
from c2nl.eval.bleu import corpus_bleu
from c2nl.eval.rouge import Rouge
from c2nl.eval.meteor import Meteor

logger = logging.getLogger()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'),
                         ['', 'K', 'M', 'B', 'T'][magnitude])


def init_from_scratch(args, train_exs, dev_exs):
    """New model, new data, new dictionary."""
    # Build a dictionary from the data questions + words (train/dev splits)
    logger.info('-' * 100)
    logger.info('Build word dictionary')
    fields_pre = []
    for field_ex in train_exs[0].keys():
        if field_ex != 'summary':
            fields_pre.append(field_ex)
    src_dict = util.build_word_and_char_dict(args,
                                             examples=train_exs + dev_exs,
                                             fields=fields_pre,
                                             dict_size=args.src_vocab_size,
                                             no_special_token=True)
    tgt_dict = util.build_word_and_char_dict(args,
                                             examples=train_exs + dev_exs,
                                             fields=['summary'],
                                             dict_size=args.tgt_vocab_size,
                                             no_special_token=False)
    logger.info('Num words in source = %d and target = %d' % (len(src_dict), len(tgt_dict)))

    # Initialize model
    model = Code2NaturalLanguage(args, src_dict, tgt_dict)

    return model


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------


def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    ml_loss = AverageMeter()
    perplexity = AverageMeter()
    epoch_time = Timer()

    current_epoch = global_stats['epoch']
    pbar = tqdm(data_loader)

    pbar.set_description("%s" % 'Epoch = %d [perplexity = x.xx, ml_loss = x.xx]' %
                         current_epoch)

    # Run one epoch
    for idx, ex in enumerate(pbar):
        bsz = ex['batch_size']
        if args.optimizer in ['sgd', 'adam'] and current_epoch <= args.warmup_epochs:
            cur_lrate = global_stats['warmup_factor'] * (model.updates + 1)
            for param_group in model.optimizer.param_groups:
                param_group['lr'] = cur_lrate

        net_loss = model.update_model(ex)
        ml_loss.update(net_loss['ml_loss'], bsz)
        perplexity.update(net_loss['perplexity'], bsz)
        log_info = 'Epoch = %d [perplexity = %.2f, ml_loss = %.2f]' % \
                   (current_epoch, perplexity.avg, ml_loss.avg)

        pbar.set_description("%s" % log_info)

    logger.info('train: Epoch %d | perplexity = %.2f | ml_loss = %.2f | '
                'Time for epoch = %.2f (s)' %
                (current_epoch, perplexity.avg, ml_loss.avg, epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint', current_epoch + 1)


# ------------------------------------------------------------------------------
# Validation loops.
# ------------------------------------------------------------------------------


def validate_official(args, data_loader, model, global_stats, mode='dev'):
    """Run one full official validation. Uses exact spans and same
    exact match/F1 score computation as in the SQuAD script.
    Extra arguments:
        offsets: The character start/end indices for the tokens in each context.
        texts: Map of qid --> raw text of examples context (matches offsets).
        answers: Map of qid --> list of accepted answers.
    """
    eval_time = Timer()
    # Run through examples
    examples = 0
    sources, hypotheses, references, copy_dict = dict(), dict(), dict(), dict()
    with torch.no_grad():
        pbar = tqdm(data_loader)
        for idx, ex in enumerate(pbar):
            batch_size = ex['batch_size']
            ex_ids = list(range(idx * batch_size, (idx * batch_size) + batch_size))
            predictions, targets, copy_info = model.model_predict(ex, replace_unk=True)

            src_sequences = [code for code in ex['code_text']]
            examples += batch_size
            for key, src, pred, tgt in zip(ex_ids, src_sequences, predictions, targets):
                hypotheses[key] = [pred]
                references[key] = tgt if isinstance(tgt, list) else [tgt]
                sources[key] = src

            if copy_info is not None:
                copy_info = copy_info.cpu().numpy().astype(int).tolist()
                for key, cp in zip(ex_ids, copy_info):
                    copy_dict[key] = cp

            pbar.set_description("%s" % 'Epoch = %d [validating ... ]' % global_stats['epoch'])

    print(hypotheses[0])
    print(references[0])
    copy_dict = None if len(copy_dict) == 0 else copy_dict
    bleu, rouge_l, meteor, precision, recall, f1 = eval_accuracies(args,
                                                                   hypotheses,
                                                                   references,
                                                                   copy_dict,
                                                                   sources=sources,
                                                                   filename=args.pred_file,
                                                                   print_copy_info=args.print_copy_info,
                                                                   mode=mode)
    result = dict()
    result['bleu'] = bleu
    result['rouge_l'] = rouge_l
    result['meteor'] = meteor
    result['precision'] = precision
    result['recall'] = recall
    result['f1'] = f1

    if mode == 'test':
        logger.info('test valid official: '
                    'bleu = %.2f | rouge_l = %.2f | meteor = %.2f | ' %
                    (bleu, rouge_l, meteor) +
                    'Precision = %.2f | Recall = %.2f | F1 = %.2f | '
                    'examples = %d | ' %
                    (precision, recall, f1, examples) +
                    'test time = %.2f (s)' % eval_time.time())

    else:
        logger.info('dev valid official: Epoch = %d | ' %
                    (global_stats['epoch']) +
                    'bleu = %.2f | rouge_l = %.2f | meteor = %.2f | '
                    'Precision = %.2f | Recall = %.2f | F1 = %.2f | examples = %d | ' %
                    (bleu, rouge_l, meteor, precision, recall, f1, examples) +
                    'valid time = %.2f (s)' % eval_time.time())

    return result


def normalize_answer(s):
    """Lower text and remove extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(lower(s))


def eval_score(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    precision, recall, f1 = 0, 0, 0
    if len(ground_truth) == 0:
        if len(prediction) == 0:
            precision, recall, f1 = 1, 1, 1
    else:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same != 0:
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1


def compute_eval_score(prediction, ground_truths):
    assert isinstance(prediction, str)
    precision, recall, f1 = 0, 0, 0
    for gt in ground_truths:
        _prec, _rec, _f1 = eval_score(prediction, gt)
        if _f1 > f1:
            precision, recall, f1 = _prec, _rec, _f1
    return precision, recall, f1


def eval_accuracies(args, hypotheses, references, copy_info, sources=None,
                    filename=None, print_copy_info=False, mode='dev'):
    """An unofficial evalutation helper.
     Arguments:
        hypotheses: A mapping from instance id to predicted sequences.
        references: A mapping from instance id to ground truth sequences.
        copy_info: Map of id --> copy information.
        sources: Map of id --> input text sequence.
        filename:
        print_copy_info:
    """
    assert (sorted(references.keys()) == sorted(hypotheses.keys()))

    # Compute BLEU scores
    # bleu_scorer = Bleu(n=4)
    # _, _, bleu = bleu_scorer.compute_score(references, hypotheses, verbose=0)
    # bleu = compute_bleu(references, hypotheses, max_order=4)['bleu']
    # _, bleu, ind_bleu = nltk_corpus_bleu(hypotheses, references)
    _, bleu, ind_bleu = corpus_bleu(hypotheses, references)

    # Compute ROUGE scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(references, hypotheses)

    # Compute METEOR scores
    if mode == 'test':
        meteor_calculator = Meteor()
        meteor, _ = meteor_calculator.compute_score(references, hypotheses)
    else:
        # 即使不是test也计算meteor
        meteor = 0
        meteor_calculator = Meteor()
        meteor, _ = meteor_calculator.compute_score(references, hypotheses)
        meteor_calculator.close()
    f1 = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()

    fw = open(filename, 'w') if filename else None
    for key in references.keys():
        _prec, _rec, _f1 = compute_eval_score(hypotheses[key][0],
                                              references[key])
        precision.update(_prec)
        recall.update(_rec)
        f1.update(_f1)
        if fw:
            if copy_info is not None and print_copy_info:
                prediction = hypotheses[key][0].split()
                pred_i = [word + ' [' + str(copy_info[key][j]) + ']'
                          for j, word in enumerate(prediction)]
                pred_i = [' '.join(pred_i)]
            else:
                pred_i = hypotheses[key]

            logobj = OrderedDict()
            logobj['id'] = key
            if sources is not None:
                logobj['code'] = sources[key]
            logobj['predictions'] = pred_i
            logobj['references'] = references[key][0] if args.print_one_target \
                else references[key]
            logobj['bleu'] = ind_bleu[key]
            logobj['rouge_l'] = ind_rouge[key]
            fw.write(json.dumps(logobj) + '\n')

    if fw: fw.close()
    return bleu * 100, rouge_l * 100, meteor * 100, precision.avg * 100, \
           recall.avg * 100, f1.avg * 100


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def main(args):
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load and process data files')

    train_exs = []
    if not args.only_test:
        args.dataset_weights = dict()
        train_files = dict()
        if args.use_src: train_files['src'] = args.train_src_files
        if args.use_cfg: train_files['cfg'] = args.train_cfg_files
        if args.use_pdg: train_files['pdg'] = args.train_pdg_files
        if args.use_mix: train_files['mix'] = args.train_mix_files
        if args.use_ast: train_files['ast'] = args.train_ast_files
        train_files['src_tag'] = args.train_src_tag_files
        train_files['tgt'] = args.train_tgt_files
        exs = util.load_data(args,
                             train_files,
                             max_examples=args.max_examples,
                             dataset_name=args.dataset_name)
        lang_name = constants.DATA_LANG_MAP[args.dataset_name]
        args.dataset_weights[constants.LANG_ID_MAP[lang_name]] = len(exs)
        train_exs.extend(exs)

        logger.info('Num train examples = %d' % len(train_exs))
        args.num_train_examples = len(train_exs)
        for lang_id in args.dataset_weights.keys():
            weight = (1.0 * args.dataset_weights[lang_id]) / len(train_exs)
            args.dataset_weights[lang_id] = round(weight, 2)
        logger.info('Dataset weights = %s' % str(args.dataset_weights))

    dev_exs = []
    dev_files = dict()
    if args.use_src: dev_files['src'] = args.dev_src_files
    if args.use_cfg: dev_files['cfg'] = args.dev_cfg_files
    if args.use_pdg: dev_files['pdg'] = args.dev_pdg_files
    if args.use_mix: dev_files['mix'] = args.dev_mix_files
    if args.use_ast: dev_files['ast'] = args.dev_ast_files
    dev_files['src_tag'] = args.dev_src_tag_files
    dev_files['tgt'] = args.dev_tgt_files
    exs = util.load_data(args,
                         dev_files,
                         max_examples=args.max_examples,
                         dataset_name=args.dataset_name,
                         test_split=True)
    dev_exs.extend(exs)
    logger.info('Num dev examples = %d' % len(dev_exs))

    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 1
    if args.only_test:
        if args.pretrained:
            model = Code2NaturalLanguage.load(args.pretrained)
        else:
            if not os.path.isfile(args.model_file):
                raise IOError('No such file: %s' % args.model_file)
            model = Code2NaturalLanguage.load(args.model_file)
    else:
        if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
            # Just resume training, no modifications.
            logger.info('Found a checkpoint...')
            checkpoint_file = args.model_file + '.checkpoint'
            model, start_epoch = Code2NaturalLanguage.load_checkpoint(checkpoint_file, args.use_cuda)
        else:
            # Training starts fresh. But the model state is either pretrained or newly (randomly) initialized.
            if args.pretrained:
                logger.info('Using pretrained model...')
                model = Code2NaturalLanguage.load(args.pretrained, args)
            else:
                logger.info('Training model from scratch...')
                model = init_from_scratch(args, train_exs, dev_exs) # vocab index

            # Set up optimizer
            model.init_optimizer()
            # log the parameter details
            logger.info('Trainable #parameters [encoder-decoder] {} [total] {}'.format(
                human_format(model.network.count_encoder_parameters() +
                             model.network.count_decoder_parameters()),
                human_format(model.network.count_parameters())))
            table = model.network.layer_wise_parameters()
            logger.info('Breakdown of the trainable paramters\n%s' % table)

    # Use the GPU?
    if args.use_cuda:
        model.cuda()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')

    if not args.only_test:
        train_dataset = data.CommentDataset(train_exs, model)
        if args.sort_by_len:
            train_sampler = data.SortedBatchSampler(train_dataset.lengths(),
                                                    args.batch_size,
                                                    shuffle=True)
        else:
            train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.use_cuda,
            drop_last=args.drop_last
        )

    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))

    # --------------------------------------------------------------------------
    # DO TEST

    if args.only_test:
        dev_dataset = data.CommentDataset(dev_exs, model)
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)

        dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=args.test_batch_size,
            sampler=dev_sampler,
            num_workers=args.data_workers,
            collate_fn=vector.batchify,
            pin_memory=args.use_cuda,
            drop_last=args.drop_last
        )
        stats = {'timer': Timer(), 'epoch': 0, 'best_valid': 0, 'no_improvement': 0}
        validate_official(args, dev_loader, model, stats, mode='test')

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    else:
        logger.info('-' * 100)
        logger.info('Starting training...')
        stats = {'timer': Timer(), 'epoch': start_epoch, 'best_valid': 0, 'no_improvement': 0}

        if args.optimizer in ['sgd', 'adam'] and args.warmup_epochs >= start_epoch:
            logger.info("Use warmup lrate for the %d epoch, from 0 up to %s." %
                        (args.warmup_epochs, args.learning_rate))
            num_batches = len(train_loader.dataset) // args.batch_size
            warmup_factor = (args.learning_rate + 0.) / (num_batches * args.warmup_epochs)
            stats['warmup_factor'] = warmup_factor

        for epoch in range(start_epoch, args.num_epochs + 1):
        # for epoch in range(start_epoch, 2*args.num_epochs + 1):
            stats['epoch'] = epoch
            if args.optimizer in ['sgd', 'adam'] and epoch > args.warmup_epochs:
                model.optimizer.param_groups[0]['lr'] = \
                    model.optimizer.param_groups[0]['lr'] * args.lr_decay

            train(args, train_loader, model, stats)
            if epoch % args.valid_every == 0 or epoch == 1:
            # if 1:
                dev_dataset = data.CommentDataset(dev_exs, model)
                dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)

                dev_loader = torch.utils.data.DataLoader(
                    dev_dataset,
                    batch_size=args.test_batch_size,
                    sampler=dev_sampler,
                    num_workers=args.data_workers,
                    collate_fn=vector.batchify,
                    pin_memory=args.use_cuda,
                    drop_last=args.drop_last
                )
                result = validate_official(args, dev_loader, model, stats)

                # Save best valid
                if result[args.valid_metric] > stats['best_valid']:
                    logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
                                (args.valid_metric, result[args.valid_metric],
                                 stats['epoch'], model.updates))
                    model.save(args.model_file)
                    stats['best_valid'] = result[args.valid_metric]
                    stats['no_improvement'] = 0
                else:
                    stats['no_improvement'] += 1
                    if stats['no_improvement'] >= args.early_stop:
                        break


def set_log(arg_log):
    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if arg_log.log_file:
        if arg_log.checkpoint:
            logfile = logging.FileHandler(arg_log.log_file, 'a')
        else:
            logfile = logging.FileHandler(arg_log.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))


def set_defaults(args):
    # Set cuda
    if args.use_cuda:
        args.cuda_available = torch.cuda.is_available()
        if args.cuda_available:
            torch.cuda.manual_seed(args.random_seed)
            assert os.environ['CUDA_VISIBLE_DEVICES'] == f'{args.GPU_id}', \
                'GPU_id in `args` should be the same as "os.environ" set in `main` '
        else:
            raise Exception('No GPU available')
    # Set model name
    model_name_not_set = 0
    if not args.model_name:
        model_name_not_set = 1
        args.model_name = 'default_name'
    # Set log + model file names
    suffix = '_test' if args.only_test else ''
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')
    args.log_file = os.path.join(args.model_dir, args.model_name + suffix + '.txt')
    args.pred_file = os.path.join(args.model_dir, args.model_name + suffix + '.json')
    if args.pretrained:
        args.pretrained = os.path.join(args.model_dir, args.pretrained + '.mdl')
    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    # Set logging
    set_log(args)
    # Set default dataset_name and model_name
    if len(args.dataset_name) == 0:
        logger.info('Dataset name is not set, use "Java" default.')
        args.dataset_name = 'java'
    if model_name_not_set:
        logger.info(f'Model name is not set, use {args.model_name} default.')
    return args


def path_joining(d_dir, mode, ex):
    file_path = os.path.join(d_dir, mode, ex)
    if os.path.exists(file_path):
        return file_path
    else:
        return None


def set_full_args(arg_a, conf_d, conf_tf, conf_tr):
    assert arg_a.use_src + arg_a.use_cfg + arg_a.use_pdg + arg_a.use_mix + arg_a.use_ast == 2, \
        '2 representations are allowed'
    data_dir = os.path.join(arg_a.root_dir, arg_a.dataset_name)
    arg_a.train_src_files = path_joining(data_dir, 'train', conf_d['src'])
    arg_a.dev_src_files = path_joining(data_dir, 'dev', conf_d['src'])
    arg_a.test_src_files = path_joining(data_dir, 'test', conf_d['src'])
    arg_a.train_cfg_files = path_joining(data_dir, 'train', conf_d['cfg'])
    arg_a.dev_cfg_files = path_joining(data_dir, 'dev', conf_d['cfg'])
    arg_a.test_cfg_files = path_joining(data_dir, 'test', conf_d['cfg'])
    arg_a.train_pdg_files = path_joining(data_dir, 'train', conf_d['pdg'])
    arg_a.dev_pdg_files = path_joining(data_dir, 'dev', conf_d['pdg'])
    arg_a.test_pdg_files = path_joining(data_dir, 'test', conf_d['pdg'])
    arg_a.train_mix_files = path_joining(data_dir, 'train', conf_d['mix'])
    arg_a.dev_mix_files = path_joining(data_dir, 'dev', conf_d['mix'])
    arg_a.test_mix_files = path_joining(data_dir, 'test', conf_d['mix'])
    arg_a.train_ast_files = path_joining(data_dir, 'train', conf_d['ast'])
    arg_a.dev_ast_files = path_joining(data_dir, 'dev', conf_d['ast'])
    arg_a.test_ast_files = path_joining(data_dir, 'test', conf_d['ast'])
    arg_a.train_tgt_files = path_joining(data_dir, 'train', conf_d['tgt'])
    arg_a.dev_tgt_files = path_joining(data_dir, 'dev', conf_d['tgt'])
    arg_a.test_tgt_files = path_joining(data_dir, 'test', conf_d['tgt'])
    if arg_a.use_code_type:
        arg_a.train_src_tag_files = path_joining(data_dir, 'train', conf_d['src_tag'])
        arg_a.dev_src_tag_files = path_joining(data_dir, 'dev', conf_d['src_tag'])
        arg_a.test_src_tag_files = path_joining(data_dir, 'test', conf_d['src_tag'])
    else:
        arg_a.train_src_tag_files = arg_a.dev_src_tag_files = arg_a.test_src_tag_files = None

    for key, value in conf_d.items():
        setattr(arg_a, key, value)
    for key, value in conf_tf.items():
        setattr(arg_a, key, value)
    for key, value in conf_tr.items():
        setattr(arg_a, key, value)

    return arg_a


def add_train_args():
    parser = argparse.ArgumentParser('Code to Natural Language Generation')
    parser.register('type', 'bool', str2bool)
    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--batch_size', type=int, default=8,
                         help='Batch size for training')
    runtime.add_argument('--test_batch_size', type=int, default=16,  # 32
                         help='Batch size during validation/testing')
    runtime.add_argument('--random_seed', type=int, default=2024,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    # Files
    files = parser.add_argument_group('Filesystem')
    # files.add_argument('--dataset_name', nargs='+', type=str, default='', required=True,
    files.add_argument('--dataset_name', nargs='+', type=str, default='java',
                       help='Name of the experimental dataset')
    files.add_argument('--model_name', type=str, default='java_clear_TT',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--model_dir', type=str, default='../../save_models/',
                       help='Directory for saved models/checkpoints/logs')
    # files.add_argument('--model_name', type=str, default='', required=True,
    files.add_argument('--root_dir', type=str, default='../../data/',
                       help='Directory of training/validation data')
    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=True,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default=None,
                           help='Path to a pretrained model to warm-start with')
    # General
    general = parser.add_argument_group('General')
    general.add_argument('--only_test', type='bool', default=True,
                         help='Only do testing')
    general.add_argument('--valid_every', type=int, default=5,
                         help='Valid every N epochs')
    general.add_argument('--use_code_type', type='bool', default=False,
                         help='Use code type as additional feature for feature representations')
    # use cuda
    cuda_set = parser.add_argument_group('Cuda settings')
    cuda_set.add_argument('--use_cuda', type='bool', default=False,
                          help='Choose whether to use cuda')
    cuda_set.add_argument('--GPU_id', type=int, default=1,
                    help='If use_parallel is False, choose the GPU by id for training')
    # use cuda
    multi_data = parser.add_argument_group('Data selected')
    multi_data.add_argument('--use_src', type='bool', default=True)
    multi_data.add_argument('--use_cfg', type='bool', default=True)
    multi_data.add_argument('--use_pdg', type='bool', default=False)
    multi_data.add_argument('--use_mix', type='bool', default=False)
    multi_data.add_argument('--use_ast', type='bool', default=False)

    return parser.parse_args()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    arg = add_train_args()
    conf_arg = set_defaults(arg)
    import config_transformer
    import config_data
    import config_training
    conf_trans = getattr(config_transformer, 'transformer_setting')()
    conf_data = getattr(config_data, 'data_setting')(conf_arg.dataset_name)
    conf_train = getattr(config_training, 'training_setting')()
    conf_args = set_full_args(conf_arg, conf_data, conf_trans, conf_train)

    # Run!
    main(conf_args)
