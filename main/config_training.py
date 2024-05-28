def training_setting():
    conf = {
        # Optimization details
        'optimizer': 'adam',
        'learning_rate': 0.0001,
        'lr_decay': 0.99,
        'grad_clipping': 5.0,
        'weight_decay': 0,
        'momentum': 0,
        'warmup_epochs': 0,
        'early_stop': 20,
        # Logging
        'print_copy_info': False,
        'print_one_target': False,
        # General
        'valid_metric': 'bleu',
        'display_iter': 25,
        'sort_by_len': True,
        # Environment
        'max_examples': -1,
        'data_workers': 5,
        'random_seed': 2024,
        'num_epochs': 200,
        'use_code_type': False,
        'parallel': False,
        'drop_last': True
    }
    return conf
