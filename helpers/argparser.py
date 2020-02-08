import argparse


class ArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.add('model_name', 'lcf_albert', str)

        self.add('dataset', 'twitter', str,
                 'twitter, restaurant, laptop')

        self.add('optimizer', 'adam', str)

        self.add('initializer', 'xavier_uniform_', str)

        self.add('learning_rate', 5e-5, float,
                 'try 5e-5, 2e-5 for BERT, 1e-3 for others')

        self.add('dropout', 0.1, float)

        self.add('l2reg', 0.01, float)

        self.add('num_epoch', 3, int, 'try larger number for non-BERT models')

        self.add('batch_size', 16, int, 'try 16, 32, 64 for BERT models')

        self.add('log_step', 5, int)

        self.add('embed_dim', 300, int)

        self.add('hidden_dim', 300, int)

        self.add('bert_dim', 768, int)

        self.add('pretrained_name', 'albert-base-v2', str)

        self.add('max_seq_len', 80, int)

        self.add('polarities_dim', 3, int)

        self.add('hops', 3, int)

        self.add('device', None, str, 'e.g. cuda:0')

        self.add('seed', 1337, int,
                 'set seed for reproducibility')

        self.add('valset_ratio', 0, float,
                 'set ratio between 0 and 1 for validation support')

        # The following parameters are only valid for the lcf-bert model
        self.add('local_context_focus', 'cdm', str,
                 'local context focus mode, cdw or cdm')

        self.add('SRD', 3, int,
                 'semantic-relative-distance, see LCF-BERT paper')

    def add(self, arg, _default, _type, _help=None):
        self.parser.add_argument(
            '--' + arg,
            default=_default,
            type=_type,
            help=_help)

    def get_options(self):
        return self.parser.parse_args()
