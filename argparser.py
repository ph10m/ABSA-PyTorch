import argparse


class ArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add('--model_name', default='lcf_bert', type=str)
        self.add('--dataset', default='laptop', type=str, help='twitter, restaurant, laptop')
        self.add('--optimizer', default='adam', type=str)
        self.add('--initializer', default='xavier_uniform_', type=str)
        self.add('--learning_rate', default=5e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
        self.add('--dropout', default=0.1, type=float)
        self.add('--l2reg', default=0.01, type=float)
        self.add('--num_epoch', default=3, type=int, help='try larger number for non-BERT models')
        self.add('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
        self.add('--log_step', default=5, type=int)
        self.add('--embed_dim', default=300, type=int)
        self.add('--hidden_dim', default=300, type=int)
        self.add('--bert_dim', default=768, type=int)
        self.add('--pretrained_bert_name', default='bert-base-uncased', type=str)
        self.add('--max_seq_len', default=80, type=int)
        self.add('--polarities_dim', default=3, type=int)
        self.add('--hops', default=3, type=int)
        self.add('--device', default=None, type=str, help='e.g. cuda:0')
        self.add('--seed', default=None, type=int, help='set seed for reproducibility')
        self.add('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')
        # The following parameters are only valid for the lcf-bert model
        self.add('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
        self.add('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')

    def add(self, arg, _default, _type, _help=None):
        self.parser.add_argument(
            arg,
            default=_default,
            type=_type,
            help=_help)

    def get_options(self):
        return self.parser.parse_args()
