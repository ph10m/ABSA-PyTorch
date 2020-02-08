from models import LSTM, TD_LSTM, IAN, RAM, TNet_LF, LCF_BERT, LCF_ALBERT, AEN_BERT
import torch

models = {
    'lstm': LSTM,
    'td_lstm': TD_LSTM,
    'ian': IAN,
    'ram': RAM,
    'tnet_lf': TNet_LF,
    'aen_bert': AEN_BERT,
    'lcf_bert': LCF_BERT,
    'lcf_albert': LCF_ALBERT,
    # default hyper-parameters for LCF-BERT model is as follws:
    # lr: 2e-5
    # l2: 1e-5
    # batch size: 16
    # num epochs: 5
}

optimizers = {
    'adadelta': torch.optim.Adadelta,  # default lr=1.0
    'adagrad': torch.optim.Adagrad,  # default lr=0.01
    'adam': torch.optim.Adam,  # default lr=0.001
    'adamax': torch.optim.Adamax,  # default lr=0.002
    'asgd': torch.optim.ASGD,  # default lr=0.01
    'rmsprop': torch.optim.RMSprop,  # default lr=0.01
    'sgd': torch.optim.SGD
}

initializers = {
  'xavier_uniform_': torch.nn.init.xavier_uniform_,
  'xavier_normal_': torch.nn.init.xavier_normal,
  'orthogonal_': torch.nn.init.orthogonal_
}
