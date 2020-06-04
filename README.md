# Elsa-Val
![LICENSE](https://img.shields.io/packagist/l/doctrine/orm.svg)
[![Gitter](https://badges.gitter.im/ABSA-PyTorch/community.svg)](https://gitter.im/ABSA-PyTorch/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-6-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

## Requirements
* pytorch
* numpy
* sklearn
* python
* transformers

To install requirements, run `pip install -r requirements.txt`.

* For non-BERT-based models,
[GloVe pre-trained word vectors](https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors) are required, please refer to [data_utils.py](./data_utils.py) for more detail.

## Usage
### Training

```sh
python train.py --model_name lcf_bert --dataset restaurant
```
* All implemented models are listed in [zoo.py](./zoo.py).
* See [train.py](./train.py) for more training arguments.
* Refer to [train_k_fold_cross_val.py](./train_k_fold_cross_val.py) for k-fold cross validation support.

## Reviews / Surveys

Qiu, Xipeng, et al. "Pre-trained Models for Natural Language Processing: A Survey." arXiv preprint arXiv:2003.08271 (2020). [[pdf]](https://arxiv.org/pdf/2003.08271)

Zhang, Lei, Shuai Wang, and Bing Liu. "Deep Learning for Sentiment Analysis: A Survey." arXiv preprint arXiv:1801.07883 (2018). [[pdf]](https://arxiv.org/pdf/1801.07883)

Young, Tom, et al. "Recent trends in deep learning based natural language processing." arXiv preprint arXiv:1708.02709 (2017). [[pdf]](https://arxiv.org/pdf/1708.02709)


### LCF-BERT ([lcf_bert.py](./models/lcf_bert.py)) ([official](https://github.com/yangheng95/LCF-ABSA))

Zeng Biqing, Yang Heng, et al. "LCF: A Local Context Focus Mechanism for Aspect-Based Sentiment Classification." Applied Sciences. 2019, 9, 3389. [[pdf]](https://www.mdpi.com/2076-3417/9/16/3389/pdf)

### Cabasc ([cabasc.py](./models/cabasc.py))

Liu, Qiao, et al. "Content Attention Model for Aspect Based Sentiment Analysis." Proceedings of the 2018 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2018.

### TD-LSTM ([td_lstm.py](./models/td_lstm.py)) ([official](https://drive.google.com/open?id=17RF8MZs456ov9MDiUYZp0SCGL6LvBQl6))

Tang, Duyu, et al. "Effective LSTMs for Target-Dependent Sentiment Classification." Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers. 2016. [[pdf]](https://arxiv.org/pdf/1512.01100)

### LSTM ([lstm.py](./models/lstm.py))

Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." Neural computation 9.8 (1997): 1735-1780. [[pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.676.4320&rep=rep1&type=pdf)]

## Licence

MIT
