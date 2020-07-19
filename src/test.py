#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
import os
import numpy as np

from src.generate import generate, generate_acrostic
from src.model import PoetryModel
from src.config import Config
import os
print(os.getcwd())

def test():
    print('Init...')
    # 导入数据
    if not os.path.exists(Config['numpy_data_path']):
        raise 'Not Found Data %s' % Config['numpy_data']
    data = np.load(Config['numpy_data_path'], allow_pickle=True)
    word2index = data['word2index'].item()
    index2word = data['index2word'].item()
    # 导入模型
    if not os.path.exists(Config['model_path']):
        raise 'Not Found Model %s' % Config['model_path']
    model = PoetryModel(len(word2index), Config['embedding_dim'], Config['hidden_dim'])
    model.load_state_dict(torch.load(Config['model_path']))
    if Config['use_gpu']:
        model.to(Config['device'])
    print('Init Done ! ')

    while True:
        print('欢迎使用诗词生成器\n输1入1 根据首句继续生成诗词\n输入2 生成藏头诗\n输入3 退出')
        try:
            ans = int(input())
        except:
            print('请输入相应数字')
            continue
        if ans == 1:
            print('请输入首句：')
            starts_words = str(input())
            print('请输入意境（不是诗词组成部分，可直接回车）：')
            prefix_words = str(input())
            poetry = generate(model, starts_words, word2index, index2word, prefix_words)
            print(poetry)
        elif ans == 2:
            print('请输入需要藏头的句子：')
            starts_words = str(input())
            print('请输入意境（不是诗词组成部分，可直接回车）：')
            prefix_words = str(input())
            poetry = generate_acrostic(model, starts_words, word2index, index2word, prefix_words)
            print(poetry)
        elif ans == 3:
            break

    print('成功退出')


if __name__ == '__main__':
    test()

