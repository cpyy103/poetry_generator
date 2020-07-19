#!/usr/bin/python
# -*- coding:utf-8 -*-
# 用于保存一些设置，便于后续更改
import torch


Config = {
    'bidirectional': True,     # LSTM是否双向
    'num_layers': 3,    # LSTM 层数
    'embedding_dim': 256,   # embedding 词向量维度
    'hidden_dim': 512,   # 隐藏层维度
    'data_path': '../input/chinese-poetry-master/json/',  # 数据路径，直接把这个github项目放到里面
    'category': 'poet.tang',    # 选择唐诗作为主要数据
    'author': None,     # 只学习某位诗人的诗
    'constrain': None,  # 诗词固定长度，一般为5或7
    'numpy_data_path': '../input/tang.npz',  # 制作好的二进制数据文件保存
    'max_len': 100,     # 控制训练数据的每句诗歌最大长度
    'use_gpu': torch.cuda.is_available(),  # 可以直接定义为True 或 False
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    'batch_size': 128,
    'learning_rate': 1e-3,
    'model_path': '../output/model_3_bi_256_512.pth',   # 3层，双向，embedding256，hidden512
    'epochs': 100,
    'print_every': 100,  # 一个训练epoch中每训练多少batch打印一次信息用于提示

    'max_generate_length': 132,  # 最大生成诗歌的长度

    'log_json_path': '../output/log'  # 日志

}
