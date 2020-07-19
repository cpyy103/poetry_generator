#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import torch
import torch.nn as nn
from src.model import PoetryModel
from src.data import get_data
from src.config import Config
from src.generate import generate, generate_acrostic
import json


def train():
    data, word2index, index2word = get_data()
    data = torch.from_numpy(data)
    data_loader = torch.utils.data.DataLoader(data, batch_size=Config['batch_size'], shuffle=True)

    model = PoetryModel(len(word2index), Config['embedding_dim'], Config['hidden_dim'])
    if os.path.exists(Config['model_path']):
        model.load_state_dict(torch.load(Config['model_path']))
    model.to(Config['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=Config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    print('training...')
    for epoch in range(Config['epochs']):
        log = []    # 日志类数据
        for index, data in enumerate(data_loader):
            # 转置成shape ( sequence_length, batch_size )  sequence_length 为max_len
            # contiguous 为了改变底层的存储，contiguous重新按顺序存储，否则执行view会报错
            data = data.long().transpose(1, 0).contiguous().to(Config['device'])

            x, y = data[:-1, :], data[1:, :]
            y_, _ = model(x)

            optimizer.zero_grad()
            loss = criterion(y_, y.view(-1))
            loss.backward()
            optimizer.step()
            if (index+1) % Config['print_every'] == 0:
                print('epoch %d, iter %d, loss %.4f' % (epoch, index+1, loss.item()))
                temp = {}   # 用来保存过程数据
                temp['epoch'] = epoch
                temp['iter'] = index+1
                temp['loss'] = loss.item()

                # 分别以不同的字开头续写诗
                print('普通诗词')
                temp['普通诗词'] = []
                for w in list('山上一把火'):
                    poetry = generate(model, w, word2index, index2word)
                    print(poetry)
                    temp['普通诗词'].append(poetry)

                # 生成藏头诗
                print('藏头诗')
                poerty_acrostic = generate_acrostic(model, '我想回校', word2index, index2word)
                print(poerty_acrostic)
                temp['藏头诗'] = poerty_acrostic

                log.append(temp)

        with open(os.path.join(Config['log_json_path'], 'epoch_%d.json' % epoch), 'w',
                  encoding='utf-8', errors='ignore') as f:
            # 将log转字符串再转字典再保存
            json.dump(json.loads(json.dumps(log)), f, ensure_ascii=False)

    torch.save(model.state_dict(), Config['model_path'])


if __name__ == '__main__':
    train()
