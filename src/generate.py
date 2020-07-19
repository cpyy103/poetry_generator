#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
from src.config import Config


def generate(model, start_words, word2index, index2word, prefix_words=None):
    '''
    给定开头几个词生成一首诗歌
    :param model: 当前所用的模型
    :param start_words: 给定开头
    :param word2index:
    :param index2word:
    :param prefix_words: 不是诗歌组成部分，用来生成诗歌意境
    :return: 生成的诗歌
    '''
    result = list(start_words)
    start_words_length = len(start_words)
    # 设置第一个词为 <START>
    input = torch.LongTensor([word2index['<START>']]).view(1, 1)
    if Config['use_gpu']:
        input = input.cuda()
    hidden = None

    # 生成意境
    if prefix_words:
        # 第一个input是<START>，然后是prefix_words中的汉字
        # 第一个hidden是None, 然后是生成的hidden
        for word in prefix_words:
            output, hidden = model(input, hidden)
            try:
                input = input.data.new([word2index[word]]).view(1, 1)
            except:
                return '语料库中没有 "%s" 字' % word

    # 生成诗词
    # 如果前面没有风格词，那么input是<START>，hidden是None
    # 否则，input是风格前缀生成的最后一个词，hidden也是生成出来的
    for i in range(Config['max_generate_length']):
        output, hidden = model(input, hidden)
        if i < start_words_length:
            word = result[i]
            try:
                input = input.data.new([word2index[word]]).view(1, 1)
            except:
                return '语料库中没有 "%s" 字' % word
        else:
            # 可能性最大的词
            top_index = output.data[0].topk(1)[1][0].item()
            word = index2word[top_index]
            result.append(word)
            input = input.data.new([top_index]).view(1, 1)
        # 结束诗词
        if word == '<EOP>':
            del result[-1]
            break

    return ''.join(result)


def generate_acrostic(model, start_words, word2index, index2word, prefix_words=None):
    '''
    生成藏头诗
    :param model: 当前模型
    :param start_words: 藏头的语句
    :param word2index:
    :param index2word:
    :param prefix_words: 不是诗歌组成部分，用来生成诗歌意境
    :return: 藏头诗
    '''
    result = []
    n_sentence = len(start_words)   # 生成几句藏头诗
    input = torch.LongTensor([word2index['<START>']]).view(1, 1)
    if Config['use_gpu']:
        input = input.cuda()
    hidden = None

    # 生成意境
    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            try:
                input = input.data.new([word2index[word]]).view(1, 1)
            except:
                return '语料库中没有 "%s" 字' % word

    pre_words = '<START>'   # 设置开始
    count = 0   # 用来记录已经生成的藏头诗的句数
    for i in range(Config['max_generate_length']):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        word = index2word[top_index]

        # 如果生成了句号感叹号，说明上个词是句末，将需要藏的头插入
        if pre_words in {'。', '！', '<START>'}:
            # 藏头诗生成完毕
            if count == n_sentence:
                break
            else:
                # 将藏头送入模型
                word = start_words[count]
                count += 1
                try:
                    input = input.data.new([word2index[word]]).view(1, 1)
                except:
                    return '语料库中没有 "%s" 字' % word
        else:
            # 送入上次生成的普通词
            input = input.data.new([word2index[word]]).view(1, 1)

        result.append(word)
        pre_words = word    # 将word标记为上一个词

    return ''.join(result)

