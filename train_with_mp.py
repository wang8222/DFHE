import collections
import torch
from KGDataLoader import parse_args
from dqn_agent_pytorch import DQNAgent
import numpy as np
import os
import random
import time
from copy import deepcopy
import logging
import torch.nn as nn
import env.HAN as HAN
import matplotlib.pyplot as plt
from env.HERec import HERec
from env.MCRec import MCRec
from env.hgnn import hgnn_env
from train_fm import use_pretrain
import os
from KGDataLoader import *
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s', "%Y-%m-%d %H:%M:%S")
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

    return logging.getLogger(logger_name)


# def use_pretrain(env, dataset='yelp_data'):
#     if dataset == 'yelp_data':
#         print('./data/yelp_data/embedding/user.embedding_' + str(env.data.entity_dim))
#         fr1 = open('./data/yelp_data/embedding/user.embedding_' + str(env.data.entity_dim), 'r')
#         fr2 = open('./data/yelp_data/embedding/item.embedding_' + str(env.data.entity_dim), 'r')
#     # elif dataset == 'douban_movie':
#     #     print('./data/douban_movie/embedding/user.embedding_' + str(env.data.entity_dim))
#     #     fr1 = open('./data/douban_movie/embedding/user.embedding_' + str(env.data.entity_dim), 'r')
#     #     fr2 = open('./data/douban_movie/embedding/item.embedding_' + str(env.data.entity_dim), 'r')
#
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     emb = env.train_data.x
#     emb.requires_grad = False
#
#     for line in fr1.readlines():
#         embeddings = line.strip().split()
#         id, embedding = int(embeddings[0]), embeddings[1:]
#         embedding = list(map(float, embedding))
#         emb[id] = torch.tensor(embedding)
#
#     for line in fr2.readlines():
#         embeddings = line.strip().split()
#         id, embedding = int(embeddings[0]), embeddings[1:]
#         embedding = list(map(float, embedding))
#         emb[id] = torch.tensor(embedding)
#
#     # emb.requires_grad = True
#     env.train_data.x = emb.to(device)


def main():
    tim1 = time.time()
    torch.backends.cudnn.deterministic = True
    # max_timesteps = 2
    # dataset = 'ACMRaw'

    args = parse_args()
    # HAN.DEGREE_THERSHOLD = 80000
    dataset = args.data_name

    infor = 'pretrain_' + str(args.data_name) + '_' + str(args.task) + '_' + str(args.log)
    model_name = 'model_' + infor + '.pth'

    max_episodes = 90 if dataset == 'yelp_data' else 150
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger1 = get_logger('log', 'log/logger_' + infor + '.log')
    logger2 = get_logger('log2', 'log/logger2_' + infor + '.log')

# #--------------------------------------------------------
#     # 加载数据
#     data = DataLoaderHGNN()
#
#     print("Node type list: ", data.node_type_list)




#     print("Total nodes: ", data.num_nodes)
#
#     # 初始化环境
#     env = hgnn_env(data=data)
#     #--------------------------------------------------
    if args.data_name == 'yelp_data':
        # u_set = [['2', '1'], ['2', '3', '7', '1'], ['2', '4', '8', '1'], ['5', '9'],
        #          ['5', '9', '5', '9'], ['5', '9', '6', '5', '9'], ['5', '9', '5', '9', '6'], ['6', '6', '6', '6'],
        #          ['2', '1', '5', '9'], ['2', '1', '6'], ['6', '6'], ['5', '9', '6'], ['6', '5', '9'],
        #          ['2', '1', '6', '6'],
        #          ['2', '1', '2', '1', '6'], ['6', '6', '6'], ['6', '6', '6', '6'], ['5', '9', '6', '6'],
        #          ['5', '9', '6', '6', '6'], ['2', '1', '6', '6', '6'], ['2', '3', '7', '1', '6'],
        #          ['2', '4', '8', '1', '6'],
        #          ['6', '5', '9', '6'], ['6', '5', '9', '6', '6'], ['2', '1', '6', '2', '1'], ['2', '1', '6', '5', '9']]
        u_set = [['2', '1'],['5', '9'],  ['2', '3', '7', '1'], ['2', '4', '8', '1'], ['6', '6']]
                #UBU   UBCaBU   UBCiBU   UCoU  UUU

        i_set = [['1', '2'],  ['3', '7'], ['4', '8'], ['1', '6', '2'], ['1', '5', '9', '2']]
                #BUB   BCaB   BCiB    BUUB    BUCoUB
        # i_set = [['1', '2'], ['1', '2', '4', '8'], ['1', '2', '3', '7'], ['1', '6', '2'], ['1', '6', '6', '2'],
        #          ['1', '5', '9', '2'], ['4', '8'], ['3', '7'], ['4', '8', '3', '7'], ['3', '7', '4', '8'],
        #          ['1', '6', '2', '4', '8'], ['1', '6', '2', '3', '7'], ['3', '7', '3', '7'], ['1', '6', '6', '6', '2'],
        #          ['1', '2', '1', '2'], ['1', '2', '1', '6', '2'], ['1', '5', '9', '6', '2'], ['4', '8', '4', '8'],
        #          ['1', '6', '5', '9', '2'], ['4', '8', '1', '2'], ['3', '7', '1', '2']]
    elif args.data_name == 'douban_movie':
        u_set = [['2', '3', '8', '1'],['2', '4', '9', '1'],['2', '5', '10', '1'],['2', '1'],['6', '11', '7'],]
        # u_set = [['7', '6', '11', '6', '11'], ['7', '2', '1', '6', '11'], ['7', '6', '11', '2', '1'],
        #          ['6', '11', '6', '11'], ['6', '11', '7', '6', '11'], ['6', '11', '6', '11', '7'],
        #          ['7', '7', '7', '7', '7'], ['7', '7', '6', '11'], ['7', '7', '2', '1'], ['6', '11', '2', '1'],
        #          ['2', '1', '6', '11'], ['2', '1', '7'], ['7', '7'],  ['7', '6', '11'],
        #          ['2', '1', '7', '7'], ['7', '2', '1', '7', '7'], ['2', '1', '7', '6', '11'],
        #          ['2', '1', '2', '1', '7'], ['7', '7', '7'], ['7', '7', '7', '7'], ['6', '11', '7', '7'],
        #          ['2', '5', '10', '1', '7'], ['7', '7', '7', '6', '11'], ['6', '11', '7', '2', '1'],
        #          ['6', '11', '7', '7', '7'], ['2', '1', '7', '7', '7'], ['2', '3', '8', '1', '7'],
        #          ['2', '4', '9', '1', '7'], ['7', '2', '1', '7', '7'],
        #          ['7', '6', '11', '7'], ['7', '6', '11', '7', '7'], ['2', '1', '7', '2', '1'],
        #          ['2', '1', '7', '6', '11']]

        i_set = [['1', '2'], ['4', '9'], ['3', '8'], ['5', '10'], ['1', '6', '11', '2'],]
        # i_set = [ ['1', '2', '4', '9'], ['1', '2', '3', '8'], ['1', '7', '2'], ['1', '7', '7', '2'],
        #          ['1', '2', '5', '10'], ['4', '9', '1', '2'], ['3', '8', '1', '2'], ['5', '10', '1', '2'],
        #            ['4', '9', '3', '8'], ['3', '8', '4', '9'],
        #          ['5', '10', '3', '8'], ['5', '10', '4', '9'], ['4', '9', '5', '10'], ['3', '8', '5', '10'],
        #          ['1', '7', '2', '5', '10'], ['5', '10', '5', '10'], ['5', '10', '1', '7', '2'],
        #          ['4', '9', '1', '7', '2'],
        #          ['3', '8', '1', '7', '2'], ['1', '7', '2', '1', '2'], ['1', '7', '2', '5', '10'],
        #          ['1', '7', '6', '11', '2'],
        #          ['1', '7', '2', '4', '9'], ['1', '7', '2', '3', '8'], ['3', '8', '3', '8'], ['1', '7', '7', '7', '2'],
        #          ['1', '2', '1', '2'], ['1', '2', '1', '7', '2'], ['1', '6', '11', '7', '2'], ['4', '9', '4', '9']]

    init_method = args.init

    timelimit = args.limit

    print("u_set: ", len(u_set), " i_set: ", len(i_set))



    if init_method == 'specify':
        mpset = eval(args.mpset)
        train_and_test(1, max_episodes, tim1, logger1, logger2, model_name, args, mpset)



def train_and_eval(env, inx, max_episodes, tim1, logger1, logger2, model_name, args, mpset):
    print("Current test: ", inx, ' Metapath Set: ', mpset)
    tim2 = time.time()
    env.etypes_lists = mpset

    if args.task == 'rec':
        for gnn in env.model.layers:
            gnn.threshold = 0.9

    env.train_GNN()
    acc = env.eval_batch()
    if args.task == 'rec':
        env.model.reset()
    print("Acc: ", acc
          , ". This test time: ", (time.time() - tim2) / 60, "min"
          , ". Current time: ", (time.time() - tim1) / 60, "min")
    return acc
   

def train_and_test(inx, max_episodes, tim1, logger1, logger2, model_name, args, mpset):
    tim2 = time.time()
    env = hgnn_env(logger1, logger2, model_name, args)
    print("Current test: ", inx, ' Metapath Set: ', str(env.etypes_lists))
    env.seed(0)
    use_pretrain(env, args.data_name)

    if args.task == 'rec':
        for gnn in env.model.layers:
            gnn.threshold = 0.98

    env.etypes_lists = mpset
    best = 0
    best_i = 0
    # val_list = [0, 0, 0]



    print('env.etypes_lists',env.etypes_lists)
    for i in range(1, max_episodes + 1):
        print('Current epoch: ', i)
        env.train_GNN()
        if i % 1 == 0:
            acc = env.test_batch(logger2)
            # val_list.append(acc)
            if acc > best:
                best = acc
                best_i = i
                print('Best: ', best, ' Best_i: ', best_i)
            logger2.info('Best Accuracy: %.5f\tBest_i : %d' % (best, best_i))
        # if i - best_i > 15:
        #     break
    print("Best: ", best, '. Best_i: ', best_i
          , ". This test time: ", (time.time() - tim2) / 60, "min"
          , ". Current time: ", (time.time() - tim1) / 60, "min")
    return best


def train_and_test_for_draw(inx, max_episodes, tim1, logger1, logger2, model_name, args, mpset):
    tim2 = time.time()
    env = hgnn_env(logger1, logger2, model_name, args)
    env.seed(0)
    use_pretrain(env, args.data_name)
    env.etypes_lists = mpset
    best = env.eval_batch()
    best_i = 0
    val_list = [best]
    print(env.etypes_lists)
    for i in range(1, max_episodes + 1):
        print('Current epoch: ', i)
        env.train_GNN()
        if i % 1 == 0:
            acc = env.eval_batch()
            val_list.append(acc)
            if acc > best:
                best = acc
                best_i = i
                print('Best: ', best, ' Best_i: ', best_i)
            logger2.info('Best Accuracy: %.5f\tBest_i : %d' % (best, best_i))
    print("Current test: ", inx, ' Metapath Set: ', str(env.etypes_lists)
          , '.\n Best: ', best, '. Best_i: ', best_i
          , ". This test time: ", (time.time() - tim2) / 60, "min"
          , ". Current time: ", (time.time() - tim1) / 60, "min")
    return val_list


if __name__ == '__main__':
     main()
