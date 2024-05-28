import torch.nn as nn
from gym import spaces
from gym.spaces import Discrete
import torch.nn.functional as F
import collections
import numpy as np

from env.HERec import HERec
from env.HAN import HAN
from env.MCRec import MCRec
from metrics import *
import time
import torch
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from sklearn.metrics import f1_score

from utils import load_data, EarlyStopping

from KGDataLoader import *

STOP = 0

NEG_SIZE_TRAIN = 20
NEG_SIZE_EVAL = 20
NEG_SIZE_RANKING = 499

USER_TYPE = 5
ITEM_TYPE = 0


class SemanticAttentionl(nn.Module):#语义注意力机制
    def __init__(self, in_size, hidden_size=64):
        super(SemanticAttentionl, self).__init__()

        self.project = nn.Sequential(#两个线性层和一个Tanh激活函数
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):#元路径重要性注意力,z为一个三维，路径数、项目数、维度

        w = self.project(z).mean(0)  # mean（0）第一维度求平均
        beta = torch.softmax(w, dim=0)  # (M, 1)，查询向量
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        return (beta * z).sum(1)  # (N, D * K)




def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1


class hgnn_env(object):
    def __init__(self, logger1, logger2, model_name, args, dataset='yelp_data', weight_decay=1e-5):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cur_best = 0
        self.args = args
        self.cf_l2loss_lambda = args.cf_l2loss_lambda
        lr = args.lr
        self.lr = lr
        self.task = args.task
        task = self.task
        dataset = args.data_name
        self.dataset = dataset
        self.past_performance = []
        self.init = -1
        self.weight_decay = weight_decay

        self.semantic_attentionl = SemanticAttentionl(64, 64)
        global USER_TYPE
        global ITEM_TYPE
        if dataset == 'yelp_data':
            USER_TYPE = 4
            ITEM_TYPE = 0
        elif dataset == 'double_movie':
            USER_TYPE = 5
            ITEM_TYPE = 0


        if task == 'rec' :
            self.etypes_lists = eval(args.mpset)#可能为元路径集合



            self.data = DataLoaderHGNN(logger1, args, dataset)
            self.metapath_transform_dict = self.data.metapath_transform_dict

            data = self.data
            if task == 'rec':
                self.model = HAN(
                    in_size=data.entity_dim,
                    hidden_size=args.hidden_dim,
                    out_size=data.entity_dim,
                    num_heads=args.num_heads,
                    dropout=0,
                    threshold=0.75).to(
                    self.device)

            self.train_data = data.train_graph
            self.train_data.x = self.train_data.x.to(self.device)

            self.train_data.node_idx = self.train_data.node_idx.to(self.device)
            self._set_action_space(len(data.metapath_transform_dict) + 1)
            self.user_policy = None
            self.item_policy = None
            self.eval_neg_dict = collections.defaultdict(list)
            self.test_neg_dict = collections.defaultdict(list)

        self.data = data
        print("Data node type list: ", self.data.node_type_list)

        self.mpset_eval_dict = dict()
        self.nd_batch_size = args.nd_batch_size
        self.rl_batch_size = args.rl_batch_size
        if task != "herec":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
        else:
            self.optimizer = None

        self.obs = self.reset()
        self._set_observation_space(self.obs)
        # self.W_R = torch.randn(self.data.n_relations + 1, self.data.entity_dim,
        #                        self.data.relation_dim).to(self.device)
        # nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        # self.kg_l2loss_lambda = args.kg_l2loss_lambda

        self.baseline_experience = 1
        logger1.info('Data initialization done')
#--------------------------------初始化完成----------------------------
    def reset_past_performance(self):#重置过去性能记录
        if self.init == -1:
            if self.optimizer:
                self.model.train()
                self.optimizer.zero_grad()
            # self.etypes_lists = [[['2', '1']], [['1', '2']]]
            # self.train_GNN()
            self.init = self.eval_batch()
            if self.task == 'rec':
                self.model.reset()
        self.past_performance = [self.init]

    def evaluate(self, model, g, features, labels, mask, loss_func):#评估
        self.model.eval()
        ids = torch.tensor(range(self.train_data.x.shape[0]))
        with torch.no_grad():
            logits = self.model(g, features, self.etypes_lists[0], self.optimizer, ids, test=True)
        loss = loss_func(logits[mask], labels[mask])
        accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

        return loss, accuracy, micro_f1, macro_f1
#------------------------------------------------------------------------

    def get_user_embedding(self, u_ids, test=False):
        h=[]
        # for path_index  in  [0,2,4,8,6]:#yelp
        for path_index in [0,2,4,6]:  # movie改！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            u_embedding = self.model(self.train_data, self.train_data.x[self.data.node_type_list == USER_TYPE], self.etypes_lists[path_index],
                       self.optimizer, u_ids, test)
            h.append(u_embedding)
#----------------------------------------------------------------------------




        # h = self.model(self.train_data, self.train_data.x[self.data.node_type_list == USER_TYPE], self.etypes_lists[0],
        #                self.optimizer, u_ids, test)
        return h

    def get_item_embedding(self, i_ids, test=False):


        # h = self.model(self.train_data, self.train_data.x[self.data.node_type_list == ITEM_TYPE], self.etypes_lists[1],
        #                self.optimizer, i_ids, test)
        h = []
        # for path_index in [1,3,7,5,9]:#yelp
        for path_index in [1, 3,5,7]:  # movie改！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            i_embedding = self.model(self.train_data, self.train_data.x[self.data.node_type_list == ITEM_TYPE], self.etypes_lists[path_index],
                       self.optimizer, i_ids, test)
            h.append(i_embedding)
        return h

    def get_all_user_embedding(self, test=False):
        all_user_ids = torch.tensor(range(self.train_data.x[self.data.node_type_list == USER_TYPE].shape[0]))
        return self.get_user_embedding(all_user_ids, test)

    def get_all_item_embedding(self, test=False):
        all_item_ids = torch.tensor(range(self.train_data.x[self.data.node_type_list == ITEM_TYPE].shape[0]))
        return self.get_item_embedding(all_item_ids, test)

    def _set_action_space(self, _max):
        self.action_num = _max
        self.action_space = Discrete(_max)

    def _set_observation_space(self, observation):
        low = np.full(observation.shape, -np.float32('inf'))
        high = np.full(observation.shape, np.float32('inf'))
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        self.etypes_lists = eval(self.args.mpset)
        state = self.get_user_state()
        # state = self.train_data.x[0]
        # if self.task == 'classification':
        #     state = self.get_class_state()[0]
        if self.optimizer:
            self.optimizer.zero_grad()
        return state

    def cal_user_state(self):
        state = [0] * (self.data.n_relations + 1)
        for mp in self.etypes_lists[0]:
            for rel in mp:
                state[int(rel)] += 1
        v = np.array(state, dtype=np.float32)
        return np.expand_dims(v / (np.linalg.norm(v) + 1e-16), axis=0)

    def cal_item_state(self):
        state = [0] * (self.data.n_relations + 1)
        for mp in self.etypes_lists[1]:
            for rel in mp:
                state[int(rel)] += 1
        v = np.array(state, dtype=np.float32)
        return np.expand_dims(v / (np.linalg.norm(v) + 1e-16), axis=0)

    def reset_eval_dict(self):
        self.eval_neg_dict = collections.defaultdict(list)

    def seed(self, random_seed):
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

    def sample_state(self, embeds, nodes):
        state = []
        for i in range(self.rl_batch_size):
            index = random.sample(nodes, min(self.nd_batch_size, len(nodes)))
            state.append(F.normalize(torch.mean(embeds[index], 0), dim=0).cpu().detach().numpy())
        return np.array(state)

    def get_user_state(self):

        return self.cal_user_state()


    def user_reset(self):
        self.etypes_lists = eval(self.args.mpset)

        state = self.get_user_state()
        # state = self.cal_user_state()
        if self.optimizer:
            self.optimizer.zero_grad()
        return state

    def get_item_state(self):
        # nodes = range(self.train_data.x[self.data.node_type_list == ITEM_TYPE].shape[0])
        # item_embeds = self.get_all_item_embedding()
        # return self.sample_state(item_embeds, nodes)
        return self.cal_item_state()
        # return np.concatenate([self.cal_item_state(), self.sample_state(item_embeds, nodes)], axis=1)

    def item_reset(self):
        self.etypes_lists = eval(self.args.mpset)
        state = self.get_item_state()
        # state = self.cal_item_state()
        if self.optimizer:
            self.optimizer.zero_grad()
        return state

    def get_class_state(self):
        nodes = range(self.train_data.x.shape[0])
        b_ids = torch.tensor(range(self.train_data.x.shape[0]))
        class_embeds = self.model(self.train_data, self.train_data.x, self.etypes_lists[0],
                                  self.optimizer, b_ids, test=False)
        # class_embeds = self.embedding_func(class_embeds)
        return self.sample_state(class_embeds, nodes)

    def class_reset(self):
        self.etypes_lists = [[['pf', 'fp']]]
        state = self.get_class_state()
        self.optimizer.zero_grad()
        return state

    def rec_step(self, actions, logger1, logger2, test, type):
        if self.optimizer:#是否存在优化器
            self.model.train()
            self.optimizer.zero_grad()
        tmpmp = copy.deepcopy(self.etypes_lists)
        done_list = [False] * len(actions)
        next_state, reward, val_acc = [], [], []
        if test:
            print(actions)
        for i, act in enumerate(actions):
            if act == STOP:
                done_list[i] = True
                if test:
                    self.train_GNN(test=True)
                else:
                    self.train_GNN()
            else:
                augment_mp = self.metapath_transform_dict[act]
                for i in range(len(self.etypes_lists[type[0]])):
                    mp = self.etypes_lists[type[0]][i]
                    if len(mp) < (4 if self.args.task != 'mcrec' else 3):
                        if self.train_data.e_n_dict[mp[-1]][1] == self.train_data.e_n_dict[augment_mp[0]][0]:
                            self.etypes_lists[type[0]].append(mp[:])
                            mp.extend(augment_mp)
                        else:
                            if self.train_data.e_n_dict[mp[0]][0] == self.train_data.e_n_dict[augment_mp[-1]][1]:
                                self.etypes_lists[type[0]].append(mp[:])
                                mp[0:0] = augment_mp
                            else:
                                for inx in range(len(mp)):
                                    rel = mp[inx]
                                    if self.train_data.e_n_dict[rel][1] == self.train_data.e_n_dict[augment_mp[0]][0]:
                                        self.etypes_lists[type[0]].append(mp[:])
                                        mp[inx + 1:inx + 1] = augment_mp
                                        break

                if self.train_data.e_n_dict[augment_mp[0]][0] == type[1] and self.args.task != 'mcrec':
                    self.etypes_lists[type[0]].append(augment_mp)
                self.etypes_lists[type[0]] = list(
                    map(lambda x: list(x), set(map(lambda x: tuple(x), self.etypes_lists[type[0]]))))


                if str(self.etypes_lists) not in self.mpset_eval_dict:
                    self.train_GNN(act, test)
            if not test:
                if str(self.etypes_lists) not in self.mpset_eval_dict:
                    val_precision = self.eval_batch()
                    self.mpset_eval_dict[str(self.etypes_lists)] = val_precision
                else:
                    val_precision = self.mpset_eval_dict[str(self.etypes_lists)]
            else:
                # val_precision = self.eval_batch(NEG_SIZE_EVAL)
                val_precision = 0

            if len(self.past_performance) == 0:
                self.past_performance.append(val_precision)

            baseline = np.mean(np.array(self.past_performance[-self.baseline_experience:]))
            rew = 5 * (val_precision - baseline)
            if rew > 0.5:
                rew = 0.5
            elif rew < -0.5:
                rew = -0.5
            if actions[0] == STOP or len(self.past_performance) == 0:
                rew = 0
            reward.append(rew)
            val_acc.append(val_precision)
            self.past_performance.append(val_precision)
            logger1.info("Action: %d" % act)
            logger1.info("Val acc: %.5f  reward: %.5f" % (val_precision, rew))
            logger1.info("-----------------------------------------------------------------------")
        r = np.mean(np.array(reward))
        val_acc = np.mean(val_acc)

        if not self.optimizer:
            if len(self.model.eval_neg_dict) != 0 and len(self.eval_neg_dict) == 0:
                self.eval_neg_dict = self.model.eval_neg_dict
            if len(self.model.test_neg_dict) != 0 and len(self.test_neg_dict) == 0:
                self.test_neg_dict = self.model.test_neg_dict

        if actions[0] != STOP and self.meta_path_equal(tmpmp):
            r, reward = -0.5, [-0.5]
        logger2.info("Action: %d  Val acc: %.5f  reward: %.5f" % (actions[0], val_acc, r))
        logger2.info("Meta-path Set: %s" % str(self.etypes_lists))
        return done_list, r, reward, val_acc

    def meta_path_equal(self, tmp):
        mpset = copy.deepcopy(self.etypes_lists)
        tmp[0].sort()
        tmp[1].sort()
        mpset[0].sort()
        mpset[1].sort()
        if tmp[0] == mpset[0] and tmp[1] == mpset[1]:
            return True
        else:
            return False

    def user_step(self, logger1, logger2, actions, test=False,
                  type=(0, USER_TYPE)):  # type - (index_of_etpyes_list, index_of_node_type)
        done_list, r, reward, val_acc = self.rec_step(actions, logger1, logger2, test, type)
        next_state = self.get_user_state()
        if self.task == 'rec':
            self.model.reset()
        return next_state, reward, done_list, (val_acc, r)

    def item_step(self, logger1, logger2, actions, test=False, type=(1, ITEM_TYPE)):
        done_list, r, reward, val_acc = self.rec_step(actions, logger1, logger2, test, type)
        next_state = self.get_item_state()
        if self.task == 'rec':
            self.model.reset()
        return next_state, reward, done_list, (val_acc, r)

    def class_step(self, logger1, logger2, actions, test=False, type=(0, 'p')):
        done_list, r, reward, val_acc = self.rec_step(actions, logger1, logger2, test, type)
        next_state = self.get_class_state()

        return next_state, reward, done_list, (val_acc, r)

    # def train_classifier(self, test=False):
    #     stopper = EarlyStopping(patience=50)
    #     loss_fcn = torch.nn.CrossEntropyLoss()
    #     ids = torch.tensor(range(self.train_data.x.shape[0]))
    #
    #     if test:
    #         epoch = 200
    #     else:
    #         epoch = 100
    #
    #     for epoch in range(epoch):
    #         self.model.train()
    #         logits = self.model(self.train_data, self.train_data.x, self.etypes_lists[0], self.optimizer, ids,
    #                             test=False)
    #         loss = loss_fcn(logits[self.train_mask], self.labels[self.train_mask])
    #
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #
    #         train_acc, train_micro_f1, train_macro_f1 = score(logits[self.train_mask], self.labels[self.train_mask])
    #         val_loss, val_acc, val_micro_f1, val_macro_f1 = self.evaluate(self.model, self.train_data,
    #                                                                       self.train_data.x,
    #                                                                       self.labels, self.val_mask, loss_fcn)
    #         early_stop = stopper.step(val_loss.data.item(), val_acc, self.model)
    #
    #         if (epoch + 1) % 20 == 0:
    #             print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
    #                   'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
    #                 epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1,
    #                 val_macro_f1))
    #
    #         if early_stop:
    #             break
    #     stopper.load_checkpoint(self.model)


    def train_GNN(self, act=STOP, test=False):

        if self.task == 'rec':

            # for epoch in range(self.epochs):
            #
            #     self.step += 1
            self.train_recommender(test, act)


    def train_recommender(self, test, act=STOP):
        n_cf_batch = 2 * self.data.n_cf_train // self.data.cf_batch_size + 1
        # n_cf_batch = 1
        cf_total_loss = 0
        if test:
            n_cf_batch = 1
        for iter in range(1, n_cf_batch + 1):
            #     print("current iter: ", iter, " ", n_cf_batch)
            time1 = time.time()
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = self.data.generate_cf_batch(self.data.train_user_dict)
            time2 = time.time()

            self.optimizer.zero_grad()

            # print("generate batch: ", time2 - time1)
            cf_batch_loss = self.calc_cf_loss(cf_batch_user,
                                              cf_batch_pos_item,
                                              cf_batch_neg_item, test, act)

            time3 = time.time()
            # print("calculate loss: ", time3 - time2)

            cf_batch_loss.backward()

            time4 = time.time()
            # print("backward: ", time4 - time3)

            self.optimizer.step()

            time5 = time.time()
            # print("step: ", time5 - time4)

            cf_total_loss += float(cf_batch_loss)
        # cf_total_loss.backward()
        # self.optimizer.step()
        # print("total_cf_loss: ", float(cf_total_loss))



    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids, test=False, act=STOP):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)
        """
        tim1 = time.time()
        # pred = self.update_embedding().to(self.device)
        unode_ids = torch.tensor([user_id - self.data.n_id_start_dict[USER_TYPE] for user_id in user_ids])

        # import pdb
        # pdb.set_trace()
#-------------------------------------------------------------------------------
        # user_embed = self.get_user_embedding(unode_ids)
        u_embeds = self.get_all_user_embedding(test)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       #--------------------------------------
        u_graph_embeds = np.loadtxt('user_graph_movie_64.txt')#改！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        #u_graph_embeds = np.loadtxt('user_graph_movie_64.txt')
        u_graph_embeds = torch.from_numpy(u_graph_embeds)
        u_graph_embeds = u_graph_embeds.float()
        torch.set_printoptions(precision=4, sci_mode=False)
        u_graph_embeds=u_graph_embeds.to(device)

        u_embeds.append(u_graph_embeds)


#------------------------------------------------------
        tim2 = time.time()

        u_embeds = torch.stack(u_embeds, dim=1)  # (N, M, D * K)
        #移到cpu
        u_embeds = u_embeds.to(device)

        self.semantic_attentionl = self.semantic_attentionl.to(device)
        #获取嵌入
        u_embeds=self.semantic_attentionl(u_embeds)  # (N, D * K)

        u_embeds_cpu = u_embeds.cpu().detach()

        np.savetxt('u_movie_embeds.txt', u_embeds_cpu.numpy())#改！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        #np.savetxt('u_movie_embeds.txt', u_embeds_cpu.numpy())



        # print("get user embedding: ", tim2 - tim1)

        # item_pos_embed = self.get_item_embedding(item_pos_ids)
        i_embeds = self.get_all_item_embedding(test)
        #-------------------------------------------------------
        i_graph_embeds = np.loadtxt('movie_graph_movie_64.txt')#改！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        #i_graph_embeds = np.loadtxt('movie_graph_movie_64.txt')

        i_graph_embeds = torch.from_numpy(i_graph_embeds)
        i_graph_embeds = i_graph_embeds.float()
        i_graph_embeds = i_graph_embeds.to(device)

        i_embeds.append(i_graph_embeds)
        #--------------------------------------------------------
        tim3 = time.time()
        i_embeds = torch.stack(i_embeds, dim=1)  # (N, M, D * K)

        i_embeds = i_embeds.to(device)
        self.semantic_attentionl = self.semantic_attentionl.to(device)

        i_embeds = self.semantic_attentionl(i_embeds)  # (N, D * K)
        i_embeds_cpu = i_embeds.cpu().detach()
        np.savetxt('i_movie_embeds.txt', i_embeds_cpu.numpy())#改！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        #np.savetxt('i_movie_embeds.txt', i_embeds_cpu.numpy())
#---------------------------------------------------------------------------

        user_embed = u_embeds[unode_ids]
        item_pos_embed = i_embeds[item_pos_ids]  # (cf_batch_size, cf_concat_dim)
        item_neg_embed = i_embeds[item_neg_ids]  # (cf_batch_size, cf_concat_dim)

        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)  # (cf_batch_size)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)  # (cf_batch_size)

        # print("pos, neg: ", pos_score, neg_score)
        # print("user_embedding: ", user_embed)
        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)
        # print("cf_loss: ", float(cf_loss))

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss

    def eval_batch(self, neg_num=NEG_SIZE_TRAIN):
        if self.task == 'rec':
            return self.eval_recommender(neg_num)


    def eval_classifier(self):
        loss_fcn = torch.nn.CrossEntropyLoss()
        val_loss, val_precision, val_micro_f1, val_macro_f1 = self.evaluate(self.model, self.train_data,
                                                                            self.train_data.x,
                                                                            self.labels, self.val_mask, loss_fcn)
        print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
            val_loss.item(), val_micro_f1, val_macro_f1))

        return val_precision

    # def eval_recommender(self, neg_num):
    #     self.model.eval()
    #     time1 = time.time()
    #     user_ids = list(self.data.train_user_dict.keys())
    #     self.seed(0)
    #     user_ids_batch = random.sample(user_ids, min(len(user_ids) - 2, self.args.train_batch_size))
    #
    #     if not self.eval_neg_dict and neg_num == NEG_SIZE_EVAL:
    #         print("neg_sum: ", NEG_SIZE_EVAL)
    #
    #     for u in user_ids_batch:
    #         if u not in self.eval_neg_dict:
    #             for _ in self.data.train_user_dict[u]:
    #                 nl = self.data.sample_neg_items_for_u(self.data.train_user_dict, u, neg_num)
    #                 self.eval_neg_dict[u].extend(nl)
    #     with torch.no_grad():
    #         u_embeds = self.get_all_user_embedding()
    #         i_embeds = self.get_all_item_embedding()
    #
    #         time2 = time.time()
    #
    #         pos_logits = torch.tensor([]).to(self.device)
    #         neg_logits = torch.tensor([]).to(self.device)
    #
    #         cf_scores = torch.matmul(
    #             u_embeds[[user_id - self.data.n_id_start_dict[USER_TYPE] for user_id in user_ids_batch]],
    #             i_embeds.transpose(0, 1))
    #         for idx, u in enumerate(user_ids_batch):
    #             pos_logits = torch.cat([pos_logits, cf_scores[idx][self.data.train_user_dict[u]]])
    #             neg_logits = torch.cat([neg_logits, torch.unsqueeze(cf_scores[idx][self.eval_neg_dict[u]], 1)])
    #         # print(pos_logits.shape)
    #         # print(neg_logits.shape)
    #         time3 = time.time()
    #         NDCG10 = self.metrics(pos_logits, neg_logits)
    #         print(f"Evaluate: NDCG10 : {NDCG10.item():.5f}")
    #         time4 = time.time()
    #         # print("ALL time: ", time4 - time1)
    #     return NDCG10.cpu().item()


    def test_batch(self, logger2):
        if self.task == 'rec':
            return self.test_recommender(logger2)


    def test_recommender(self, logger2):
        self.model.eval()
        user_ids = list(self.data.test_user_dict.keys())
        user_ids_batch = user_ids[:]
        NDCG10 = 0
        with torch.no_grad():
            for u in user_ids_batch:
                if u not in self.test_neg_dict:
                    nl = self.data.sample_neg_items_for_u_test(self.data.train_user_dict, self.data.test_user_dict,
                                                               u, NEG_SIZE_RANKING)
                    for _ in self.data.test_user_dict[u]:
                        self.test_neg_dict[u].extend(nl)
            # self.train_data.x.weight = nn.Parameter(self.train_data.x.weight.to(self.device))
            # all_embed = self.update_embedding().to(self.device)

            # u_embeds = self.get_all_user_embedding(True)
            # i_embeds = self.get_all_item_embedding(True)

            u_embeds = np.loadtxt('u_movie_embeds.txt')#改！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            i_embeds = np.loadtxt('i_movie_embeds.txt')
            # u_embeds = np.loadtxt('u_movie_embeds.txt')  # 改！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            # i_embeds = np.loadtxt('i_movie_embeds.txt')
            u_embeds = torch.from_numpy(u_embeds)
            i_embeds = torch.from_numpy(i_embeds)
            pos_logits = torch.tensor([]).to(self.device)
            neg_logits = torch.tensor([]).to(self.device)

            cf_scores = torch.matmul(
                u_embeds[[user_id - self.data.n_id_start_dict[USER_TYPE] for user_id in user_ids_batch]],
                i_embeds.transpose(0, 1))
            device = 'cuda'  # 或者 'cpu'

            pos_logits = pos_logits.to(device)
            neg_logits=neg_logits.to(device)
            cf_scores = cf_scores.to(device)

            for idx, u in enumerate(user_ids_batch):
                pos_logits = torch.cat([pos_logits, cf_scores[idx][self.data.test_user_dict[u]]])
                neg_logits = torch.cat([neg_logits, torch.unsqueeze(cf_scores[idx][self.test_neg_dict[u]], 1)])
            HR1, HR3, HR10, HR20, NDCG10, NDCG20 = self.metrics(pos_logits, neg_logits,
                                                                training=False)
            logger2.info(
                "HR1 : %.4f, HR3 : %.4f, HR10 : %.4f, HR20 : %.4f, NDCG10 : %.4f, NDCG20 : %.4f" % (
                    HR1, HR3, HR10, HR20, NDCG10.item(), NDCG20.item()))
            print(
                f"Test: HR1 : {HR1:.4f}, HR3 : {HR3:.4f}, HR10 : {HR10:.4f}, NDCG10 : {NDCG10.item():.4f}, NDCG20 : {NDCG20.item():.4f}")
        return NDCG10.cpu().item()



    def metrics(self, batch_pos, batch_nega, training=True):
        hit_num1 = 0.0
        hit_num3 = 0.0
        hit_num10 = 0.0
        hit_num20 = 0.0
        # hit_num50 = 0.0
        # mrr_accu10 = torch.tensor(0)
        # mrr_accu20 = torch.tensor(0)
        # mrr_accu50 = torch.tensor(0)
        ndcg_accu10 = torch.tensor(0).to(self.device)
        ndcg_accu20 = torch.tensor(0).to(self.device)
        # ndcg_accu50 = torch.tensor(0)

        if training:
            batch_neg_of_user = torch.split(batch_nega, NEG_SIZE_TRAIN, dim=0)
        else:
            batch_neg_of_user = torch.split(batch_nega, NEG_SIZE_RANKING, dim=0)
        if training:
            for i in range(batch_pos.shape[0]):
                pre_rank_tensor = torch.cat((batch_pos[i].view(1, 1), batch_neg_of_user[i]), dim=0).to(self.device)
                _, indices = torch.topk(pre_rank_tensor, k=pre_rank_tensor.shape[0], dim=0)
                rank = torch.squeeze((indices == 0).nonzero().to(self.device))
                rank = rank[0]
                if rank < 10:
                    ndcg_accu10 = ndcg_accu10 + torch.log(torch.tensor([2.0]).to(self.device)) / torch.log(
                        (rank + 2).type(torch.float32))
            return ndcg_accu10 / batch_pos.shape[0]
        else:
            for i in range(batch_pos.shape[0]):
                pre_rank_tensor = torch.cat((batch_pos[i].view(1, 1), batch_neg_of_user[i]), dim=0).to(self.device)
                _, indices = torch.topk(pre_rank_tensor, k=pre_rank_tensor.shape[0], dim=0)
                rank = torch.squeeze((indices == 0).nonzero().to(self.device))
                rank = rank[0]
                # if rank < 50:
                #     ndcg_accu50 = ndcg_accu50 + torch.log(torch.tensor([2.0]).to(self.device)) / torch.log(
                #         (rank + 2).type(torch.float32))
                #     mrr_accu50 = mrr_accu50 + 1 / (rank + 1).type(torch.float32)
                #     hit_num50 = hit_num50 + 1
                if rank < 20:
                    ndcg_accu20 = ndcg_accu20 + torch.log(torch.tensor([2.0]).to(self.device)) / torch.log(
                        (rank + 2).type(torch.float32))
                    hit_num20 = hit_num20 + 1
                    # mrr_accu20 = mrr_accu20 + 1 / (rank + 1).type(torch.float32)
                if rank < 10:
                    ndcg_accu10 = ndcg_accu10 + torch.log(torch.tensor([2.0]).to(self.device)) / torch.log(
                        (rank + 2).type(torch.float32))
                    hit_num10 = hit_num10 + 1
                # if rank < 10:
                # mrr_accu10 = mrr_accu10 + 1 / (rank + 1).type(torch.float32)
                if rank < 3:
                    hit_num3 = hit_num3 + 1
                if rank < 1:
                    hit_num1 = hit_num1 + 1
            # return hit_num1 / batch_pos.shape[0], hit_num3 / batch_pos.shape[0], hit_num10 / batch_pos.shape[
            #     0], hit_num50 / \
            #        batch_pos.shape[0], mrr_accu10 / batch_pos.shape[0], mrr_accu20 / batch_pos.shape[0], mrr_accu50 / \
            #        batch_pos.shape[0], \
            #        ndcg_accu10 / batch_pos.shape[0], ndcg_accu20 / batch_pos.shape[0], ndcg_accu50 / batch_pos.shape[0]
            return hit_num1 / batch_pos.shape[0], hit_num3 / batch_pos.shape[0], hit_num10 / batch_pos.shape[
                0], hit_num20 / batch_pos.shape[
                       0], ndcg_accu10 / batch_pos.shape[0], ndcg_accu20 / batch_pos.shape[0]

