# 精度验证脚本
import operator

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

import numpy as np

from src.config import Config
from src.dataset import DataGenerator

class Test:
    """测试类"""
    def __init__(self, n_entity, n_relation, test_triple, entities_emb, 
                 relations_emb, train_triple=None, is_filter=False):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.test_triple = test_triple
        self.is_filter = is_filter
        self.entities_emb = entities_emb
        self.relations_emb = relations_emb
        self.hits10 = 0
        self.mean_rank = 0
    
    def get_distance(self, vec_tri, norm=1):
        if norm == 1:
            return float(ops.abs(vec_tri[0] + vec_tri[1] - vec_tri[2]).sum())
        return float(ops.square(vec_tri[0] + vec_tri[1] - vec_tri[2]).sum())
    
    def rank(self):
        hits = 0
        rank_sum = 0
        step = 0
        
        for triple in self.test_triple:
            rank_head_dict = {}
            rank_tail_dict = {}
            
            # for entity in range(self.n_entity):
            print(triple)
            for entity in range(triple[0]+1):
                # 实体编号即为 [0, 实体数)
                corrupt_head_tri = (self.entities_emb[entity], self.relations_emb[triple[1]], 
                                    self.entities_emb[triple[2]])
                corrupt_tail_tri = (self.entities_emb[triple[0]], self.relations_emb[triple[1]],
                                    self.entities_emb[entity])
                # 不考虑过滤
                rank_head_dict[(entity, triple[1], triple[2])] = self.get_distance(corrupt_head_tri)
                rank_tail_dict[(triple[0], triple[1], entity)] = self.get_distance(corrupt_tail_tri)
            # 排序
            rank_head_sorted = sorted(rank_head_dict.items(), key=operator.itemgetter(1))
            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1))
            # rank_sum and hits
            for i, sorted_triple in enumerate(rank_head_sorted):
                # head 相同
                if i >= 10: 
                    break # just see top10
                if triple[0] == sorted_triple[0][0]:
                    if i < 10:
                        hits += 1
                    rank_sum += i + 1
                    break
            for i, sorted_triple in enumerate(rank_tail_sorted):
                # tail 相同
                if i >= 10: 
                    break # just see top10
                if triple[2] == sorted_triple[0][2]:
                    if i < 10:
                        hits += 1
                    rank_sum += i + 1
                    break
            
            step += 1
            if step % 1 == 0:
                print(f"step: {step}, hits: {hits/2/step}, rank_sum: {rank_sum/2/step}")
        
        self.hits10 = hits / (2 * len(self.test_triple))
        self.mean_rank = rank_sum / (2 * len(self.test_triple))
        return self.hits10, self.mean_rank

param_path = "./checkpoints/model_transH_epoch10_1661550314.ckpt"
param_dict = ms.load_checkpoint(param_path)

# 对transH
print(param_dict)

config = Config(
            root_dir='E:/comptition/maoshenAI/mycode/submit/data/id_data/', 
            # root_dir='/dataset/data/id_data/', # 对云端训练
            dataset='FB15k-237/', 
            mode='valid',
            model="transR",
            batch_size=512, 
            n_entity=14541, 
            n_relation=237, 
            n_entity_dim=50, 
            n_relation_dim=50)

ds = DataGenerator(config.root_dir, config.dataset, config.mode, config.n_entity)
test = Test(config.n_entity, config.n_relation, ds.data, 
            param_dict["entities_emb"], param_dict["relations_emb"])

test.rank()

