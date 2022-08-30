# 精度验证脚本
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

import numpy as np
import tqdm

from src.config import Config
from src.dataset import DataGenerator


class Test:
    """测试类"""
    def __init__(self, n_entity, n_relation, test_triple, entities_emb, 
                 relations_emb, proj_weight=None, train_triple=None, is_filter=False, norm=1):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.test_triple = test_triple
        self.is_filter = is_filter
        self.entities_emb = entities_emb
        self.relations_emb = relations_emb
        self.proj_weight = proj_weight
        self.norm = norm
        self.hits10 = 0
        self.mean_rank = 0
    
    def get_distance(self, vec_tri, norm=1):
        if norm == 1:
            return float(ops.abs(vec_tri[0] + vec_tri[1] - vec_tri[2]).sum())
        return float(ops.square(vec_tri[0] + vec_tri[1] - vec_tri[2]).sum())
    
    def project(self, entity_emb, proj_vec):
        return entity_emb - (entity_emb * proj_vec).sum() * proj_vec

    def rank(self):
        hits = 0
        rank_sum = 0
        step = 0
        
        for triple in tqdm.tqdm(self.test_triple):
                
            proj = self.proj_weight[triple[1]]
            # 将entities映射
            entities_proj = self.entities_emb - ops.dot(self.entities_emb, proj.reshape(-1,1)) * proj
            relation = self.relations_emb[triple[1]]
            head_proj = entities_proj[triple[0]] # shape=(n_dim), 投影后的head
            tail_proj = entities_proj[triple[2]] # shape=(n_dim), 投影后的tail

            relations = ms.numpy.repeat(ops.ExpandDims()(relation,0), self.n_entity, axis=0) # shape=(n_entity, n_dim)
            head_projs = ms.numpy.repeat(ops.ExpandDims()(head_proj,0), self.n_entity, axis=0)
            tail_projs = ms.numpy.repeat(ops.ExpandDims()(tail_proj,0), self.n_entity, axis=0) # shape=(n_entity, n_dim)

            # corrupt head
            dist_op = ops.Abs()
            if self.norm == 2:
                dist_op = ops.Square()
            corrupt_head_dists = dist_op(entities_proj + relations - tail_projs).sum(axis=1) # 获得各个节点的距离
            dist, index = ops.sort(corrupt_head_dists)
            hits += int(triple[0] in index[:10])
            rank_sum += np.where(index.asnumpy() == triple[0])[0]
            
            # corrupt tail
            corrupt_tail_dists = dist_op(head_projs + relations - entities_proj).sum(axis=1)
            dist, index = ops.sort(corrupt_tail_dists)
            hits += int(triple[2] in index[:10])
            rank_sum += np.where(index.asnumpy() == triple[2])[0]
            
            #print(int(triple[2] in index[:10])+int(triple[0] in index[:10]),
            #          np.where(index.asnumpy() == triple[0])[0], np.where(index.asnumpy() == triple[2])[0])
        self.hits10 = hits / (2 * len(self.test_triple))
        self.mean_rank = rank_sum / (2 * len(self.test_triple))
        return self.hits10, self.mean_rank


param_path = "./checkpoints/model_transH_epoch16_1661816229.ckpt"
param_dict = ms.load_checkpoint(param_path)

# 对transH
print(param_dict)

config = Config(
            root_dir='E:/comptition/maoshenAI/mycode/submit/data/id_data/', 
            # root_dir='/dataset/data/id_data/', # 对云端训练
            dataset='FB15k-237/', 
            mode='test',
            model="transH",
            batch_size=512, 
            n_entity=14541, 
            n_relation=237, 
            n_entity_dim=50, 
            n_relation_dim=50)

ds = DataGenerator(config.root_dir, config.dataset, config.mode, config.n_entity)
test = Test(config.n_entity, config.n_relation, ds.data, 
            param_dict["entities_emb"], param_dict["relations_emb"], param_dict["w"], 
            model=config.model)

test.rank()

