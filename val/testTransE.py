# 精度验证脚本

import mindspore as ms
import mindspore.ops as ops

import numpy as np
import tqdm


class TestTransE:
    """测试类"""
    def __init__(self, n_entity, n_relation, test_triple, entities_emb, 
                 relations_emb, is_filter=False, norm=1):
        self.n_entity = n_entity             # 实体数量
        self.n_relation = n_relation         # 关系数量
        self.test_triple = test_triple       # 测试集的三元组。ms.Tensor, shape=(N, 3)，N为测试样本数。
        # 过滤则判断测试时，corrupt样本是否在训练集中出现，如果出现，则也认为hit成功
        # **此代码不支持过滤**
        self.is_filter = is_filter           # 是否过滤
        self.entities_emb = entities_emb     # 实体编码
        self.relations_emb = relations_emb   # 关系编码
        self.norm = norm                     # 计算距离时所用范数
        self.hits10 = 0                      # hits10结果
        self.mean_rank = 0                   # mean_rank结果

        self.dist_op = ops.Abs() if self.norm == 1 else ops.Square() # 计算距离算子


    def rank(self):
        hits = 0
        rank_sum = 0
        
        for triple in tqdm.tqdm(self.test_triple):
            # triple: 包含3个整数的tuple=(head,relation,tail)，形如 (1,2,3)
            entities = self.entities_emb
            head = self.entities_emb[triple[0]]
            tail = self.entities_emb[triple[2]]
            relation = self.relations_emb[triple[1]] # shape=(n_dim)
            

            # 为了后续便于一一对应计算，将向量重复
            # 说明：此处使用封装的向量运算，相比使用for循环完成计算hit10等，前者速度大大提高
            relation_repeat = ms.numpy.repeat(ops.ExpandDims()(relation,0), self.n_entity, axis=0) # shape=(n_entity, n_dim)
            head_repeat = ms.numpy.repeat(ops.ExpandDims()(head,0), self.n_entity, axis=0)
            tail_repeat = ms.numpy.repeat(ops.ExpandDims()(tail,0), self.n_entity, axis=0) # shape=(n_entity, n_dim)

            # corrupt head
            # 将triple的head分别替换成各个实体，计算替换后的距离，
            # 这里使用所有实体的张量直接计算，而不是通过for循环
            corrupt_head_dists = self.dist_op(entities + relation_repeat - tail_repeat).sum(axis=1) # 获得各个节点的距离
            dist, index = ops.sort(corrupt_head_dists) # 排序
            hits += int(triple[0] in index[:10])       # 看真实的head是否在排序后的前10中，即 hits10
            rank_sum += np.where(index.asnumpy() == triple[0])[0] # 计算真实的head的实际位次
            
            # corrupt tail
            # 将triple的tail分别替换成各个实体，计算替换后的距离，
            # 这里使用所有实体的张量直接计算，而不是通过for循环
            corrupt_tail_dists = self.dist_op(head_repeat + relation_repeat - entities).sum(axis=1)
            dist, index = ops.sort(corrupt_tail_dists) 
            hits += int(triple[2] in index[:10])
            rank_sum += np.where(index.asnumpy() == triple[2])[0]
            
        # 计算hits10和mean_rank
        self.hits10 = hits / (2 * len(self.test_triple)) # 之前对所有entity累加head和tail，计算平均值
        self.mean_rank = rank_sum / (2 * len(self.test_triple))

        return self.hits10, self.mean_rank

