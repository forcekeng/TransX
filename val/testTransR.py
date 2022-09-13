# transH精度验证脚本

import mindspore as ms
import mindspore.ops as ops

import numpy as np
import tqdm


class TestTransR:
    """测试类"""
    def __init__(self, n_entity, n_relation, test_triple, entities_emb, 
                 relations_emb, mat, is_filter=False, train_triple=None, norm=1):
        self.n_entity = n_entity             # 实体数量
        self.n_relation = n_relation         # 关系数量
        self.test_triple = test_triple       # 测试集的三元组。ms.Tensor, shape=(N, 3)，N为测试样本数。
        self.is_filter = is_filter           # 是否过滤，**此代码不支持过滤**。过滤则判断测试集中的样本是否在训练集中出现过
        # 训练集，用于filter时判断corrupt的样本是否出现在训练集中
        self.train_triple = self._get_train_triple(train_triple)
        self.entities_emb = entities_emb     # 实体编码
        self.relations_emb = relations_emb   # 关系编码
        self.mat = mat               # 映射矩阵，transR模型专用
        self.norm = norm                     # 计算距离时所用范数
        self.hits10 = 0                      # hits10结果
        self.mean_rank = 0                   # mean_rank结果

        self.dist_op = ops.Abs() if self.norm == 1 else ops.Square() # 计算距离


    def _get_train_triple(self, train_triple):
        """将训练集的三元组转换成集合，便于filter时快速过滤
        train_triple: ms.Tensor : shape=(n_sample, 3), dtype=int32. 训练集三元组
        return: set(list[tuple()]): 将训练集三元组转换成集合，便于快速查询
        """
        if (not self.is_filter) or (train_triple is None):
            return
        train_triple = train_triple.asnumpy().tolist()   #转成list
        train_triple = set([tuple(tri) for tri in train_triple]) # 内部元素转成tuple，所有tuple组成集合
        return train_triple


    def rank(self):
        hits = 0
        rank_sum = 0
        
        for triple in tqdm.tqdm(self.test_triple[:200]):
            # triple: 包含3个整数的tuple，形如 (1,2,3)
            # 将所有entities减去各自在通过变换矩阵变换
            # self.entities_emb.shape = (n_entity, n_entity_dim) # n_entity_dim == n_relation_dim
            # self.mat.shape.shape = (n_entity, n_entity_dim, n_relation_dim)
            # ops.batch_dot(self.entities_emb, self.mat).shape = (n_entity, n_relation_dim)            
            head = self.entities_emb[triple[0]]
            tail = self.entities_emb[triple[2]]
            relation = self.relations_emb[triple[1]] # shape=(n_dim)
            mat = self.mat[triple[1]]
    
            entities_proj = ops.dot(self.entities_emb, mat)
            head_proj = ops.dot(head.reshape(1,-1), mat)[0]
            tail_proj = ops.dot(tail.reshape(1,-1), mat)[0]
            
            
            # 为了后续便于一一对应计算，将向量重复
            # 说明：此处使用封装的向量运算，相比使用for循环完成计算hit10等，前者速度大大提高
            relations = ms.numpy.repeat(ops.ExpandDims()(relation,0), self.n_entity, axis=0) # shape=(n_entity, n_dim)
            head_projs = ms.numpy.repeat(ops.ExpandDims()(head_proj,0), self.n_entity, axis=0)
            tail_projs = ms.numpy.repeat(ops.ExpandDims()(tail_proj,0), self.n_entity, axis=0) # shape=(n_entity, n_dim)

            # corrupt head
            # 将triple的head分别替换成各个实体，计算替换后的距离，
            # 这里使用所有实体的张量直接计算，而不是通过for循环
            corrupt_head_dists = self.dist_op(entities_proj + relations - tail_projs).sum(axis=1) # 获得各个节点的距离
            dist, index = ops.sort(corrupt_head_dists) # 排序
            ## ## hits@10 for corrupt head
            if (not self.is_filter) or (self.train_triple is None):
                # 无需filter或无训练集，直接看前10
                hits += int(triple[0] in index[:10])       # 看真实的head是否在排序后的前10中，即 hits10
            else: # filter情形，去除真实存在在训练集中的情形
                not_hit = 0
                for entity in index:
                    if not_hit > 10:
                        break
                    if entity == triple[0]:
                        hits += 1
                        break
                    cur_tri = (int(entity), int(triple[1]), int(triple[2]))
                    if cur_tri not in self.train_triple:
                        not_hit += 1
            ## mean_rank for corrupt head
            rank_sum += np.where(index.asnumpy() == triple.asnumpy()[0])[0] # 计算真实的head的实际位次

            # corrupt tail
            # 将triple的tail分别替换成各个实体，计算替换后的距离，
            # 这里使用所有实体的张量直接计算，而不是通过for循环
            corrupt_tail_dists = self.dist_op(head_projs + relations - entities_proj).sum(axis=1)
            dist, index = ops.sort(corrupt_tail_dists) 
            ## hits@10 for corrupt tail
            if (not self.is_filter) or (self.train_triple is None):
                hits += int(triple[2] in index[:10])
            else:
                not_hit = 0
                for entity in index:
                    if not_hit > 10:
                        break
                    if entity == triple[2]:
                        hits += 1
                        break
                    cur_tri = (int(triple[0]), int(triple[1]), int(entity))
                    if cur_tri not in self.train_triple:
                        not_hit += 1
            ## mean-rank for corrupt tail
            rank_sum += np.where(index.asnumpy() == triple.asnumpy()[2])[0]
            
        # 计算hits10和mean_rank
        self.hits10 = hits / 400#(2 * len(self.test_triple)) # 之前对所有entity累加head和tail，计算平均值
        self.mean_rank = rank_sum / 400#(2 * len(self.test_triple)) + 1 # +1因为下标从1开始而不是0
        return self.hits10, self.mean_rank
