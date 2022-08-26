import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

import numpy as np


class TransR(nn.Cell):
    def __init__(self, n_entity, n_relation, n_entity_dim, n_relation_dim, margin=1.0, norm=1):
        super(TransR, self).__init__()
        self.n_entity = n_entity     # 数据集[包含训练、验证和测试集]实体数
        self.n_relation = n_relation  # 数据集[包含训练、验证和测试集]关系类型数
        self.norm = norm     # 所用范数
        self.margin = margin # 算法中参数
        self.n_entity_dim = n_entity_dim   # 实体编码维度
        self.n_relation_dim = n_relation_dim   # 关系编码维度
        
        # 实体编码
        uniformreal = ops.UniformReal(seed=1)  # 正态分布生成器
        self.entities_emb = ms.Parameter(uniformreal((n_entity, n_entity_dim)), name='entities_emb')
        # 关系编码
        self.relations_emb = ms.Parameter(uniformreal((n_relation, n_relation_dim)), name="relations_emb")
        # 实体映射到关系的矩阵，每个实体都有自己的映射矩阵
        self.mat = ms.Parameter(uniformreal((n_entity, n_entity_dim, n_relation_dim)), name="mat")
        

    def construct(self, pos_triple, neg_triple):
        """
        pos_triple: ms.Tensor : shape=(batch_size, 3, n_dim)
        neg_triple: ms.Tensor : shape=(batch_size, 3, n_dim)
        """
        # 取出正、负数样本编码向量
        pos_head, pos_relation, pos_tail = self.embed[pos_triple] # shape = (batch_size, n_dim)
        neg_head, neg_relation, neg_tail = self.embed[neg_triple]
        
        # 计算距离
        pos_distance = self.get_distance(pos_head, pos_relation, pos_tail, self.norm) # shape = (batch_size)
        neg_distance = self.get_distance(neg_head, neg_relation, neg_tail, self.norm)
        
        # 计算损失
        loss = ops.maximum(0, pos_distance - neg_distance + self.margin).sum() # 所有损失求和
        return loss
    
    def embed(self, triple):
        """获得编码向量"""
        head = self.entities_emb[triple[:, 0]]
        relation = self.relations_emb[triple[:, 1]]
        tail = self.entities_emb[triple[:, 2]]

        # 标准化，只使用二范数标准化
        normalizer = ops.L2Normalize(axis=-1)
        head = normalizer(head)
        relation = normalizer(relation)
        tail = normalizer(tail)

        # 对head和tail进行映射
        head_proj_mat = self.mat[triple[:, 0]] # 获取映射矩阵
        tail_proj_mat = self.mat[triple[:, 2]]
        head = self._project(head, head_proj_mat) # 映射结果
        tail = self._project(tail, tail_proj_mat)
        return head, relation, tail 

    def _project(self, entity_emb, proj_mat):
        """将实体向量进行映射
        entity_emb: ms.Tensor : shape=(batch_size, n_entity_dim)
        proj_mat: ms.Tensor : shape=(batch_size, n_entity_dim, n_relation_dim)
        return: ms.Tensor : shape=(batch_size, n_relation_dim)
        """
        return ops.batch_dot(entity_emb, proj_mat)


    def get_distance(self, head, relation, tail, norm=1):
        """计算距离
        head: ms.Tensor : shape=(batch_size, n_dim)
        relation: ms.Tensor : shape=(batch_size, n_dim)
        tail: ms.Tensor : shape=(batch_size, n_dim)
        return: ms.Tensor : shape=(batch_size)
        """
        if norm == 1:
            return ops.abs(head + relation - tail).sum(axis=1) # L1距离
        return ops.square(head + relation - tail).sum(axis=1) # L2距离