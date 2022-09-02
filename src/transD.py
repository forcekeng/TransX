import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

import numpy as np


class TransD(nn.Cell):
    def __init__(self, n_entity, n_relation, n_entity_dim, n_relation_dim, margin=1.0, norm=1):
        super(TransD, self).__init__()
        self.n_entity = n_entity     # 数据集[包含训练、验证和测试集]实体数
        self.n_relation = n_relation  # 数据集[包含训练、验证和测试集]关系类型数
        self.norm = norm     # 损失函数所用范数
        self.margin = margin # 算法中参数

        self.n_entity_dim = n_entity_dim   # 实体编码维度
        self.n_relation_dim = n_relation_dim   # 关系编码维度，仅考虑其等于实体编码维度的情形
        # 实体编码
        uniformreal = ops.UniformReal(seed=1)
        self.normalizer = ops.L2Normalize(axis=-1)
        self.entities_emb = ms.Parameter(self.normalizer(uniformreal((n_entity, n_entity_dim))), name='entities_emb')
        # 关系编码
        self.relations_emb = ms.Parameter(self.normalizer(uniformreal((n_relation, n_relation_dim))), name="relations_emb")
        # 实体映射向量
        self.entities_proj = ms.Parameter(self.normalizer(uniformreal((n_entity, n_entity_dim))), name="entities_proj")
        # 关系映射向量
        self.relations_proj = ms.Parameter(self.normalizer(uniformreal((n_relation, n_relation_dim))), name="relations_proj")

        
    def construct(self, pos_triple, neg_triple):
        """
        pos_triple: ms.Tensor : shape=(batch_size, 3, n_dim)
        neg_triple: ms.Tensor : shape=(batch_size, 3, n_dim)
        """
        # 取出正、负数样本编码向量
        pos_head, pos_relation, pos_tail = self.embed(pos_triple) # shape = (batch_size, n_dim)
        neg_head, neg_relation, neg_tail = self.embed(neg_triple)
        
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

        # 对head和tail进行映射
        head_proj = self.entities_proj[triple[:, 0]] # 获取映射矩阵
        relation_proj = self.relations_proj[triple[:, 1]]
        tail_proj = self.entities_proj[triple[:, 2]]
        head = self._project(head, head_proj, relation_proj) # 映射结果
        tail = self._project(tail, tail_proj, relation_proj)

        # 归一化
        self.entities_proj[triple[:, 0]] = self.normalizer(self.entities_proj[triple[:, 0]])
        self.entities_proj[triple[:, 2]] = self.normalizer(self.entities_proj[triple[:, 2]])
        self.relations_proj[triple[:, 1]] = self.normalizer(self.relations_emb[triple[:, 1]])

        return head, relation, tail 


    def _project(self, entity_emb, entity_proj, relation_proj):
        """将实体向量进行映射
        entity_emb: ms.Tensor : shape=(batch_size, n_entity_dim)
        entity_proj: ms.Tensor : shape=(batch_size, n_entity_dim)
        relation_proj: ms.Tensor : shape=(batch_size, n_relation_dim)
        return: ms.Tensor : shape=(batch_size, n_relation_dim==n_entity_dim)
        """
        # assert self.n_entity_dim == self.n_relation_dim # 仅仅考虑编码长度相同的情形
        return entity_emb + ops.batch_dot(entity_emb, entity_proj) * relation_proj


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