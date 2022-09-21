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
        
        self.normalizer = ops.L2Normalize(axis=-1)  # 归一化器

        self.abs = ops.Abs()
        self.maximum = ops.Maximum()
        self.square = ops.Square()
        uniformreal = ops.UniformReal(seed=1)  # 正态分布生成器

        # 实体编码
        self.normalizer = ops.L2Normalize(axis=-1)
        self.entities_emb = ms.Parameter(self.normalizer(uniformreal((n_entity, n_entity_dim))), name='entities_emb')
        # 关系编码
        self.relations_emb = ms.Parameter(self.normalizer(uniformreal((n_relation, n_relation_dim))), name="relations_emb")
        # 实体映射到关系的矩阵，根据每个关系都有自己的映射矩阵
        self.mat = ms.Parameter(self._init_mat(), name="mat")

    def _init_mat(self):
        mat = np.zeros(shape=(self.n_relation, self.n_entity_dim, self.n_relation_dim), dtype=np.float32)
        for i in range(self.n_relation):
            mat[i] = np.eye(self.n_entity_dim, self.n_relation_dim, dtype=mat.dtype)
        return ms.Tensor(mat)


    def construct(self, pos_triple, neg_triple):
        """
        pos_triple: ms.Tensor : shape=(batch_size, 3, n_dim)
        neg_triple: ms.Tensor : shape=(batch_size, 3, n_dim)
        """
        # 归一化
        self.entities_emb.set_data(self.normalizer(self.entities_emb))
        self.relations_emb.set_data(self.normalizer(self.relations_emb))

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
        head_raw = self.entities_emb[triple[:, 0]]
        tail_raw = self.entities_emb[triple[:, 2]]
        relation = self.relations_emb[triple[:, 1]]
        # 对head和tail进行映射
        rel_proj_mat = self.mat[triple[:, 1]] # 获取映射矩阵
        head_p = self._project(head_raw, rel_proj_mat) # 映射结果
        tail_p = self._project(tail_raw, rel_proj_mat)

        head = self.normalizer(head_p)
        tail = self.normalizer(tail_p)
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
            return self.abs(head + relation - tail).sum(axis=1) # L1距离
        return self.square(head + relation - tail).sum(axis=1) # L2距离