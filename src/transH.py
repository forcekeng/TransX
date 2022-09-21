import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

import numpy as np

class TransH(nn.Cell):
    """TransH模型"""
    def __init__(self, n_entity, n_relation, n_dim, margin=1.0, norm=1):
        super(TransH, self).__init__()
        self.n_entity = n_entity     # 数据集[包含训练、验证和测试集]实体数
        self.n_relation = n_relation  # 数据集[包含训练、验证和测试集]关系类型数

        self.n_dim = n_dim   # 编码维度
        self.norm = norm     # 所用范数
        self.margin = margin # 算法中参数
       
        self.normalizer = ops.L2Normalize(axis=-1)  # 归一化器
        self.abs = ops.Abs()
        self.maximum = ops.Maximum()
        self.square = ops.Square()

        # 实体编码
        uniformreal = ops.UniformReal(seed=1)  # 正态分布生成器
        self.entities_emb = ms.Parameter(self.normalizer(uniformreal((n_entity, n_dim))), name='entities_emb')
        # 关系编码
        self.relations_emb = ms.Parameter(self.normalizer(uniformreal((n_relation, n_dim))), name="relations_emb")
        # 投影方向向量w
        self.w = ms.Parameter(self.normalizer(uniformreal((n_relation, n_dim))), name="w")
    

    def construct(self, pos_triple, neg_triple):
        """
        pos_triple: ms.Tensor : shape=(batch_size, 3, n_dim)
        neg_triple: ms.Tensor : shape=(batch_size, 3, n_dim)
        """
        # 归一化
        self.entities_emb.set_data(self.normalizer(self.entities_emb))
        self.relations_emb.set_data(self.normalizer(self.relations_emb))
        self.w.set_data(self.normalizer(self.w))

        # 取出正、负数样本编码向量
        pos_head, pos_relation, pos_tail = self.embed(pos_triple) # shape = (batch_size, n_dim)
        neg_head, neg_relation, neg_tail = self.embed(neg_triple)
        
        # 计算距离
        pos_distance = self.get_distance(pos_head, pos_relation, pos_tail, self.norm) # shape = (batch_size)
        neg_distance = self.get_distance(neg_head, neg_relation, neg_tail, self.norm)
        
        # 计算损失
        loss = self.maximum(0, pos_distance - neg_distance + self.margin).sum() # 所有损失求和
        return loss
    
    def embed(self, triple):
        """获得编码向量"""
        # 自身归一化
        head = self.entities_emb[triple[:, 0]]
        relation = self.relations_emb[triple[:, 1]]
        tail = self.entities_emb[triple[:, 2]]
        proj_w = self.w[triple[:,1]]
        
        # 计算投影后向量
        head = self._projection(head, proj_w)
        tail = self._projection(tail, proj_w)
                      
        return head, relation, tail 


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
    

    def _projection(self, entity_emb, proj_vec):
        """投影
        """
        return entity_emb - ops.batch_dot(entity_emb, proj_vec) * proj_vec
        