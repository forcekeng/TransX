# 数据集处理定义
import os
import numpy as np
import mindspore as ms

class DataLoader:
    """读取关系数据的数据集类"""
    def __init__(self, root_dir, mode, n_entity):
        """
        root_dir: 数据所在根目录
        mode: string 数据模式，可选 {'train', 'train_valid', 'valid', 'test'}，其中 'train_valid'
            表示将训练集和验证集合并
        as_tensor: 是否将数据用ms.Tensor表示，测试时不需要。
        """
        self.root_dir = root_dir
        self.mode = mode
        self.n_entity = n_entity
        
        self.data = self._read_data()
        
    def _read_data(self):
        data = []
        for m in self.mode.split('_'):
            path = os.path.join(self.root_dir, self.mode+"2id.txt")
            print(f"read file path: {path}")
            with open(path) as f:
                lines = f.readlines()
                for trituple in lines:
                    head, tail, rel = trituple.split(' ')
                    data.append((int(head.strip()), int(rel.strip()), int(tail.strip())))
        return ms.Tensor(data, dtype=ms.dtype.int32)
    
    def get_batch_data(self, batch_size=1024, random=True):
        """随机获取一批数据
        """
        indexes = np.random.randint(0, len(self.data), size=batch_size, dtype=np.int32)
        data = self.data.asnumpy()
        pos_data = data[indexes] # 随机选取的样本,当作正样本
        # 将样本corrupt,将其head和tail随机替换,变成负样本
        # 尽管可能导致少量负样本其实为正样本,但因为其概率小,而且多次出现概率极低,可忽略
        neg_data = pos_data.copy()
        if np.random.uniform() > 0.5:
            corrupt_head = np.random.randint(0, self.n_entity, size=batch_size, dtype=np.int32)
            neg_data[:, 0] = corrupt_head
        else:
            corrupt_tail = np.random.randint(0, self.n_entity, size=batch_size, dtype=np.int32)
            neg_data[:, 2] = corrupt_tail
        return ms.Tensor(pos_data), ms.Tensor(neg_data)
        
