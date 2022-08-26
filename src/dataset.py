# 数据集处理定义
import os
import random


class DataGenerator:
    """读取关系数据的数据集类"""
    def __init__(self, root_dir, dataset, mode, n_entity):
        """
        root_dir: 数据所在根目录
        dataset: string, 数据集，可选 {'FB15k-237', 'WN18RR'}
        mode: string 数据模式，可选 {'train', 'train_valid', 'valid', 'test'}，其中 'train_valid'
            表示将训练集和验证集合并
        """
        self.root_dir = root_dir
        self.dataset = dataset
        self.mode = mode
        self.n_entity = n_entity
        
        self.data = self._read_data()
        
    def _read_data(self):
        data = []
        for m in self.mode.split('_'):
            path = os.path.join(self.root_dir, self.dataset, self.mode+".txt")
            print(f"read file path: {path}")
            with open(path) as f:
                lines = f.readlines()
                for trituple in lines:
                    head, rel, tail = trituple.split('\t')
                    data.append((int(head.strip()), int(rel.strip()), int(tail.strip())))
        return data
    
    def _corrupt_trituple(self, trituple):
        """制造错误的关系
        trituple: 正常的三元组
        return: 错误（corrupted）的三元组
        """
        corrupt_tri = trituple
        count = 0  # 设定重复次数，防止数据错误导致死循环
        while (corrupt_tri in self.data and count<100):
            (h,r,t) = corrupt_tri
            if random.random() < 0.5: # random.random() 生成 [0,1]之间的随机数
                corrupt_h = random.randint(0, self.n_entity-1)
                corrupt_tri = (corrupt_h, r, t)
            else:
                corrupt_t = random.randint(0, self.n_entity-1)
                corrupt_tri = (h, r, corrupt_t)
        return corrupt_tri
                
        
    def __getitem__(self, index):
        """自定义随机访问函数
        index: 索引下表
        return: 包含正常和corrupted数据的元组，形如((h,r,t), (ch, cr, ct))
        注：对测试集等不需要corrupted数据，直接忽略第2项即可
        """
        normal_tri = self.data[index]
        corrupt_tri = self._corrupt_trituple(normal_tri)
        return (normal_tri, corrupt_tri)
    
    def __len__(self):
        return len(self.data)


def get_batch_dataloader(ds: DataGenerator, batch_size:int, shuffle=True):
    import mindspore
    data_loader = mindspore.dataset.GeneratorDataset(ds, column_names=['positive', 'negative'], shuffle=shuffle)
    data_loader = data_loader.batch(batch_size)
