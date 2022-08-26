# 训练脚本
from operator import mod
import os
import time

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from src.config import Config
from src.dataset import DataGenerator
from src.transE import TransE
from src.transD import TransD
from src.transH import TransH
from src.transR import TransR

from mindspore import context
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class TrainStep(nn.TrainOneStepCell):
    """单步训练"""
    def __init__(self, net, optimizer):
        super(TrainStep, self).__init__(net, optimizer)
        self.grad_op = ops.GradOperation(get_by_list=True)
    
    def construct(self, pos_triple, neg_triple):
        loss = self.network(pos_triple, neg_triple)
        grads = self.grad_op(self.network, self.weights)(pos_triple, neg_triple)
        return loss, self.optimizer(grads)


def save_model(net, commit=""):
    # 保存模型
    save_dir = '/model/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    ms.save_checkpoint(net, f"{save_dir}model_{commit}_{int(time.time())}.ckpt")

def train(config:Config):
    """训练及模型参数配置"""
    ds = DataGenerator(config.root_dir, config.dataset, config.mode, config.n_entity)
    data_loader = ms.dataset.GeneratorDataset(ds, column_names=['positive', 'negative'], shuffle=True)
    data_loader = data_loader.batch(config.batch_size)
    print('data_loader initialized!')

    if config.model == "transE":
        net = TransE(config.n_entity, config.n_relation, config.n_entity_dim) # n_relation_dim==n_entity_dim
    elif config.model == "transH":
        net = TransH(config.n_entity, config.n_relation, config.n_entity_dim)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=config.learning_rate)
    train_net = TrainStep(net, optimizer)

    # 开始训练
    loss_record = []
    for epoch in range(config.n_epoch):
        loss = 0.0
        for (pos_triple, neg_triple) in data_loader:
            # 同一批数据多训练几次，而不是每次换一批数据
            # 避免频繁生成数据，而且通常一堆数据一次迭代不够
            # for _ in range(3):
            out = train_net(pos_triple, neg_triple)
            loss += float(out[0])
        print(f"epoch [{epoch}] : loss = {loss/len(ds.data)}")
        loss_record.append(loss)
        if (epoch+1) % 50 == 0:
            save_model(net, commit=f"epoch{str(epoch+1)}")
    save_model(net, commit="final")
    print(loss_record)

if __name__ == '__main__':
    config = Config(
                # root_dir='E:/comptition/maoshenAI/mycode/submit/data/id_data/', 
                root_dir='/dataset/data/id_data/', # 对云端训练
                dataset='FB15k-237/', 
                mode='valid',
                model="transH",
                model_save_path="/model/",
                norm=1, 
                n_epoch=3, 
                batch_size=512, 
                n_entity=14541, 
                n_relation=237, 
                n_entity_dim=50, 
                n_relation_dim=50)

    train(config)