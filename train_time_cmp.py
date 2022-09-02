# 训练脚本
import os
import time
import sys
import datetime


import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.serialization import load_checkpoint, load_param_into_net


from src.config import Config
from dataset_bak import DataGenerator
from src.transE import TransE
from src.transD import TransD
from src.transH import TransH
from src.transR import TransR

from mindspore import context
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

class TrainStep(nn.TrainOneStepCell):
    """单步训练"""
    def __init__(self, net, optimizer):
        super(TrainStep, self).__init__(net, optimizer)
        self.grad_op = ops.GradOperation(get_by_list=True)
    
    def construct(self, pos_triple, neg_triple):
        loss = self.network(pos_triple, neg_triple)
        grads = self.grad_op(self.network, self.weights)(pos_triple, neg_triple)
        return loss, self.optimizer(grads)


def save_model(net, save_dir, commit=""):
    # 保存模型
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    ms.save_checkpoint(net, f"{save_dir}model_{commit}_{int(time.time())}.ckpt")

def train(config:Config, log_file=sys.stdout, pretrained=False):
    """训练及模型参数配置"""
    ds = DataGenerator(config.root_dir, config.dataset, config.mode, config.n_entity)
    data_loader = ms.dataset.GeneratorDataset(ds, column_names=['positive', 'negative'], shuffle=True)
    data_loader = data_loader.batch(config.batch_size)
    print('data_loader initialized!')

    if config.model == "transE":
        net = TransE(config.n_entity, config.n_relation, config.n_entity_dim) # n_relation_dim==n_entity_dim
    elif config.model == "transH":
        net = TransH(config.n_entity, config.n_relation, config.n_entity_dim)
    elif config.model == "transR":
        net = TransR(config.n_entity, config.n_relation, config.n_entity_dim,
                config.n_relation_dim)
    elif config.model == "transD":
        net = TransD(config.n_entity, config.n_relation, config.n_entity_dim, 
                config.n_relation_dim)
    else:
        print(f"Error: Unknown model :{config.model}, please choose one from "
                "[transE,transH, transR, transD].")
    if pretrained:
        params = load_checkpoint(config.model_pretrained_path)
        load_param_into_net(net, params)

    optimizer = nn.SGD(net.trainable_params(), learning_rate=config.learning_rate)
    train_net = TrainStep(net, optimizer)

    # 开始训练
    loss_record = []
    for epoch in range(config.n_epoch):
        loss = 0.0
        time_start = time.time()
        for n_batch, (pos_triple, neg_triple) in enumerate(data_loader):
            # 同一批数据多训练几次，而不是每次换一批数据
            # 避免频繁生成数据，而且通常一堆数据一次迭代不够
            t1 = datetime.datetime.now()
            print("t1:",t1)
            out = train_net(pos_triple, neg_triple)
            loss += float(out[0])
            t2 = datetime.datetime.now()
            print("t2:",t2)
            print(n_batch, f"one batch spend {t2 - t1} seconds!")
            print(f"epoch [{epoch}], batch [{n_batch}], loss = {float(out[0])/config.batch_size}")
        print(f"norm check:", ms.numpy.norm(net.entities_emb[pos_triple[:5,0]], axis=1))
        print(f"epoch [{epoch}] : loss = {loss/len(ds.data)}")
        print(f"this epoch spends {(time.time()-time_start)/60} minutes!\n")

        loss_record.append(loss/len(ds.data))
        if epoch % 5 == 0:
            save_model(net, config.model_save_dir, commit=f"{config.model}_epoch{str(epoch+1)}")
    save_model(net, config.model_save_dir, commit=f"{config.model}_final")
    print("loss_record:\n",loss_record, file=log_file)

if __name__ == '__main__':
    config = Config(
                root_dir='E:/comptition/maoshenAI/mycode/submit/data/id_data/', 
                # root_dir='/dataset/data/id_data/', # 对云端训练
                dataset='FB15k-237/', 
                mode='train',
                model="transE",
                model_pretrained_path="",
                model_save_dir="/model/",
                log_save_file="log.out",
                norm=1, 
                n_epoch=500, 
                batch_size=1024, 
                learning_rate=0.001,
                n_entity=14541, 
                n_relation=237, 
                n_entity_dim=50, 
                n_relation_dim=50)
    with open(config.log_save_file, "a") as log_file:
        train(config, log_file, pretrained=False)
