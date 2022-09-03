# 训练脚本
import os
import time
import sys

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.serialization import load_checkpoint, load_param_into_net


from src.config import Config
from src.dataset import DataLoader
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

    data_loader = DataLoader(config.root_dir, config.dataset, config.mode, config.n_entity)
    print('data_loader initialized!')
    optimizer = nn.SGD(net.trainable_params(), learning_rate=config.learning_rate)
    train_net = TrainStep(net, optimizer)

    # 开始训练
    loss_record = []
    loss = 0.0
    time_start = time.time()
    for epoch in range(config.n_epoch):
        pos_triple, neg_triple = data_loader.get_batch_data(batch_size=1024)
        out = train_net(pos_triple, neg_triple)
        loss += float(out[0])
        if (epoch+1) % 1000 == 0:
            print(f"epoch [{epoch}]") # 输出到控制台
            print(f"1000 iterations spends {(time.time()-time_start)/60} minutes!\n")
            print(f"loss average = {loss / 1000}")
            
            print(f"1000 iterations spends {(time.time()-time_start)/60} minutes!\n", file=log_file)
            print(f"loss average = {loss / 1000}", file=log_file)
            loss_record.append(loss)
            loss = 0.0
            time_start = time.time()
        if (epoch+1) % 10000 == 0:
            save_model(net, config.model_save_dir, commit=f"{config.model}_epoch{str(epoch+1)}")
    save_model(net, config.model_save_dir, commit=f"{config.model}_final")
    print("loss_record:\n",loss_record, file=log_file)


if __name__ == '__main__':
    config = Config(
                # root_dir='E:/comptition/maoshenAI/mycode/submit/data/id_data/', 
                root_dir='/dataset/data/', # 对云端训练
                dataset='FB15K/', 
                mode='train',
                model="transR",
                model_pretrained_path="",
                model_save_dir="/model/",
                log_save_file="/model/fb15k_logR.out",
                norm=1, 
                n_epoch=1000000, 
                batch_size=4096, 
                learning_rate=0.001,
                n_entity=14951, 
                n_relation=1345, 
                n_entity_dim=50, 
                n_relation_dim=50)
    with open(config.log_save_file, "a") as log_file:
        train(config, log_file, pretrained=False)
        

