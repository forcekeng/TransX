# 训练脚本
import os
import time
import sys

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.dataset import DataLoader
from src.transE import TransE
from src.transD import TransD
from src.transH import TransH
from src.transR import TransR
from src.config import fb15k_config, wn18_config, global_config

from mindspore import context
context.set_context(mode=context.GRAPH_MODE, device_target=global_config.device)


class TrainStep(nn.TrainOneStepCell):
    """单步训练"""
    def __init__(self, net, optimizer):
        super(TrainStep, self).__init__(net, optimizer)
        self.grad_op = ops.GradOperation(get_by_list=True)
    
    def construct(self, pos_triple, neg_triple):
        loss = self.network(pos_triple, neg_triple)
        grads = self.grad_op(self.network, self.weights)(pos_triple, neg_triple)
        return loss, self.optimizer(grads)


def save_model(net:nn.Cell, save_dir:str, commit=""):
    """
    net: 需要保存的模型
    save_dir: 保存目录
    commit: 模型说明，附加在模型名称后面
    """
    # 保存模型
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    ms.save_checkpoint(net, f"{save_dir}model_{commit}_{int(time.time())}.ckpt")


def train(data_dir, model, config):
    """训练及模型参数配置"""
    if model == "transE":
        net = TransE(config.n_entity, config.n_relation, 
                        config.transE.n_dim,
                        margin=config.transE.margin)
    elif model == "transH":
        net = TransH(config.n_entity, config.n_relation, 
                        config.transH.n_dim,
                        margin=config.transH.margin)
    elif model == "transR":
        net = TransR(config.n_entity, config.n_relation,
                        config.transR.n_entity_dim, 
                        config.transR.n_relation_dim, 
                        margin=config.transR.margin)
    elif model == "transD":
        net = TransD(config.n_entity, config.n_relation, 
                        config.transD.n_entity_dim, 
                        config.transD.n_relation_dim, 
                        margin=config.transD.margin)
    else:
        print(f"Error: Unknown model :{model}, please choose one from "
                "[transE,transH, transR, transD].")
    if config.pretrained:
        params = load_checkpoint(config.pre_model_path)
        load_param_into_net(net, params)

    data_loader = DataLoader(data_dir, mode="train", n_entity=config.n_entity)
    print('data_loader initialized!')

    optimizer = nn.SGD(net.trainable_params(), learning_rate=config.learning_rate)
    train_net = TrainStep(net, optimizer)

    # 开始训练
    print("<=================== 开始训练 ===================>")
    loss_record = []
    loss = 0.0
    time_start = time.time()
    for epoch in range(config.iterations):
        pos_triple, neg_triple = data_loader.get_batch_data(batch_size=1024)
        out = train_net(pos_triple, neg_triple)
        loss += float(out[0].asnumpy())
        if (epoch+1) % 1000 == 0:
            print(f"epoch [{epoch+1}], loss = {loss / 1000}, 1000 iterations spend "\
                    f"{(time.time()-time_start)/60} minutes!")
            
            with open(config.log_save_file, 'a') as f:
                print(f"epoch [{epoch+1}], loss = {loss / 1000}, 1000 iterations spend "\
                    f"{(time.time()-time_start)/60} minutes!", file=f)
            loss_record.append(loss/1000)
            loss = 0.0
            time_start = time.time()
        if (epoch+1) % 10000 == 0:
            save_model(net, config.model_save_dir, 
                        commit=f"{model}_epoch{str(epoch+1)}")
    save_model(net, config.model_save_dir,
                        commit=f"{model}_final")
    print("loss_record:\n",loss_record) # 打印训练损失
    print(">=================== 训练结束 ===================<")


if __name__ == '__main__':
    dataset = "fb15k"   # 数据，可选 {"fb15k", "wn18"}
    data_dir = "/dataset/data/FB15K/" # 数据集所在文件夹，其下包含train2id.txt等文件
    model = "transE"    # 模型，可选 {"transD", "transE", "transH", "transR"}
    
    print(f"dataset = {dataset}, model = {model}")
    # 其他配置请参阅 src/config.py 文件
    if dataset.lower() == "fb15k":
        config = fb15k_config
    elif dataset.lower() == "wn18":
        config = wn18_config
    config.model = model

    train(data_dir, model, config)

