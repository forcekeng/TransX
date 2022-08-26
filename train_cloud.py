# 训练脚本
import os
import time
import argparse

import moxing as mox

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from src.config import Config
from src.dataset import DataGenerator
from src.transE import TransE
from src.transD import TransD
from src.transH import TransH
from src.transR import TransR


parser = argparse.ArgumentParser(description='MindSpore Lenet Example')

# define 2 parameters for running on modelArts
# data_url,train_url是固定用于在modelarts上训练的参数，表示数据集的路径和输出模型的路径
parser.add_argument('--data_url',
                    help='path to training/inference dataset folder',
                    default='./data')

parser.add_argument('--train_url',
                    help='model folder to save/load',
                    default='./model')


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

def train(config:Config):
    """训练及模型参数配置"""
    ds = DataGenerator(config.root_dir, config.dataset, config.mode, config.n_entity)
    data_loader = ms.dataset.GeneratorDataset(ds, column_names=['positive', 'negative'], shuffle=True)
    data_loader = data_loader.batch(config.batch_size)
    print('data_loader initialized!')

    net = TransE(config.n_entity, config.n_relation, config.n_entity_dim) # n_relation_dim==n_entity_dim
    optimizer = nn.Adam(net.trainable_params(), learning_rate=config.learning_rate)
    train_net = TrainStep(net, optimizer)

    # 开始训练
    for epoch in range(config.n_epoch):
        loss = 0.0
        for (pos_triple, neg_triple) in data_loader:
            # 同一批数据多训练几次，而不是每次换一批数据
            # 避免频繁生成数据，而且通常一堆数据一次迭代不够
            # for _ in range(3):
            out = train_net(pos_triple, neg_triple)
            loss += float(out[0])
        print(f"epoch [{epoch}] : loss = {loss/len(ds.data)}")
        if (epoch+1) % 50 == 0:
            save_model(net, save_dir=config.model_save_path, commit=f"epoch{str(epoch+1)}")
    save_model(net, save_dir=config.model_save_path, commit="final")



if __name__ == '__main__':
    args = parser.parse_args()

    ######################## 将数据集从obs拷贝到训练镜像中 （固定写法）########################   
    # 在训练环境中定义data_url和train_url，并把数据从obs拷贝到相应的固定路径
    obs_data_url = args.data_url
    args.data_url = '/home/work/user-job-dir/inputs/data/'
    obs_train_url = args.train_url
    args.train_url = '/home/work/user-job-dir/outputs/model/'
    try:
        mox.file.copy_parallel(obs_data_url, args.data_url)
        print("Successfully Download {} to {}".format(obs_data_url,
                                                      args.data_url))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(
            obs_data_url, args.data_url) + str(e))
    ######################## 将数据集从obs拷贝到训练镜像中 ########################

    config = Config(
                # root_dir='E:/comptition/maoshenAI/mycode/submit/data/id_data/', 
                root_dir=f"{args.data_url}id_data/", # 对云端训练
                dataset='FB15k-237/', 
                mode='valid',
                model_save_path=args.train_url,
                norm=1, 
                n_epoch=1, 
                batch_size=512, 
                n_entity=14541, 
                n_relation=237, 
                n_entity_dim=50, 
                n_relation_dim=50)

    train(config)


    ######################## 将输出的模型拷贝到obs（固定写法） ########################   
    # 把训练后的模型数据从本地的运行环境拷贝回obs，在启智平台相对应的训练任务中会提供下载
    try:
        mox.file.copy_parallel(args.train_url, obs_train_url)
        print("Successfully Upload {} to {}".format(args.train_url,
                                                    obs_train_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(args.train_url,
                                                       obs_train_url) + str(e))
    ######################## 将输出的模型拷贝到obs ########################   

