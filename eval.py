import time
import mindspore as ms

from src.dataset import DataLoader
from val.testTransE import TestTransE
from val.testTransD import TestTransD
from val.testTransH import TestTransH
from val.testTransR import TestTransR

from src.config import global_config
from mindspore import context
context.set_context(mode=context.GRAPH_MODE, device_target=global_config.device)


def evaluate(data_dir, dataset, model, param_path, is_filter=False):
    time_start = time.time()
    if dataset.replace("-","").lower() == "fb15k":
        n_entity, n_relation = 14951, 1345
    elif dataset.lower() == "wn18":
        n_entity, n_relation = 40943, 18

    # 加载训练权重(参数字典，包含实体编码entities_emb等)
    param_dict = ms.load_checkpoint(param_path)
    
    # 加载训练数据
    test_data = DataLoader(data_dir, "test", n_entity).data

    train_data = None
    if is_filter:
        train_data = DataLoader(data_dir, "train", n_entity).data
    # 测试
    if model == "transD":        
        tester = TestTransD(n_entity, n_relation, test_data, 
                    param_dict["entities_emb"], param_dict["relations_emb"], 
                    param_dict["entities_proj"], param_dict["relations_proj"], 
                    train_triple=train_data, is_filter=is_filter)
        
    if model == "transE":
        tester = TestTransE(n_entity, n_relation, test_data, 
                    param_dict["entities_emb"], param_dict["relations_emb"],
                    train_triple=train_data, is_filter=is_filter)

    if model == "transH":
        tester = TestTransH(n_entity, n_relation, test_data, 
                    param_dict["entities_emb"], param_dict["relations_emb"], 
                    param_dict["w"],
                    train_triple=train_data, is_filter=is_filter)

    if model == "transR":
        tester = TestTransR(n_entity, n_relation, test_data, 
                    param_dict["entities_emb"], param_dict["relations_emb"], 
                    param_dict["mat"],
                    train_triple=train_data, is_filter=is_filter)
    
    hits10, mean_rank = tester.rank()
    time_end = time.time()

    print("*"*50)
    print(f"model: {model}, dataset: {dataset}, test, is_filter: {is_filter}")
    print(f"hits10 = {hits10}, mean_rank = {mean_rank}")
    print("*"*50)
    print(f"It spends {(time_end-time_start)/60} minutes!")


if __name__ == "__main__":
    # 数据所在目录,为包含 FB15K/和WN18/两个目录的目录
    # 目录均以正斜杠结尾
    data_dir=r"/dataset/data/FB15K/"    
    # 数据集:可选{"FB15K", "WN18"}     
    dataset="FB15K" 
    # 模型:可选{"transD", "transE", "transH", "transR"}
    model="transH"     
    # 训练好的参数，即实体等相关编码，为.ckpt文件
    param_path = r"./checkpoints/model_transH_epoch210000_fb15k.ckpt" # 填入训练好的权重参数,即编码
    # 是否进行过滤，即排除在训练集中出现过的一些样本
    is_filter = True
    evaluate(data_dir, dataset, model, param_path, is_filter)
