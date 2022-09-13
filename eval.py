import mindspore as ms

from src.config import Config
from src.dataset import DataLoader

from val.testTransE import TestTransE
from val.testTransD import TestTransD
from val.testTransH import TestTransH
from val.testTransR import TestTransR


def evaluate(data_dir, dataset, model, param_path, is_filter=False):
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

    print("*"*50)
    print(f"model: {model}, dataset: {dataset}, test, is_filter: {is_filter}")
    print(f"hits10 = {hits10}, mean_rank = {mean_rank}")
    print("*"*50)



if __name__ == "__main__":
    # 数据所在目录,为包含 FB15k-237/和WN18RR/两个目录的目录
    # 目录均以正斜杠结尾
    data_dir=r"E:/comptition/maoshenAI/mycode/submit/data/id_data/"    
    # 数据集:可选{"FB15k", "WN18"}     
    dataset="FB15k" 
    # 模型:可选{"transD", "transE", "transH", "transR"}
    model="transH"     
    # 训练好的参数，即实体等相关编码，为.ckpt文件
    param_path = r"E:/comptition/maoshenAI/mycode/checkpoints/model_transH_epoch210000_fb15k.ckpt" # 填入训练好的权重参数,即编码
    evaluate(data_dir, dataset, model, param_path)
