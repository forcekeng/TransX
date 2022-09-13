from easydict import EasyDict as ed

global_config = ed({
    "device": "GPU"            # 硬件平台，可选{"CPU", "GPU", "Ascend"}
})

fb15k_config = ed({
    # 全局参数
    "model": "transE",              # 模型，可选 {"transD", "transE", "transH", "transR"}
    "n_entity": 14951,              # 实体数量
    "n_relation": 1345,             # 关系数量
    
    # 训练参数
    "pre_model_path": "",        # 预训练模型保存路径
    "pretrained": False,         # 是否使用预训练模型，为True时需要pre_model_path指定正确路径
    "model_save_dir": "checkpoints/",   # 模型保存路径
    "log_save_file": "log.out",  # 训练日志保存路径
    "iterations": 1000000,      # 最大迭代次数
    "batch_size": 4096,         # 批处理大小
    "learning_rate": 0.001,                # 学习率
    
    # 模型参数
    ## transD参数
    "transD": {
        "n_entity_dim": 50,         # 实体编码维度
        "n_relation_dim": 50,       # 关系编码维度
        "margin": 1.0,              # 算法中计算损失时参数
        "norm": 1                   # 计算损失所用范数，可选{1,2}
    },
    ## transE参数
    "transE": {     
        "n_dim": 50,        # 编码维度，实体编码维度==关系编码维度
        "margin": 1.0,      # 算法中计算损失时参数
        "norm": 1           # 计算损失所用范数
    },
    ## transH参数
    "transH": {
        "n_dim": 100,       # 编码维度，实体编码维度==关系编码维度
        "margin": 1.0,      # 算法中计算损失时参数
        "norm": 1           # 计算损失所用范数
    },
    ## transR参数
    "transR": {
        "n_entity_dim": 100,        # 实体编码维度
        "n_relation_dim": 100,      # 关系编码维度
        "margin": 1.0,              # 算法中计算损失时参数
        "norm": 1                   # 计算损失所用范数
    }
})



wn18_config = ed({
    # 全局参数
    "model": "transE",
    "n_entity": 40943,              # 实体数量
    "n_relation": 18,               # 关系数量

    # 训练参数
    "pre_model_path": "",        # 预训练模型保存路径
    "pretrained": False,         # 是否使用预训练模型，为True时需要pre_model_path指定正确路径
    "model_save_dir": "checkpoints/",   # 模型保存路径
    "log_save_file": "log.out",  # 训练日志保存路径
    "iterations": 1000000,      # 最大迭代次数
    "batch_size": 4096,         # 批处理大小
    "learning_rate": 0.001,                # 学习率

    # 模型参数
    ## transD参数
    "transD": {
        "n_entity_dim": 50,         # 实体编码维度
        "n_relation_dim": 50,       # 关系编码维度
        "margin": 1.0,              # 算法中计算损失时参数
        "norm": 1                   # 计算损失所用范数，可选{1,2}
    },
    ## transE参数
    "transE": {     
        "n_dim": 50,        # 编码维度，实体编码维度==关系编码维度
        "margin": 1.0,      # 算法中计算损失时参数
        "norm": 1           # 计算损失所用范数
    },
    ## transH参数
    "transH": {
        "n_dim": 50,       # 编码维度，实体编码维度==关系编码维度
        "margin": 1.0,      # 算法中计算损失时参数
        "norm": 1           # 计算损失所用范数
    },
    ## transR参数
    "transR": {
        "n_entity_dim": 30,         # 实体编码维度
        "n_relation_dim": 30,       # 关系编码维度
        "margin": 1.0,              # 算法中计算损失时参数
        "norm": 1                   # 计算损失所用范数
    }
})
