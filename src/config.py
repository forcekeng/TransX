class Config:
    """配置训练相关参数"""
    def __init__(self, root_dir:str, dataset:str, 
                mode:str, norm:int, 
                n_epoch:int, batch_size:int, learning_rate=0.01,
                n_entity=13781, n_relation=237, n_entity_dim=50, 
                n_relation_dim=50,
                model_save_path="./net.ckpt"):
        self.root_dir = root_dir
        self.dataset = dataset
        self.mode = mode
        self.norm = norm
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.n_entity_dim = n_entity_dim
        self.n_relation_dim = n_relation_dim
        self.learning_rate = learning_rate
        self.model_save_path = model_save_path

    def save_config(self, only_net=True):
        """
        only_net: bool : 仅保留网络相关参数，而不保存文件夹路径 root_dir等
        """
        pass