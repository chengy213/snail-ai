import yaml

from utils.path_tool import get_abs_path


# 提示词配置加载
def load_prompts_config(config_path: str=get_abs_path("config/prompts.yml"),encoding="utf-8"):
    """
    加载提示词配置
    :param config_path: 配置文件路径
    :param encoding: 编码
    :return: 配置文件内容
    """
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

# 系统智能助手配置加载
def load_oamr_config(config_path: str=get_abs_path("config/oamr.yml"),encoding="utf-8"):
    """
    加载系统智能助手配置
    :param config_path: 配置文件路径
    :param encoding: 编码
    :return: 配置文件内容
    """
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

# 向量库配置加载
def load_vector_store_config(config_path: str=get_abs_path("config/vector_store.yml"),encoding="utf-8"):
    """
    加载向量库配置
    :param config_path: 配置文件路径
    :param encoding: 编码
    :return: 配置文件内容
    """
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

prompts_config = load_prompts_config()
oamr_config = load_oamr_config()
vector_store_config = load_vector_store_config()