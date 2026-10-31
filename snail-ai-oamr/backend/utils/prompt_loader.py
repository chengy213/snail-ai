from utils.config_handler import prompts_config
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

# 闲聊提示词加载
def load_chit_chat_prompt():
    """
    加载闲聊提示词
    :return: 闲聊提示词
    """
    try:
        chit_chat_prompt_path = get_abs_path(prompts_config["chit_chat_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_chit_chat_prompt]在yaml[prompts.yml]没有chit_chat_prompt_path配置项")
        raise e  # 报错程序不继续执行

    try:
        return open(chit_chat_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_chit_chat_prompt]解析闲聊提示词出错：{str(e)}")
        raise e  # 报错程序不继续执行


# 业务问题提示词加载
def load_business_qa_prompt():
    """
    加载业务问题提示词
    :return: 业务问题提示词
    """
    try:
        business_qa_prompt_path = get_abs_path(prompts_config["business_qa_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_business_qa_prompt]在yaml[prompts.yml]没有business_qa_prompt_path配置项")
        raise e  # 报错程序不继续执行
    try:
        return open(business_qa_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_business_qa_prompt]解析业务问题提示词出错：{str(e)}")
        raise e

# 系统问题提示词加载
def load_system_issue_prompt():
    """
    加载系统问题提示词
    :return: 系统问题提示词
    """
    try:
        system_issue_prompt_path = get_abs_path(prompts_config["system_issue_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_issue_prompt]在yaml[prompts.yml]没有system_issue_prompt_path配置项")
        raise e
    try:
        return open(system_issue_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_issue_prompt]解析系统问题提示词出错：{str(e)}")
        raise e

# 数据统计提示词加载
def load_data_statistics_prompt():
    """
    加载数据统计提示词
    :return: 数据统计提示词
    """
    try:
        data_statistics_prompt_path = get_abs_path(prompts_config["data_statistics_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_data_statistics_prompt]在yaml[prompts.yml]没有data_statistics_prompt_path配置项")
        raise e
    try:
        return open(data_statistics_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_data_statistics_prompt]解析数据统计提示词出错：{str(e)}")
        raise e

# 业务操作提示词加载
def load_business_operation_prompt():
    """
    加载业务操作提示词
    :return: 业务操作提示词
    """
    try:
        business_operation_prompt_path = get_abs_path(prompts_config["business_operation_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_business_operation_prompt]在yaml[prompts.yml]没有business_operation_prompt_path配置项")
        raise e
    try:
        return open(business_operation_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_business_operation_prompt]解析业务操作提示词出错：{str(e)}")
        raise e

# 问题分类提示词加载
def load_question_classification_prompt():
    """
    加载问题分类提示词
    :return: 问题分类提示词
    """
    try:
        question_classification_prompt_path = get_abs_path(prompts_config["question_classification_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_question_classification_prompt]在yaml[prompts.yml]没有question_classification_prompt_path配置项")
        raise e
    try:
        return open(question_classification_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_question_classification_prompt]解析问题分类提示词出错：{str(e)}")
        raise e