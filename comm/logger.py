import logging
import sys

class Logger:
    def __init__(self, name='TrainerLogger', level=logging.INFO, stream=sys.stdout):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False  # 防止重复打印
        
        # 如果没有Handler才添加，防止Jupyter环境反复run多个handler叠加
        if not self.logger.handlers:
            handler = logging.StreamHandler(stream)
            formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(name)s]: %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)

    def print_model_params(self, model):
        total_params = 0
        self.info("模型参数详情:")
        self.info("参数名称              参数形状                 参数数量")
        self.info("-" * 60)
        for name, param in model.named_parameters():
            param_num = param.numel()
            total_params += param_num
            self.info(f"{name:20} {list(param.shape)!s:25} {param_num:10,}")
        self.info("-" * 60)
        self.info(f"模型总参数数量: {total_params:,}")