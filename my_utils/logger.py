import logging
import os
import sys
import threading

# --------------------------------------------------
# 1. 单例元类 (保持不变)
# --------------------------------------------------
class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]

# --------------------------------------------------
# 2. 重构后的 GlobalLogger 类
# --------------------------------------------------
class GlobalLogger(metaclass=SingletonMeta):
    """
    一个与框架无关的、可移植的全局日志记录器。

    通过单例模式，确保在整个应用程序中只有一个实例。
    """
    _GLOBAL_LOGGER_NAME = "MySystemGlobalLogger"

    def __init__(self):
        """
        构造函数非常干净，只进行最基础的内部状态初始化。
        它不依赖任何外部框架。
        """
        self.logger = logging.getLogger(self._GLOBAL_LOGGER_NAME)
        self.is_configured = False

    def setup(self, log_dir: str, level: int = logging.INFO, rank: int = 0, world_size: int = 1, **kwargs):
        """
        从外部接收配置并完成日志器的设置。

        这个方法是幂等的（Idempotent），可以被安全地多次调用，但只有第一次会生效。
        它不直接调用任何分布式库，而是接收 rank 和 world_size 作为参数。

        Args:
            log_dir (str): 存放日志文件的目录。
            level (int): 日志级别。
            rank (int): 当前进程的 rank 号。
            world_size (int): 总进程数。
        """
        # 1. 防止重复配置
        if self.is_configured:
            self.logger.warning("Logger is already configured. Ignoring subsequent setup call.")
            return
        
        # 健壮性：在添加新 handler 之前，清空可能存在的旧 handler
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 2. 设置日志级别
        self.logger.setLevel(level)

        # --- 构建一个动态的日志格式字符串 ---
        log_format = f"[%(asctime)s] [Rank {rank}/{world_size}]"

        # 检查 kwargs 中是否有可选的 Megatron mpu 信息
        dp_rank = kwargs.get('data_parallel_rank')
        if dp_rank is not None:
            dp_size = kwargs.get('data_parallel_world_size', '?')
            log_format += f" [DP_Rank {dp_rank}/{dp_size}]"
        
        # 检查 kwargs 中是否有自定义的标签
        extra_label = kwargs.get('extra_log_label')
        if extra_label:
            log_format += f" [{extra_label}]"

        log_format += " [%(levelname)s] [%(funcName)s:%(lineno)d] - %(message)s"
        
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

        # 4. 配置 StreamHandler (输出到控制台)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        # 5. 配置 FileHandler (输出到文件)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"rank_{rank}.log")
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 6. 禁止向上传播日志，防止 root logger 重复打印
        self.logger.propagate = False
        
        # 7. 标记为已配置
        self.is_configured = True
        
        if rank == 0:
            self.logger.info(f"GlobalLogger configured. Log files will be saved in '{log_dir}'.")

    def get_logger(self) -> logging.Logger:
        """
        获取全局 logger 实例。

        如果 logger 尚未配置，会打印警告并提供一个默认的基础配置。
        """
        if not self.is_configured:
            # 提供一个后备的、安全的默认配置，防止在 setup 调用前使用时程序崩溃
            self.logger.warning("GlobalLogger is being used before it was properly configured. "
                               "Applying a basic default configuration.")
            
            # 仅配置一个简单的控制台输出，避免静默失败
            if not self.logger.hasHandlers():
                handler = logging.StreamHandler(sys.stdout)
                formatter = logging.Formatter("[%(asctime)s] [UNCONFIGURED] - %(message)s")
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
            # 标记为已配置(基础配置)，防止重复打印警告
            self.is_configured = True 

        return self.logger
    

# ==========================================================
# === NEW: Lightweight Global Accessor Function ===
# ==========================================================
def get_global_logger() -> logging.Logger:
    """
    A lightweight, global access point to the GlobalLogger singleton.

    This is the recommended way to get the logger from anywhere in the project.
    It hides the complexity of the singleton class and provides a simple,
    memorable function call.
    """
    return GlobalLogger().get_logger()