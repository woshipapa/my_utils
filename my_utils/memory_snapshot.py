import torch
import torch.distributed as dist
import os
import logging
import time

# --- Fallback Logger (以防万一) ---
_default_logger = logging.getLogger("MemorySnapshotter")
if not _default_logger.handlers:
    _default_logger.addHandler(logging.StreamHandler())
    _default_logger.setLevel(logging.INFO)

class MemorySnapshotter:
    """
    一个层级（堆栈感知）的内存快照工具, 
    使用 start/stop API 包装 PyTorch 的 _record_memory_history。
    
    工作流程:
    start("A"):
      start("B"):
      stop("B"): -> 立即保存一个 "B.pt" 快照 (包含 A+B 的历史)
    stop("A"):  -> 立即保存一个 "A.pt" 快照 (包含 A+B 的历史)
    
    (注意: "A.pt" 包含了 "B.pt" 的所有内容，你需要 *diff* 这两个文件来隔离 B 之后、A 之前的操作。)
    """
    def __init__(self, save_dir="memory_snapshots"):
        self.is_enabled = os.environ.get("ENABLE_MEMORY_SNAPSHOT", "0") == "1"
        self.save_dir = save_dir
        self.logger = _default_logger
        
        if self.is_enabled:
            if not (torch.cuda.is_available()):
                self.logger.warning(
                    "ENABLE_MEMORY_SNAPSHOT=1 但 torch.cuda.is_available() 是 False. 正在禁用."
                )
                self.is_enabled = False
            else:
                os.makedirs(self.save_dir, exist_ok=True)
                self.logger.info(
                    f"MemorySnapshotter is ENABLED. "
                    f"快照将保存到: {self.save_dir}"
                )
        else:
            self.logger.info("MemorySnapshotter is DISABLED.")

        # [!!] 我们的核心: 一个堆栈, 就像 MyTimer V2 [!!]
        self._active_names = [] 

    def set_logger(self, logger):
        """注入一个外部 logger, 就像 MyTimer 一样。"""
        self.logger = logger
    
    def start(self, name: str, max_entries=100000):
        """
        开始(或继续)记录内存历史。
        如果这是第一个 'start' (堆栈为空), 它会启动全局记录器。
        """
        if not self.is_enabled:
            return
        
        try:
            if not self._active_names:
                # [!!] 堆栈为空, 这是 "根" 调用 [!!]
                # 启动并*重置*历史
                torch.cuda.memory._record_memory_history(
                    enabled=True
                )
                self.logger.info(f"[MemSnapshot] START-ROOT: '{name}' (已启动并重置历史)")
            else:
                # 已经是子调用, 确保记录器仍在运行
                torch.cuda.memory._record_memory_history(enabled=True)
                self.logger.info(f"[MemSnapshot] START-CHILD: '{name}'")
            
            # 压入堆栈
            self._active_names.append(name)
            
        except Exception as e:
            self.logger.error(f"[MemSnapshot] FAILED to start recording for '{name}': {e}")
            self.is_enabled = False # 禁用以避免垃圾邮件

    def stop(self, name: str):
        """
        停止一个计时器, *立即* 将 *到目前为止* 的所有历史转储(dump)到一个 .pt 文件。
        如果这是最后一个 'stop' (堆栈变空), 它会停止全局记录器。
        """
        if not self.is_enabled:
            return

        # --- (与 MyTimer V2 相同的堆栈检查逻辑) ---
        if not self._active_names or self._active_names[-1] != name:
            self.logger.warning(
                f"[MemSnapshot] Mismatched STOP call! "
                f"Expected '{self._active_names[-1] if self._active_names else 'None'}' "
                f"but got '{name}'."
            )
            # 健壮性: 如果我们能找到它, 就弹出它
            if name not in self._active_names:
                self.logger.error(f"[MemSnapshot] FAILED to stop '{name}'. 不在活动堆栈中。")
                return
            # 弹出所有子节点, 直到找到它
            while self._active_names.pop() != name:
                pass
        else:
            # 匹配, 正常弹出
            self._active_names.pop()
        # --- (堆栈检查结束) ---

        try:
            # 1. 生成文件名
            time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            rank = torch.distributed.get_rank() if dist.is_initialized() else 0
            
            # 格式: [save_dir]/[name]__[time]__rank[rank].pt
            filename = os.path.join(
                self.save_dir, 
                f"{name.replace(' ', '_').replace('/', '-')}__{time_str}__rank{rank}.pt"
            )
            
            # 2. [!!] 核心: 转储 *当前* 的全部历史 [!!]
            torch.cuda.memory._dump_snapshot(filename)
            
            self.logger.info(f"[MemSnapshot] STOP: '{name}'. "
                             f"快照 (包含 *至今为止* 的所有历史) "
                             f"已保存到: {filename}")
            
            # 3. [!!] 关键逻辑: 如果堆栈 *现在* 为空, 停止并重置历史 [!!]
            if not self._active_names:
                self.logger.info(f"[MemSnapshot] STOP-ROOT: "
                                 f"所有快照已停止。正在重置内存历史。")
                torch.cuda.memory._record_memory_history(enabled=None) # 重置

        except Exception as e:
            self.logger.error(f"[MemSnapshot] FAILED to dump snapshot for '{name}': {e}")

# --- NoOp (空操作) 版本 ---
class NoOpMemorySnapshotter:
    """A 'do nothing' version to match NoOpTimer."""
    def set_logger(self, logger): pass
    def start(self, name: str, max_entries=100000): pass
    def stop(self, name: str): pass

# --- [!!] 创建全局实例 [!!] ---
# 你的代码将 `from my_memory_utils import global_snapshotter`
if os.environ.get("ENABLE_MEMORY_SNAPSHOT", "0") == "1":
    global_snapshotter = MemorySnapshotter(save_dir="memory_snapshots")
else:
    global_snapshotter = NoOpMemorySnapshotter()