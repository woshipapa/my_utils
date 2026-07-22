import torch


def set_oom_flag():
    """在分布式存储中设置 OOM 标志"""
    try:
        if torch.distributed.is_initialized():
            # 获取默认的 TCPStore
            store = torch.distributed.distributed_c10d._get_default_store()
            if store is not None:
                # 设置一个 key，值可以是任意非空字符串
                store.set("GLOBAL_OOM_TRIGGERED", "1")
                print("🚩 [Signal] 已向集群广播 OOM 信号。")
    except Exception as e:
        print(f"⚠️ 广播 OOM 信号失败: {e}")


def check_oom_flag():
    """检查集群中是否有人触发了 OOM"""
    try:
        if torch.distributed.is_initialized():
            store = torch.distributed.distributed_c10d._get_default_store()
            if store is not None:
                # 检查 key 是否存在
                # timeout 设置短一点，避免死等
                try:
                    val = store.get("GLOBAL_OOM_TRIGGERED")
                    if val == b"1":
                        return True
                except:
                    return False
    except:
        pass
    return False
