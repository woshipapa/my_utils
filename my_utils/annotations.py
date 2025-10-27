from functools import wraps

def parametrize_shapes(shape_list, batch_size_range):
    """
    对每个shape配置，自动遍历batch_size_range，调用func(bs, num_frames, H, W, desc, **kwargs)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            for shape in shape_list:
                for bs in batch_size_range:
                    # 取 shape 的各参数
                    if len(shape) == 3:
                        num_frames, H, W = shape
                    else:
                        raise ValueError("shape 参数格式应为 (num_frames, H, W")
                    print(f"\n==== 测试参数: bs={bs}, num_frames={num_frames}, H={H}, W={W}, desc= ====")
                    result = func(
                        bs=bs, num_frames=num_frames, H=H, W=W,  **kwargs
                    )
                    results.append((bs, num_frames, H, W, result))
            return results
        return wrapper
    return decorator
