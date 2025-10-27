import torch
from megatron.core import mpu

def pad_for_sequence_parallel(tensor, padding_value, dim=-1):
    print(f"before padding tensor shape: {tensor.shape}")
    length = tensor.shape[dim]
    seq_parallel_world_size = mpu.get_context_parallel_world_size()
    if length % seq_parallel_world_size == 0:
        return tensor

    pad_num = seq_parallel_world_size - (length % seq_parallel_world_size)
    pad_shape = (
        (*tensor.shape[:dim], pad_num, *tensor.shape[dim + 1 :])
        if dim != -1
        else (*tensor.shape[:dim], pad_num)
    )
    pad = torch.full(pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
    tensor = torch.cat([tensor, pad], dim=dim)
    print(f"after padding tensor shape: {tensor.shape}")
    return tensor

def remove_pad_by_value(tensor, padding_value, dim=-1):
    # Move the target dim to the last, so we can work easily
    print(f"before removing padding tensor shape: {tensor.shape}")
    tensor = tensor.transpose(dim, -1)
    shape = tensor.shape
    flat_tensor = tensor.reshape(-1, shape[-1])  # Collapse other dims

    def get_unpad_length(row):
        mask = row != padding_value
        if mask.any():
            return mask.nonzero()[-1].item() + 1
        else:
            return 0

    unpad_lengths = [get_unpad_length(row) for row in flat_tensor]

    max_len = max(unpad_lengths)
    trimmed = flat_tensor[:, :max_len]

    # Reshape back and transpose to original
    new_shape = shape[:-1] + (max_len,)
    tensor = trimmed.reshape(new_shape).transpose(dim if dim != -1 else len(new_shape) - 1, -1)
    print(f"after removing padding tensor shape: {tensor.shape}")
    return tensor
