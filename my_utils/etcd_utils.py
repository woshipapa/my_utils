import etcd3
import uuid
import time
import os
import socket

def etcd_barrier(prefix, world_size,
                 etcd_host="127.0.0.1", etcd_port=2379,
                 timeout=300, interval=0.2):
    
    cli = etcd3.client(host=etcd_host, port=etcd_port)

    # 自动为每个进程生成唯一 ID
    member_id = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"

    # 写入自己的 key
    key = f"{prefix}/{member_id}"
    cli.put(key, "1")

    start = time.time()

    while True:
        count = 0
        for _ in cli.get_prefix(prefix):
            count += 1

        if count >= world_size:
            return True

        if time.time() - start > timeout:
            raise TimeoutError(f"barrier timeout: {count}/{world_size}")

        time.sleep(interval)
