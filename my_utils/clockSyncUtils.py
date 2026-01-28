import time
import torch
import torch.distributed as dist

class ClockSynchronizer:
    """
    基于 NTP (Network Time Protocol) 逻辑的分布式时钟同步工具。
    用于计算 Producer (4090) 和 Consumer (H100) 之间的系统时钟偏差。
    """
    
    @staticmethod
    def sync(is_server: bool, peer_rank: int, group=None):
        """
        执行同步。
        
        Args:
            is_server (bool): 
                True  = 基准方 (通常是 Producer/4090)，Offset 为 0。
                False = 跟随方 (通常是 Consumer/H100)，计算相对于 Server 的 Offset。
            peer_rank (int): 对方在 group 中的 rank。
            group: 分布式通信组，默认为 None (全局组)。
            
        Returns:
            float: time_offset (Server_Time - Client_Time)。
                   Consumer 拿到这个值后，加到自己的 timer 上，就变成了 Producer 的时间。
        """
        # 1. 【强制 Barrier】确保双方都已到达起跑线
        # 这消除了 "Consumer 先发包，但 Producer 还在忙没准备好" 导致的虚假延迟
        dist.barrier(group=group)
        
        # 准备数据容器 (必须是 float64/Double 以保留微秒精度)
        # NCCL 后端通常要求 Tensor 在 CUDA 上
        tensor_buffer = torch.zeros(2, dtype=torch.float64, device=torch.cuda.current_device())
        
        if is_server:
            # === Server Logic (Producer) ===
            
            # A. 等待 Client 的 Ping
            # 此时 Producer 已经在 Barrier 处等好了，收到数据包的瞬间就是 T2
            dist.recv(tensor_buffer, src=peer_rank, group=group)
            t2 = time.time() # T2: Server Receive Timestamp
            
            # B. 记录回复时间 T3
            t3 = time.time() # T3: Server Send Timestamp
            
            # C. 将 T2, T3 打包发回
            tensor_buffer[0] = t2
            tensor_buffer[1] = t3
            dist.send(tensor_buffer, dst=peer_rank, group=group)
            
            print(f"[ClockSync] Server ready. Reference time provided.")
            return 0.0

        else:
            # === Client Logic (Consumer) ===
            
            # A. 发送 Ping
            t1 = time.time() # T1: Client Send Timestamp
            dist.send(tensor_buffer, dst=peer_rank, group=group) # 发个空包触发 T2
            
            # B. 接收 Pong (包含 T2, T3)
            dist.recv(tensor_buffer, src=peer_rank, group=group)
            t4 = time.time() # T4: Client Receive Timestamp
            
            t2 = tensor_buffer[0].item()
            t3 = tensor_buffer[1].item()
            
            # === NTP 核心计算 ===
            # RTT (往返时延) = 总流逝时间 - Server处理时间
            rtt = (t4 - t1) - (t3 - t2)
            latency = rtt / 2.0
            
            # Offset = T_server - T_client
            # T_server_at_t1 = t2 - latency
            offset = (t2 - latency) - t1
            
            print("="*40)
            print(f"[ClockSync] Client Synchronization Result:")
            print(f"  > RTT     : {rtt*1000:.3f} ms")
            print(f"  > Latency : {latency*1000:.3f} ms (One-way)")
            print(f"  > Offset  : {offset:.6f} s")
            print("="*40)
            
            return offset
        


import time
import socket
import struct
import pickle

class SocketClockSynchronizer:
    def __init__(self, port=12345):
        self.port = port
        self.sock = None
        self.conn = None

    def sync(self, is_server: bool, peer_ip: str = '0.0.0.0'):
        """
        不依赖 torch.distributed，使用原生 TCP Socket 进行同步。
        """
        offset = 0.0
        
        try:
            # ==========================================
            # 1. 建立 TCP 连接 (相当于握手)
            # ==========================================
            if is_server:
                # Producer 作为 Server
                server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_sock.bind(('0.0.0.0', self.port))
                server_sock.listen(1)
                print(f"[SocketSync] Listening on port {self.port}...")
                
                self.conn, addr = server_sock.accept()
                print(f"[SocketSync] Connected by {addr}")
                
            else:
                # Consumer 作为 Client
                # 稍微 sleep 一下确保 Server 起来了
                time.sleep(2) 
                print(f"[SocketSync] Connecting to {peer_ip}:{self.port}...")
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # 尝试连接，带重试
                for i in range(10):
                    try:
                        self.sock.connect((peer_ip, self.port))
                        break
                    except ConnectionRefusedError:
                        print(f"Retrying connection... ({i+1}/10)")
                        time.sleep(1)
                self.conn = self.sock

            # ==========================================
            # 2. 简易 Barrier (确保双方都 Ready)
            # ==========================================
            # 逻辑：双方互发一个字节，都收到才继续
            self.conn.sendall(b'1') # 我到了
            self.conn.recv(1)       # 等你也到了
            
            # 此时双方都已冲过 Barrier，CPU 相对空闲
            
            # ==========================================
            # 3. NTP 对表
            # ==========================================
            if is_server:
                # --- Server Logic ---
                # A. 等待 Ping
                data = self.conn.recv(1024) # 阻塞等待
                t2 = time.time()            # T2: Recv Time
                
                # B. 发送 Pong (T2, T3)
                t3 = time.time()            # T3: Send Time
                payload = struct.pack('!dd', t2, t3) # double (8 bytes) * 2
                self.conn.sendall(payload)
                
                return 0.0
                
            else:
                # --- Client Logic ---
                # A. 发送 Ping
                t1 = time.time()            # T1: Send Time
                self.conn.sendall(b'PING')
                
                # B. 接收 Pong
                data = self.conn.recv(1024)
                t4 = time.time()            # T4: Recv Time
                
                if len(data) == 16:
                    t2, t3 = struct.unpack('!dd', data)
                    
                    rtt = (t4 - t1) - (t3 - t2)
                    latency = rtt / 2.0
                    offset = (t2 - latency) - t1
                    
                    print(f"[SocketSync] RTT: {rtt*1000:.3f}ms, Offset: {offset:.6f}s")
                    return offset
                else:
                    print("Error: Invalid packet size")
                    return 0.0
                    
        finally:
            # 清理连接
            if self.conn: self.conn.close()
            if is_server and 'server_sock' in locals(): server_sock.close()