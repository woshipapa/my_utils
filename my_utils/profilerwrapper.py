import torch
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
import torch.distributed as dist
import os
from datetime import datetime
import numpy as np

import traceback, sys
class ProfilerWrapper:
    def __init__(self,is_st: bool, activities=None, profile_memory=False, record_shapes=True, 
                 log_dir='/workspace/code/pf_logs', 
                 mem_dir='/workspace/code/memory_logs',
                 thresold: int = 4,
                 enabled_record_memory_pickle: bool = False,
                 enabled_record_cuda_average_time: bool = False,
                 enable_record_cuda_mm: bool = False,
                 enable_print_summary: bool = False
                 ):
        # pytorch profiler
        self.activities = activities or [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        self.profile_memory = profile_memory
        self.record_shapes = record_shapes
        self.profiler = None
        self.log_dir = log_dir


        # distributed
        self.rank = dist.get_rank()
        # self.device = f'cuda:{self.rank}'
        self.device = torch.cuda.current_device()

        # file_name_prefix
        self.is_st = is_st
        self.prefix = 'cogvideo' if is_st is False else 'opensora'



        # if not os.path.exists(log_dir):
        #     os.mkdir(log_dir)
        # if not os.path.exists(mem_dir):
        #     os.mkdir(mem_dir)
        self.output_dir = mem_dir        
        self.counter = 0
        self.total_self_cuda_time = 0.0
        self.thresold = 4

        # print cuda_time_total sort table
        self.enable_print_summary = enable_print_summary


        # draw for cuda_time_total
        self.enable_record_cuda_average_time = enabled_record_cuda_average_time
        self.cuda_times = []
        self.run_indices = []
        self.fa_times = []


        # draw for cuda memory
        self.enable_record_cuda_mm = enable_record_cuda_mm
        self.allocated_mm = []
        self.reserved_mm = []
        
        # memory record pickle using pytorch snapshot
        self.enabled_record_memory_pickle = enabled_record_memory_pickle
        if enabled_record_memory_pickle:
            torch.cuda.memory._record_memory_history(max_entries=100000)   

    # def trace_handler(self, prof: torch.profiler.profile):
    #     # 获取时间用于文件命名
    #     timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    #     file_name = f"{self.output_dir}/visual_mem_{timestamp}"

    #     # 导出tracing格式的profiling
    #     prof.export_chrome_trace(f"{file_name}.json")

    #     # 导出mem消耗可视化数据
    #     prof.export_memory_timeline(f"{file_name}.html")

    def __enter__(self):
        """启动 Profiler"""
        self.profiler = profile(
            activities=self.activities,
            profile_memory=self.profile_memory,
            record_shapes=self.record_shapes,
            with_stack=False,
            # on_trace_ready=self.trace_handler
        )

        self.profiler.__enter__()
        # return self.profiler
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """退出 Profiler并打印结果"""
        self.profiler.__exit__(exc_type, exc_value, traceback)
        # self.profiler.export_chrome_trace("opensora_trace.json")  # 导出为 trace.json
        # print("Profiler finished, trace exported to trace.json")
        self.counter += 1
        self.run_indices.append(self.counter)
        self.record()
    
    def record(self):
        # print profiler table 
        if self.enable_print_summary:
            self.print_summary()
        if self.enable_record_cuda_average_time:
            self.record_cuda_average_time()
        if self.enable_record_cuda_mm:    
            self.record_cuda_mm()
        self.counter += 1

    def record_cuda_mm(self):
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        # memory_usage_per_gpu = []

        self.allocated_mm.append(allocated)
        self.reserved_mm.append(reserved)

        print(f"counter : {self.counter} : allocated_mm = {allocated}GB, reserved_mm = {reserved}GB")
    

    def record_function(self, name, func):
        """记录函数的性能"""


        try:
            with record_function(name):
                return func()
        except torch.cuda.OutOfMemoryError as e:
            print(f'[rank {self.rank}] out of memory!')
            self.leave()
            traceback.print_exc()
            sys.exit(1)    

    def record_cuda_average_time(self):
            # 获取总时间（无需计算求和）
            spatial_total_time = 0
            temporal_total_time = 0
            transformer_total_time = 0
            fa_time = 0
            for item in self.profiler.key_averages():
                if item.key == 'spatial block':
                    spatial_total_time = item.cuda_time_total
                if item.key == 'temporal block':
                    temporal_total_time =  item.cuda_time_total
                if item.key == 'cogvideo block':
                    transformer_total_time = item.cuda_time_total
                if item.key == 'aten::_flash_attention_forward':
                    fa_time = item.cuda_time_total

            total_time = (spatial_total_time + temporal_total_time + transformer_total_time) / 1000     


            self.fa_times.append(fa_time/1000)

            self.total_self_cuda_time += total_time
            
            self.cuda_times.append(total_time)
            # self.run_indices.append(self.counter)

    def leave(self):
        
        self.export_trace()
        if self.enable_record_cuda_average_time:
            self.plot_cuda_times()
        if self.enable_record_cuda_mm:    
            self.plot_memory_line()
        if self.enabled_record_memory_pickle:
            self.close_memory_record()
    


    def plot_memory_line(self):
        # 创建柱状图
        fig, ax = plt.subplots(figsize=(10, 6))

        # print(f"run_indices len is {len(self.run_indices)}, allocated_mm is {self.allocated_mm}")

        # 绘制折线图
        ax.plot(self.run_indices, self.allocated_mm, label='Allocated Memory', marker='o', color='b')
        ax.plot(self.run_indices, self.reserved_mm, label='Reserved Memory', marker='x', color='r')

        max_allocated = max(self.allocated_mm)
        max_reserved = max(self.reserved_mm)
        # max_allocated_idx = self.allocated_mm.index(max_allocated)
        # max_reserved_idx = self.reserved_mm.index(max_reserved)
        # # 在每个点上显示数值
        # for i, (alloc, res) in enumerate(zip(self.allocated_mm, self.reserved_mm)):
        #     # print(f"i : {i}, alloc,reserved = {alloc,res}")
        #     plt.text(i, alloc, f'{alloc:.2f}G', fontsize=8, color='b', ha='center', va='bottom', alpha=0.7)
        #     plt.text(i, res, f'{res:.2f}G', fontsize=8, color='r', ha='center', va='bottom', alpha=0.7)

        alloc_diff = np.diff(self.allocated_mm)
        average_slope = np.mean(alloc_diff)


        # if os.path.exists(f"{self.prefix}_memory_usage.txt"):
        #     os.remove(f"{self.prefix}_memory_usage.txt")
        with open(f"{self.prefix}_memory_usage.txt",'a') as f:
            f.write(f'Allocated Memory (GB)  (device = {self.device})  average_slope = {average_slope}\n')
            for alloc in self.allocated_mm:
                f.write(f'{alloc}GB\n')  # 每行写入一个值，保留两位小数
            
            f.write(f'\nReserved Memory (GB) (device = {self.device})\n')
            for res in self.reserved_mm:
                f.write(f'{res}GB\n')  # 每行写入一个值，保留两位小数


        # 设置图表标题和标签
        ax.set_title(f'Memory Usage at Each Layer ({self.device})')
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Memory (GB)')

        # 添加图例
        ax.legend()

        # 显示图形
        plt.tight_layout()
        plt.savefig(f'{self.prefix}_memory_usage_line_chart.png', dpi=300)

    def close_memory_record(self):
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        file_name = f"{self.prefix}_mem_{timestamp}.pickle"

        torch.cuda.memory._dump_snapshot(file_name)   
        torch.cuda.memory._record_memory_history(enabled=None)           


    def export_trace(self):
        name = f'{self.prefix}_trace.json'
        # if os.path.exists(name):
        #     os.remove(name)
        if self.rank == 0:
            self.profiler.export_chrome_trace(name)

    def plot_cuda_times(self):
        """
        绘制 CUDA 时间的折线图。
        """



        plt.figure(figsize=(10, 6))
        plt.plot(self.run_indices, self.cuda_times, marker='o', label="Self CUDA Time (ms)")
        average_time = sum(self.cuda_times) / len(self.run_indices)
        fa_average_time = sum(self.fa_times) / len(self.run_indices)
        ratio = fa_average_time / average_time
        plt.subplots_adjust(bottom=0.3)
        plt.figtext(0.1, 0.1, f"Average Time: {average_time:.4f} ms", ha="left", fontsize=10)
        plt.figtext(0.1, 0.07, f"FA Average Time: {fa_average_time:.4f} ms", ha="left", fontsize=10)
        plt.figtext(0.1, 0.04, f"FA / Average Time Ratio: {ratio:.4f}", ha="left", fontsize=10)

        # self.plot_bar_chart(average_time,fa_average_time)

        plt.xlabel("Run Index")
        plt.ylabel("Self CUDA Time (ms)")
        plt.title("Self CUDA Time Over Runs")
        plt.legend()
        plt.grid(True)

        plt.tight_layout(rect=[0, 0.2, 1, 1])  # 再次调整以确保完整显示
        plt.savefig(f"{self.prefix}_cuda_time_plot.png")  # 保存图表
        # plt.show()  # 显示图表

    def plot_bar_chart(self, average_time_ms, fa_average_time_ms):
        """
        绘制柱状图显示平均时间、FA 时间及其差值。
        
        Args:
            average_time_ms (float): 平均时间（单位为毫秒）。
            fa_average_time_ms (float): FA 时间（单位为毫秒）。
        """
        # 计算差值
        difference_time = average_time_ms - fa_average_time_ms

        # 准备数据
        labels = ['Average Time', 'FA Average Time', 'Difference (Avg - FA)']
        values = [average_time_ms, fa_average_time_ms, difference_time]
        colors = ['blue', 'orange', 'green']

        # 创建柱状图
        plt.figure(figsize=(8, 6))
        plt.bar(labels, values, color=colors, alpha=0.7)
        
        # 添加数据标签
        for i, value in enumerate(values):
            plt.text(i, value + 0.5, f'{value:.2f} ms', ha='center', va='bottom', fontsize=10)
        
        # 设置图表信息
        plt.title("Comparison of Times", fontsize=14)
        plt.ylabel("Time (ms)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 保存并显示图表
        plt.savefig("cuda_time_comparison_bar_chart.png")    


    def print_summary(self):
        """直接打印性能分析的总结，并绘制图表"""
        # 获取每个操作的性能数据
        key_averages = self.profiler.key_averages()

        # 打印每个操作的性能数据
        total_cpu_time = 0.0
        total_cuda_time = 0.0
        total_cpu_memory_usage = 0.0  # 初始化为 0.0
        total_cuda_memory_usage = 0.0  # 初始化为 0.0        

        
        

        print(self.profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))  # 输出表格，限制行数为 10
        print(f"counter : {self.counter} ")

        
        # 绘制 CPU 和 CUDA 时间占比图
        # if self.rank == 0:
        #     self.save_profiler_table(self.rank)

        # if dist.get_rank() == 0:
        #     self.plot_top5_cuda_time(save_path=self.log_dir)
        # self.plot_time_distribution(operations, cpu_times, cuda_times)


    def save_profiler_table(self, rank):
        """根据 rank 保存 profiler 输出的表格到文件"""
        # 获取 key averages
        key_averages = self.profiler.key_averages()

        
        

        log_file = f"{self.log_dir}/{self.prefix}_rank_{rank}.log"
        # log_file = f"{self.log_dir}/profiler_rank_{rank}_cogvideo.log"
        # 打开文件进行写入
        with open(log_file, "a") as f:
            # 将 Profiler 输出表格写入文件
            f.write(key_averages.table(sort_by="self_cuda_time_total"))  # 可以根据需要调整排序方式
            f.write("\n" + "-"*40 + "\n")  # 添加分隔线，方便查看




    def plot_top5_cuda_time(self, save_path=None):
        # 获取 profiler 的 key_averages
        key_averages = self.profiler.key_averages()

        # 根据 self_cuda_time_total 排序
        sorted_averages = sorted(key_averages, key=lambda avg: avg.self_cuda_time_total, reverse=True)

        # 获取前 5 个 key
        top5 = sorted_averages[:5]
        # 获取剩余的合并为 "Other"
        other = sorted_averages[5:]

        # 提取名称和时间
        labels = [avg.key for avg in top5] + ['Other']
        times = [avg.self_cuda_time_total for avg in top5] + [sum(avg.self_cuda_time_total for avg in other)]

        # 绘制饼状图
        plt.figure(figsize=(12, 12))
        plt.pie(times, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('Top 5 CUDA Time (Other aggregated)', fontsize=14)
        plt.axis('equal')  # 确保图形为圆形

        # 保存为 PNG 图片
        plt.savefig(save_path, format='png', bbox_inches='tight')
        plt.close()  # 关闭图形以释放资源