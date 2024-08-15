# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for license information.

# ---------------------------------------------------------------------------------
# This file contains some parts inspired by the llama-recipes library.
# - Source: https://github.com/meta-llama/llama-recipes

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------

from .callback import *

import gc
import threading
import psutil
import torch


class MemoryCallback(Callback):
    r"""
    Trace the memory usage.
    """
    
    def on_train_epoch_start(self, trainer) -> None:
        """Called when train epoch begins"""
        self.memtrace = MemoryTrace()
    
    def on_train_epoch_end(self, trainer) -> None:
        """Called when train epoch ends"""
        memtrace = self.memtrace
        memtrace.__exit__()
        trainer.logging(f"Max CUDA memory allocated was {memtrace.peak} GB")
        trainer.logging(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
        trainer.logging(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
        trainer.logging(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
        trainer.logging(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
    
    def on_validation_epoch_start(self, trainer) -> None:
        """Called when validation epoch begins"""
        self.memtrace = MemoryTrace()
    
    def on_validation_epoch_end(self, trainer) -> None:
        """Called when validation epoch ends"""
        self.memtrace.__exit__()
    
    def on_test_epoch_start(self, trainer) -> None:
        """Called when test epoch begins"""
        self.memtrace = MemoryTrace()
    
    def on_test_epoch_end(self, trainer) -> None:
        """Called when test epoch ends"""
        self.memtrace.__exit__()


def byte2gb(x):
    return int(x / 2**30)

# This context manager is used to track the peak memory usage of the process
class MemoryTrace:
    def __init__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = byte2gb(torch.cuda.memory_allocated())
        self.process = psutil.Process()
        self.cpu_begin = byte2gb(self.cpu_mem_used())
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)
            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = byte2gb(torch.cuda.memory_allocated())
        self.peak = byte2gb(torch.cuda.max_memory_allocated())
        cuda_info = torch.cuda.memory_stats()
        self.peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
        self.cuda_malloc_retires = cuda_info.get("num_alloc_retries", 0)
        self.peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
        self.m_cuda_ooms = cuda_info.get("num_ooms", 0)
        self.used = byte2gb(self.end - self.begin)
        self.peaked = byte2gb(self.peak - self.begin)
        self.max_reserved = byte2gb(torch.cuda.max_memory_reserved())

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = byte2gb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = byte2gb(self.cpu_peak - self.cpu_begin)