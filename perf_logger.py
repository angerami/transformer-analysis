"""Performance logger for tracking resource usage across analysis phases."""

import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import psutil


class PerfLogger:
    def __init__(self, job_id):
        self.job_id = job_id
        self.process = psutil.Process()
        self.phases = {}
        self.loop_stats = []

    @contextmanager
    def phase(self, name):
        """Track major phases: load, configure, loop, finalize, write"""
        start_time = time.perf_counter()
        start_mem = self.process.memory_info().rss / 1024**2

        yield

        self.phases[name] = {
            "elapsed_sec": time.perf_counter() - start_time,
            "memory_start_mb": start_mem,
            "memory_end_mb": self.process.memory_info().rss / 1024**2,
        }
        self.phases[name]["memory_delta_mb"] = (
            self.phases[name]["memory_end_mb"] - start_mem
        )

    @contextmanager
    def loop_item(self, iteration, log_every=10):
        """Track individual loop iterations, conditionally"""
        if iteration % log_every != 0:
            yield
            return

        start_time = time.perf_counter()

        yield

        self.loop_stats.append(
            {
                "iteration": iteration,
                "elapsed_sec": time.perf_counter() - start_time,
                "memory_mb": self.process.memory_info().rss / 1024**2,
            }
        )

    def log_report(self, phase=None, context=""):
        """Generate formatted report for specific phase or most recent"""
        if not self.phases:
            return "No phases completed yet"

        phase_name = phase or list(self.phases.keys())[-1]
        if phase_name not in self.phases:
            return f"Phase '{phase_name}' not found"

        stats = self.phases[phase_name]
        prefix = f"[{context}] " if context else ""

        return (
            f"{prefix}{phase_name}: "
            f"{stats['elapsed_sec']:.2f}s, "
            f"Δmem: {stats['memory_delta_mb']:+.1f}MB, "
            f"mem: {stats['memory_end_mb']:.1f}MB"
        )

    def to_metadata(self):
        """Export as dict for dataframe row"""
        meta = {
            "job_id": self.job_id,
            "timestamp": datetime.now().isoformat(),
        }

        # Flatten phases
        for phase_name, stats in self.phases.items():
            for stat_name, value in stats.items():
                meta[f"{phase_name}_{stat_name}"] = value

        # Loop summary stats
        if self.loop_stats:
            loop_times = [s["elapsed_sec"] for s in self.loop_stats]
            meta["loop_mean_sec"] = np.mean(loop_times)
            meta["loop_max_sec"] = np.max(loop_times)
            meta["loop_n_samples"] = len(self.loop_stats)

        return meta
