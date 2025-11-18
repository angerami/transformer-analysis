"""Performance logger for tracking resource usage across analysis phases."""

import time
import psutil
import json
import logging
import numpy as np
from datetime import datetime
from contextlib import contextmanager


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
            'elapsed_sec': time.perf_counter() - start_time,
            'memory_start_mb': start_mem,
            'memory_end_mb': self.process.memory_info().rss / 1024**2,
        }
        self.phases[name]['memory_delta_mb'] = (
            self.phases[name]['memory_end_mb'] - start_mem
        )
    
    @contextmanager
    def loop_item(self, iteration, log_every=10):
        """Track individual loop iterations, conditionally"""
        if iteration % log_every != 0:
            yield
            return
            
        start_time = time.perf_counter()
        start_mem = self.process.memory_info().rss / 1024**2
        
        yield
        
        self.loop_stats.append({
            'iteration': iteration,
            'elapsed_sec': time.perf_counter() - start_time,
            'memory_mb': self.process.memory_info().rss / 1024**2,
        })
    
    def log_report(self, phase=None, context=''):
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
            'job_id': self.job_id,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Flatten phases
        for phase_name, stats in self.phases.items():
            for stat_name, value in stats.items():
                meta[f'{phase_name}_{stat_name}'] = value
        
        # Loop summary stats
        if self.loop_stats:
            loop_times = [s['elapsed_sec'] for s in self.loop_stats]
            meta['loop_mean_sec'] = np.mean(loop_times)
            meta['loop_max_sec'] = np.max(loop_times)
            meta['loop_n_samples'] = len(self.loop_stats)
        
        return meta


def main():
    """Demonstrate PerfLogger usage with logging integration"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler('demo.log'),
            logging.StreamHandler()
        ]
    )
    
    # Initialize performance logger
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    perf = PerfLogger(job_id)
    
    logging.info(f"Starting job {job_id}")
    
    # Phase 1: Simulate model loading
    with perf.phase('load_model'):
        logging.info("Loading model...")
        # Simulate work
        data = [i for i in range(10000000)]
        time.sleep(1)
    logging.info(perf.log_report(context='pythia-70m'))
    
    # Phase 2: Configuration
    with perf.phase('configure'):
        logging.info("Configuring analysis...")
        time.sleep(0.5)
    logging.info(perf.log_report())
    
    # Phase 3: Loop with conditional logging
    n_iterations = 50
    with perf.phase('loop'):
        for i in range(n_iterations):
            with perf.loop_item(i, log_every=10):
                # Simulate processing
                result = sum(range(100000))
                time.sleep(0.1)
    logging.info(perf.log_report())
    
    # Phase 4: Finalization
    with perf.phase('finalize'):
        logging.info("Aggregating results...")
        time.sleep(0.3)
    logging.info(perf.log_report())
    
    # Phase 5: Write output
    with perf.phase('write_output'):
        logging.info("Writing outputs...")
        metadata = perf.to_metadata()
        with open(f'perf_{job_id}.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        time.sleep(0.2)
    logging.info(perf.log_report())
    
    # Summary
    logging.info("\n" + "="*60)
    logging.info("Performance Summary:")
    for phase_name in perf.phases.keys():
        logging.info(perf.log_report(phase=phase_name))
    logging.info("="*60)
    
    logging.info(f"Metadata saved to perf_{job_id}.json")


if __name__ == '__main__':
    main()
