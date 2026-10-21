"""Fake a training run so the progress bar has something to show: python demo.py [name] [seconds]"""
import math, random, sys, time
from train_progress import Progress

name = sys.argv[1] if len(sys.argv) > 1 else "demo-run"
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 120
steps = 500
with Progress(name, total=steps) as p:
    for step in range(1, steps + 1):
        time.sleep(seconds / steps)
        p.update(step, loss=2.5 * math.exp(-step / 150) + random.uniform(0, 0.05))
