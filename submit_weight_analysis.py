from run_weight_analysis import model_shorts, main
import time
from datetime import datetime

t_0 = time.perf_counter()
abs_start = datetime.now()
print('Start time: ' + abs_start.strftime("%Y-%m-%d %H:%M:%S"))

times = []

for model_name in model_shorts.keys():
    start = time.perf_counter()
    print("="*20 + f'Processing {model_name}' + "="*20)
    main(model_name=model_name)
    print("-"*60 + "\n"*2)
    times.append((model_name,time.perf_counter() - start ))

print('Time summary\n' + '-'*20)
for k, v in times:
    print(f"{k} : {v:.2f}")
t_f = time.perf_counter()
abs_end = datetime.now()
print('\nEnd time: ' + abs_end.strftime("%Y-%m-%d %H:%M:%S"))
print(f"Total Elapsed : {t_f - t_0:.2f}")

