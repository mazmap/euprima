import time
from tqdm import tqdm

def benchmark(function, input_generator, ig_args, display_progress_bar=True):
    summary = []
    if display_progress_bar:
        print("Start benchmark for", function.__name__)

    for arg in tqdm(ig_args):
        inputs = input_generator(arg)

        t_start = time.perf_counter()
        for input in inputs:
            if isinstance(input, tuple): 
                function(*input)
            else: 
                 function(input)
        t_end = time.perf_counter()
        t_avg = (t_end - t_start) / len(inputs)
        summary.append(t_avg)

    return summary
