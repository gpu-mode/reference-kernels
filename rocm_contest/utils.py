import torch
import os
import copy
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def getLogger():
    global logger
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        if int(os.environ.get('LOG_MORE', 0)):
            formatter = logging.Formatter(
                fmt="[%(name)s %(levelname)s] %(asctime)s.%(msecs)03d - %(process)d:%(processName)s - %(pathname)s:%(lineno)d - %(funcName)s\n%(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger


logger = getLogger()

def perftest(num_iters=101, num_warmup=5, testGraph=False, num_rotate_args=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            run_iters(num_warmup, func, *args, **kwargs)
            rotate_args = [(copy.deepcopy(args),
                            copy.deepcopy(kwargs))
                           for _ in range(num_rotate_args)]

            if int(os.environ.get('LOG_MORE', 0)):
                latencies = []
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                for _ in range(num_iters):
                    start_event.record()
                    data = func(*args, **kwargs)
                    end_event.record()
                    end_event.synchronize()
                    latencies.append(start_event.elapsed_time(end_event))
                avg = np.mean(latencies) * 1000
                logger.info(f'avg: {avg} us/iter from cuda.Event')

            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                             profile_memory=True,
                             with_stack=True,
                             with_modules=True,
                             ) as prof:
                data = run_iters_rotate(num_iters, func, rotate_args)
            avg = get_trace_perf(prof, num_iters)
            return data, avg
        return wrapper
    return decorator

def run_iters(num_iters, func, *args, **kwargs):
    data = None
    for _ in range(num_iters):
        data = func(*args, **kwargs)
    return data


def run_iters_rotate(num_iters, func, rotate_args):
    data = None
    num_rotate_args = len(rotate_args)
    for _ in range(num_iters):
        args, kwargs = rotate_args[_ % num_rotate_args]
        data = func(*args, **kwargs)
    return data


def get_trace_perf(prof, num_iters):
    assert (num_iters > 1)
    num_iters -= 1
    df = []
    cols = ['name', 'self_cpu_time_total', 'self_device_time_total',
            'device_type', 'device_index',]
    for el in prof.events():
        df.append([getattr(el, x, None) for x in cols])
    df = pd.DataFrame(df, columns=cols)
    df['cnt'] = 1
    rets = []
    for name, d in df.groupby('name', sort=False):
        r = d.iloc[1:][['cnt',
                        'self_cpu_time_total',
                        'self_device_time_total']].sum()
        if not r.empty:
            device_type = str(d['device_type'].iat[0]).split('.')[-1]
            r['name'] = name
            r['device_type'] = device_type
            r['device_index'] = str(d['device_index'].iat[0])
            if device_type == 'CUDA':
                r['device_time_total'] = r['self_device_time_total']
                r['host_time_total'] = 0
            else:
                r['host_time_total'] = r['self_device_time_total']
                r['device_time_total'] = 0

        rets.append(r)
    df = pd.DataFrame(rets)

    cols = ['name', 'cnt', 'host_time_total', 'device_time_total',
            'device_type', 'device_index',]
    cols = [el for el in cols if el in df.columns]
    df = df[(df.host_time_total > 0) | (df.device_time_total > 0)]

    timerList = ['host_time_total', 'device_time_total', ]
    df = df[cols].sort_values(timerList, ignore_index=True)
    avg_name = '[avg us/iter]'
    for el in timerList:
        df.at[avg_name, el] = df[el].sum()/num_iters
    if int(os.environ.get('LOG_MORE', 0)):
        pd.set_option('display.max_colwidth', 120)
        logger.info(f'{df}')
    return df.at[avg_name, 'device_time_total']


def checkAllclose(a, b, rtol=1e-2, atol=1e-2, msg='', printNum=8):
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)
    mask = ~isClose
    if isClose.all():
        logger.info(f'{msg}[checkAllclose {atol=} {rtol=} passed~]')
        return True
    else:
        num = mask.sum()
        printNum = min(printNum, num)
        percent = num/a.numel()
        delta = (a-b)[mask]
        if percent > 0.01:
            logger.info(f'''{msg}[checkAllclose {atol=} {rtol=} failed!]
    a    : {a.shape}
           {a[mask][:printNum]}
    b    : {b.shape}
           {b[mask][:printNum]}
    delta:
           {delta[:printNum]}''')
        else:
            logger.info(
                f'''{msg}[checkAllclose {atol=} {rtol=} waring!] a and b results are not all close''')
        logger.info(
            f'-->max delta:{delta.max()}, delta details: {percent:.1%} ({num} of {a.numel()}) elements')
        return False