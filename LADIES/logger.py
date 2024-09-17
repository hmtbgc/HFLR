import datetime
import time
import os

def new_log(log_path, args):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    filename = os.path.join(log_path, timestamp)
    for arg in vars(args):
        arg_text = f'-{arg}[{getattr(args, arg)}]'
        filename += arg_text
    f = open(filename + ".txt", "w")
    return f

def PRINT_LOG(text, file):
    print(text)
    print(text, file=file)