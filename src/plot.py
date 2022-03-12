import pandas as pd
import matplotlib.pyplot as plt


def show_data(data, title=None):
    """ data dict to pandas df w/ plot """
    df = pd.DataFrame.from_dict(data)
    df.plot(title=title, figsize=(15,5))
    return df.T

### Task-wise loss
def parse_taskwise_loss(data, get_epochs=True):
    """Assume data is f.readlines() of a log file."""
    task_loss = {
        "expression": [],
        "holder": [],
        "polarity": [],
        "target": [],
        "total": [],
    }
    
    epochs = [0]
    prev_batches = -1
    
    for line in data:
        # goes first to get correct len of task_loss item for current line
        if get_epochs:
            if "Epoch:" in line and "Batch:" in line:
                batch = int(line.split("Batch:")[-1].split('(')[0].strip())
                if prev_batches < batch:
                    prev_batches = batch
                else:
                    longest = max([len(task_loss[i]) for i in task_loss])
                    epochs.append(longest)
                    prev_batches = -1
            
        if "loss" in line:
            task = line.split("INFO] ")[-1].split('loss:')[0].strip().lower()
            loss = float(line.split('loss:')[-1].split(' ')[0].strip())
            task_loss[task].append(loss)
            
    for task in list(task_loss):
        if task_loss[task] == []:
            task_loss.pop(task)
        
    return task_loss, epochs

def show_taskwise_loss(name, get_epochs=True, title=None):
    def show_epochs(epochs):
        for x in epochs:
            plt.axvline(x, c="k", ls="-.")
        

    with open(name) as f:
        data = f.readlines()

    loss, epochs = parse_taskwise_loss(data, get_epochs)

    df = pd.DataFrame.from_dict(loss)
    plt = df.plot(title=title if title else name, figsize=(15,5))
    show_epochs(epochs) if get_epochs else None
    
    return df.T  # show table horizontally


### Loss (no-epochs)
def parse_loss(data):
    """ General instance of taskwise loss (no epoch lines) """
    loss, _ = parse_taskwise_loss(data, get_epochs=False)
    return loss

def show_loss(name):
    return show_taskwise_loss(name, get_epochs=False)


### Metrics
def parse_metrics(data):
    """Assume data is f.readlines() of a log file."""
    scores = {
        "absa": [],
        "easy": [],
        "hard": [],
        "binary": [],
        "proportional": [],
        "span": [],
        "macro": [],
    }
    
    for line in data:
        if " overall: " in line:
            metric = line.split("INFO] ")[-1].split("overall:")[0].strip()
            if "(RACL)" in metric:
                metric = metric.split("(RACL)")[-1].strip()
            loss = float(line.split('overall: ')[-1].split(' (')[0].strip())
            scores[metric.lower()].append(loss)
            
    for metric in list(scores):
        if scores[metric] == []:
            scores.pop(metric)
        
    return scores

def show_metrics(name, title=None):

    with open(name) as f:
        data = f.readlines()

    scores = parse_metrics(data)
    df = show_data(scores, title=title if title else name)
    
    return df  # show table horizontally


### Study
def parse_large_logs(data):
    runs = []
    current_run = ["Parsing {} lines".format(len(data))]
    for line in data:
        if "new run" in line:
            runs.append(current_run)
            current_run = [line]
        else:
            current_run.append(line)
    runs.append(current_run)
    return runs

def get_runs(name):
    
    with open(name) as f:
        data = f.readlines()
        
    return parse_large_logs(data)

def show_study_loss(name, title=None):
    runs = get_runs(name)[1:] # skip first "run"
    loss_dfs = {}
    metrics_dfs = {}
    
    for i, run in enumerate(runs):
        loss = parse_loss(run)
        metrics = parse_metrics(run)
        
        try:
            loss_df = show_data(loss, "Loss:{}".format(title if title is not None else name))
            metrics_df = show_data(metrics, "Metrics:{}".format(title if title is not None else name))
            plt.show()
            
            loss_dfs[i] = loss_df.T
            metrics_dfs[i] = metrics_df.T
            
            for row in run:
                if "Current" in row:
                    print("Above plots w/ following params:\n {}".format(row.split("Current params:")[-1]))
        
        except TypeError:
            print("No data for run {}".format(i))
    
    return loss_dfs, metrics_dfs


### DEPRECATED ### Batch-wise loss 
def parse_batchwise_loss(data):
    """This logging format has been changed. Needed only for older logs"""
    epochs = {}
    for line in data:
        if "Epoch:" in line:
            tail = line.split('Epoch:')[-1]
            tail = tail.split('Batch:')
            epoch = int(tail[0].strip())
            tail = tail[-1].split('Loss:')
            batch = int(tail[0].strip())
            loss = float(tail[-1].split(' ')[0].strip())
            if epoch in epochs:
                epochs[epoch][batch] = loss
            else:
                epochs[epoch] = {batch: loss}
    return epochs

def show_batchwise_loss(name):
    """This logging format has been changed. Needed only for older logs"""

    with open(name) as f:
        data = f.readlines()

    loss = parse_batchwise_loss(data)
    
    df = pd.DataFrame.from_dict(loss, orient="index")
    df.plot(title=name)
    
    return df



