import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


def show_data(data, title=None):
    """ data dict to pandas df w/ plot """
    try:
        df = pd.DataFrame.from_dict(data)
    except ValueError:
        df = pd.DataFrame()
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
        "scope": [],
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
            
        if "loss:" in line:
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
    dev_scores = {
        "absa": [],
        "easy": [],
        "hard": [],
        "binary": [],
        "proportional": [],
        "span": [],
        "macro": [],
    }
    test_scores = {
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
            score = float(line.split('overall: ')[-1].split(' (')[0].strip())
            dev_scores[metric.lower()].append(score)
        
        if "Score:" in line:
            print(line.split("INFO] ")[1])

        if "FINAL" in line:
            metric = line.split("FINAL")[1].split(":").strip()
            if metric in test_scores:
                score = line.split(":")[-1]
                test_scores[metric] = score # TODO Finish implementing
            
    for metric in list(dev_scores):
        if dev_scores[metric] == []:
            dev_scores.pop(metric)
        
    return dev_scores

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
    
    header = title
    for i, run in enumerate(runs):
        loss = parse_loss(run)
        metrics = parse_metrics(run)
        
        if title is None:
            header = name + f" Run:{i}"
        try:
            loss_df = show_data(loss, "Loss:{}".format(header))
            metrics_df = show_data(metrics, "Metrics:{}".format(header))
            plt.show()
            
            loss_dfs[i] = loss_df.T
            metrics_dfs[i] = metrics_df.T
            
            for row in run:
                if "Current" in row:
                    print("Above plots w/ following params:\n {}".format(row.split("Current params:")[-1]))
                    print("="*75)
        except TypeError:
            print("No data for run {}".format(i))
    
    return loss_dfs, metrics_dfs

def show_same_smoothed(same, stop_at=1, header="smoothed", show_df=True):
    """
    Plots as many runs as in first study in same. 

    Parameters:
        same: list of studies w/ exact same parameter configurations
    """

    for plot, title in enumerate(["Loss", "Metrics"]):
        for run in same[0][plot]:
            average = sum([study[plot][run] for study in same])/len(same)
            average.plot(title=title + ": " + header, figsize=(15,5))

            if run >= stop_at:
                break
    return average.T

def smooth(study, runs, header=""):
    loss, metric = study

    avg_loss = sum([loss[i] for i in runs])/len(runs)
    avg_metric = sum([metric[i] for i in runs])/len(runs)

    avg_loss.plot(title="Loss "+header, figsize=(15,5))
    avg_metric.plot(title="Metric "+header, figsize=(15,5))

    print("Final values:")
    display(avg_loss.tail(3))
    display(avg_metric.tail(3))

    return avg_loss, avg_metric


### Stats

def get_final_scores(study, runs = None, metric="absa", header=""):
    iterator = runs if runs is not None else study[1]
    
    finals =  [
        study[1][run][metric].tail(1).item()
        for run in iterator
    ]
    
    #x-axis ranges from -3 and 3 with .001 steps
    x = np.arange(min(finals), max(finals), 0.001)
    interval = (max(finals) - min(finals)) / np.mean(finals)
    dist =  norm.pdf(x, np.mean(finals), np.std(finals)) * interval
    
    #plot normal distribution with mean 0 and standard deviation 1
    plt.plot(x, norm.pdf(x, dist))
    
    plt.hist(finals, bins=6)
    plt.title("Dist. of final scores "+header)
 
    plt.show()
    
    return finals



### DEPRECATED BELOW ------------------------------------------------------
### Batch-wise loss 
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



