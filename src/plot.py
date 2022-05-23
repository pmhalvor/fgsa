import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


def show_data(data, title=None, epoch_cap=None, show=True):
    """ data dict to pandas df w/ plot """
    try:
        df = pd.DataFrame.from_dict(data)
    except ValueError:
        df = pd.DataFrame()

    if epoch_cap is not None:
        df = df[:epoch_cap]

    df.plot(title=title, figsize=(15,5)) if show else None
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

        if len(epochs) > 50:
            plt.set_xticks([x if i%10==0 else 10 for (i, x) in enumerate(epochs[1:])], [x if i%10==0 else None for (i,x) in enumerate(range(len(epochs[1:])))])
            plt.set_xlabel("Epochs")
            plt.set_ylabel("Loss")

            for i, x in enumerate(epochs[1:]):
                if i%10==0:
                    plt.axvline(x, c="k", ls="-.")
        else:
            for i, x in enumerate(epochs[1:]):
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
def parse_metrics(data, show=True):
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
            if metric.lower() == "binary":
                continue
            dev_scores[metric.lower()].append(score)
        
        if "Score:" in line and show:
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

def show_study_loss(name, title=None, epoch_cap=None, show=True):
    runs = get_runs(name)[1:] # skip first "run"
    loss_dfs = {}
    metrics_dfs = {}
    
    header = title
    for i, run in enumerate(runs):
        loss = parse_loss(run)
        metrics = parse_metrics(run, show)
        
        if title is None:
            header = name + f" Run:{i}"
        try:
            loss_df = show_data(loss, "Loss{}".format(header), epoch_cap, show)
            if show:
                plt.xlabel("Epochs")
                plt.ylabel("Loss")
                plt.show()

            metrics_df = show_data(metrics, "Metrics{}".format(header), epoch_cap, show)
            if show:
                plt.xlabel("Epochs")
                plt.ylabel("Score")
                plt.show()
            
            loss_dfs[i] = loss_df.T
            metrics_dfs[i] = metrics_df.T
            
            for row in run:
                if "Current" in row and show:
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
    plt_metric = avg_metric.plot(title="Metric "+header, figsize=(15,5))
    plt_metric.set_xlabel("Epochs")
    plt_metric.set_ylabel("Score")

    print("Final values:")
    display(avg_loss.tail(3))
    display(avg_metric.tail(3))

    return avg_loss, avg_metric


### Stats
def get_final_scores(study=None, runs = None, metric="absa", title="Dist. of final scores", studies=None, compare=None):
    iterator = runs if runs is not None else study[1]
    
    if study is not None:
        finals =  [
            study[1][run][metric].tail(1).item()
            for run in iterator
        ]
    elif studies is not None:
        finals =  [
            study[1][run][metric].tail(1).item()
            for run in iterator
            for study in studies
        ]
        final_dev = [
            list(study[1][run][metric].tail(2))[0]
            for run in iterator
            for study in studies
        ]
    else:
        return []
        
    ### Population distribution
    # x-axis ranges from min, max  
    mu =  np.mean(finals)
    sigma = np.std(finals)
    x = np.arange(mu-3*sigma, mu+3*sigma, 0.0001)
    
    dist =  norm.pdf(x, mu, sigma) 
    alpha = (2*sigma)+mu    
    beta = mu-(2*sigma)
    
    print("Population average (evaluation) ", mu)
    print("Population average (development)", sum(final_dev)/len(final_dev))
    
    print("Top    2.5% must be over", alpha)
    print("Bottom 2.5% must be under", beta)
    
    # set up plot
    plt.figure(figsize=(15,7)) 
    plt.title(title, fontsize=20)
    plt.xlim([mu-3*sigma, mu+3*sigma])
    
    # plot normal distribution with mean 0 and standard deviation 1
    plt.plot(x, dist)
    
    # plot alpha and beta thresholds
    plt.fill_between(x[x>alpha], dist[x>alpha], step="pre", alpha=0.6, facecolor='g')
    plt.fill_between(x[x<beta], dist[x<beta], step="pre", alpha=0.6, facecolor='r')
    
    # show grouped scores
    plt.hist(finals, bins=13, alpha=0.2)
    
    # readability
    legend = ["$N(\\theta_0, \\sigma_0^2)$", "$\\theta_\\alpha$", "$\\theta_\\beta$", "IMN scores"]
    xticks = (
        [beta, mu-sigma, mu, mu+sigma, alpha], 
        [
            f"$\\theta_\\beta$={round(beta, 4)}", 
            round(mu-sigma,4), 
            f"$\\theta_0$={round(mu, 4)}", 
            round(mu+sigma,4), 
            f"$\\theta_\\alpha$={round(alpha, 4)}"
        ], 
    )
    
    
    ### Other scores to compare
    colors = ["cadetblue", "brown", "lime", "cyan", "yellow", "magenta"]
    if compare is not None:
        for i, model in enumerate(compare):
            plt.bar([compare[model]], [5], color=colors[i], width=0.005, alpha=0.8)
            legend.append(model)
            plt.text(compare[model], 5.2, f"{model}\n{round(compare[model], 4)}", ha="center")
        
 
    plt.legend(legend, fontsize=13)
    plt.xticks(xticks[0], xticks[1], fontsize=13)
    plt.xlabel("Score", fontsize=12)
    plt.ylabel("Count")
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



