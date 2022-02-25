import pandas as pd

def parse_taskwise_loss(data):
    """Assume data is f.readlines() of a log file."""
    task_loss = {
        "expression": [],
        "holder": [],
        "polarity": [],
        "target": [],
    }
    for line in data:
        tokens = line.split(' ')
        for i, t in enumerate(tokens):
            if "loss" in t:
                task = line.split("INFO] ")[-1].split('loss:')[0].strip()
                loss = float(t.split(':')[-1])
                task_loss[task].append(loss)
    return task_loss


def parse_batchwise_loss(data):
    """Assume data is f.readlines() of a log file."""
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


def show_taskwise_loss(name, get_epochs=True):
    def show_epochs(epochs):
        for x in epochs:
            plt.axvline(x, c="k", ls="-.")
        

    with open(name) as f:
        data = f.readlines()

    loss, epochs = parse_taskwise_loss(data, get_epochs)

    df = pd.DataFrame.from_dict(loss)
    plt = df.plot(title=name)
    show_epochs(epochs) if get_epochs else None
    
    return df.T  # show table horizontally


def show_batchwise_loss(name):
    with open(name) as f:
        data = f.readlines()

    loss = parse_batchwise_loss(data)
    
    df = pd.DataFrame.from_dict(loss, orient="index")
    df.plot(title=name)
    
    return df

