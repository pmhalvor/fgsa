import torch


def pad(batch):
    """
    Pad batches according to largest sentence.

    A sentence in the batch has shape [3, sentence_length] and looks like:
        (
            tensor([  102,  3707, 29922,  1773,  4658, 13502,  1773,  3325,  3357, 19275,
                    3896,  3638,  3169, 10566,  8030, 30337,  2857,  3707,  4297, 24718,
                    9426, 29916, 28004,  8004, 30337, 15271,  4895, 10219,  6083,  4297,
                    26375, 20322, 26273,   103]), 
            tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 
            tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0])
        )
    """
    longest_sentence = max([ids.size(0) for ids, _, _ in batch])
    padded_ids, padded_masks, padded_labels = [], [], []

    for id, mask, label in batch:
        padded_ids.append(
            torch.nn.functional.pad(
                id,
                (0, longest_sentence - id.size(0)),
                value=0  # padding token can vary between Berts
            )
        )
        padded_masks.append(
            torch.nn.functional.pad(
                mask,
                (0, longest_sentence - mask.size(0)),
                value=0  # 1 means item present, 0 means padding TODO check if needed 
            )
        )
        padded_labels.append(
            torch.nn.functional.pad(
                label,
                (0, longest_sentence - label.size(0)),
                value=-1  # NOTE cannot pad with 0 since thats used as label O FIXME make sure negative works
            )
        )

    ids = torch.stack(padded_ids).long()
    masks = torch.stack(padded_masks).long()
    labels = torch.stack(padded_labels).long()

    return ids, masks, labels


def compare(tensor_1, tensor_2):
    b = torch.eq(tensor_1, tensor_2)

    for ele in b:
        assert ele
    
    return True


def decode(labels, ignore_id=-1):
    """  
    Parameters:
        labels (list): single row of data containing labels to decode
        ignore_id (int): defaults to 0, since 0 excluded from evaluation? TODO check this

    Encodings

        Value       Label
        0           O                       \n
        1       B-Expression                \n
        2       I-Expression                \n
        3       B-Holder                    \n
        4       I-Holder                    \n
        5       B-Target-Positive           \n
        6       I-Target-Positive           \n
        7       B-Target-Negative           \n
        8       I-Target-Negative           \n

    """
    def append_all(e, h, p, t):
        expressions.append(e)
        holders.append(h)
        polarity.append(p)
        targets.append(t)
    
    expressions, holders, polarity, targets = [], [], [], []
    
    for ele in labels:
        if ele == 0:
            append_all(0, 0, 0, 0)
        elif ele == 1:
            append_all(1, 0, 0, 0)  # expression 1
        elif ele == 2:
            append_all(2, 0, 0, 0)  # expression 2
        elif ele == 3:
            append_all(0, 1, 0, 0)  # holder 1
        elif ele == 4:
            append_all(0, 2, 0, 0)  # holder 2
        elif ele == 5:
            append_all(0, 0, 1, 1)  # polarity 1  target 2
        elif ele == 6:
            append_all(0, 0, 1, 2)  # polarity 1  target 2
        elif ele == 7:
            append_all(0, 0, 2, 1)  # polarity 2  target 1
        elif ele == 8:
            append_all(0, 0, 2, 2)  # polarity 2  target 2
        else:
            append_all(0, 0, 0, 0)

    return expressions, holders, polarity, targets


def decode_batch(batch):
    """
    Wrapper class for decoding.

    Take a batch of labels containing one-hot encoded predictions,
    and splits back into lists for targets, expressions, and sentiment.

    NOTE: These evaluation metrics mirror baseline metrics.
    More detailed metrics should be built for our model. 
    """

    expressions, holders, polarity, targets = [], [], [], []

    for tensor in batch:
        e, h, p, t = decode(tensor.list())
        expressions.append(e)
        holders.append(h)
        polarity.append(p)
        targets.append(t)

    return (expressions, holders, polarity, targets)
    


