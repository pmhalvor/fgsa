import torch
import numpy as np


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


def decode(labels, mask):
    """  
    Parameters:
        labels (list): single row of data containing labels to decode
        mask (None): parameter never used

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


def decode_target(labels, mask):
    """  
    Parameters:
        labels (list): single row of data containing labels to decode
        mask (None): parameter never used

    Encodings

        Value       Label
        0           O                \n
        1       B-Positive           \n
        2       I-Positive           \n
        3       B-Negative           \n
        4       I-Negative           \n

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
            append_all(0, 0, 1, 1)  # polarity 1  target 1
        elif ele == 2:
            append_all(0, 0, 1, 2)  # polarity 1  target 2
        elif ele == 3:
            append_all(0, 0, 2, 1)  # polarity 2  target 1
        elif ele == 4:
            append_all(0, 0, 2, 2)  # polarity 2  target 2
        else:
            append_all(0, 0, 0, 0)

    return expressions, holders, polarity, targets


def decode_mask(labels, mask):
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
    
    for ele, m in zip(labels, mask):
        if m == 0:
            break
        elif ele == 0:
            append_all(0, 0, 0, 0)  # outside
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


def decode_mask_target(labels, mask):
    """  
    Parameters:
        labels (list): single row of data containing labels to decode
        ignore_id (int): defaults to 0, since 0 excluded from evaluation? TODO check this

    Encodings

        Value       Label
        0           O                \n
        5       B-Positive           \n
        6       I-Positive           \n
        7       B-Negative           \n
        8       I-Negative           \n

    """
    def append_all(e, h, p, t):
        expressions.append(e)
        holders.append(h)
        polarity.append(p)
        targets.append(t)
    
    expressions, holders, polarity, targets = [], [], [], []
    
    for ele, m in zip(labels, mask):
        if m == 0:
            break
        elif ele == 0:
            append_all(0, 0, 0, 0)  # outside
        elif ele == 1:
            append_all(0, 0, 1, 1)  # polarity 1  target 1
        elif ele == 2:
            append_all(0, 0, 1, 2)  # polarity 1  target 2
        elif ele == 3:
            append_all(0, 0, 2, 1)  # polarity 2  target 1
        elif ele == 4:
            append_all(0, 0, 2, 2)  # polarity 2  target 2
        else:
            append_all(0, 0, 0, 0)

    return expressions, holders, polarity, targets


def decode_batch(batch, mask=None, targets_only=False):
    """
    Wrapper class for decoding.

    Take a batch of labels containing one-hot encoded predictions,
    and splits back into lists for targets, expressions, and sentiment.

    NOTE: These evaluation metrics mirror baseline metrics.
    More detailed metrics should be built for our model. 
    """

    expressions, holders, polarities, targets = [], [], [], []

    # decide decoder according what data we are looking at
    if mask is not None and targets_only is True:
        decoder = decode_mask_target  # FIXME clean this up
    elif mask is not None:
        decoder = decode_mask 
    elif targets_only is True:
        decoder = decoder_target
        mask = batch
    else:
        decoder = decode
        mask = batch

    for tensor, m in zip(batch, mask):
        e, h, p, t = decoder(tensor.tolist(), mask=m.tolist())
        expressions.append(e)
        holders.append(h)
        polarities.append(p)
        targets.append(t)

    return {
        "expressions": expressions, 
        "holders": holders, 
        "polarities": polarities, 
        "targets": targets,
    }
    

def score(true_aspect, predict_aspect, true_sentiment, predict_sentiment, train_op):
    """
    Takes batch of inputs as lists

    Parameters:
        true_aspect (list): not padded, built from same decoding as predicted
        predict_aspect (list): not padded, expects used masked when decoding
        true_sentiment (list): labels for target_polarity.txt. padded or not?
        predict_sentiment (list): 
    """
    
    if train_op:
        begin = 1
        inside = 2
    else:
        begin = 1
        inside = 2

        # predicted sentiment distribution for aspect terms that are correctly extracted
        pred_count = {'pos':0, 'neg':0, 'neu':0}
        # gold sentiment distribution for aspect terms that are correctly extracted
        rel_count = {'pos':0, 'neg':0, 'neu':0}
        # sentiment distribution for terms that get both span and sentiment predicted correctly
        correct_count = {'pos':0, 'neg':0, 'neu':0}
        # sentiment distribution in original data
        total_count = {'pos':0, 'neg':0, 'neu':0}

        polarity_map = {1: 'pos', 2: 'neg', 3: 'neu'}

        # count of predicted conflict aspect term
        predicted_conf = 0

    correct, predicted, relevant = 0, 0, 0

    for i in range(len(true_aspect)):
        true_seq = true_aspect[i]
        predict = predict_aspect[i]
        
        for num in range(len(true_seq)):
            # print('num', true_seq[num])
            if true_seq[num] == begin:
                relevant += 1
                if not train_op:
                    if true_sentiment[i][num]!=0:
                        total_count[polarity_map[true_sentiment[i][num]]]+=1
                     
                if predict[num] == begin:
                    match = True 
                    for j in range(num+1, len(true_seq)):
                        if true_seq[j] == inside and predict[j] == inside:
                            continue
                        elif true_seq[j] != inside  and predict[j] != inside:
                            break
                        else:
                            match = False
                            break

                    if match:
                        correct += 1
                        if not train_op:
                            # do not count conflict examples
                            if true_sentiment[i][num]!=0:
                                rel_count[polarity_map[true_sentiment[i][num]]]+=1
                                pred_count[polarity_map[predict_sentiment[i][num]]]+=1
                                if true_sentiment[i][num] == predict_sentiment[i][num]:
                                    correct_count[polarity_map[true_sentiment[i][num]]]+=1

                            else:
                                predicted_conf += 1



        for pred in predict:
            if pred == begin:
                predicted += 1

    p_aspect = correct / (predicted + 1e-6)
    r_aspect = correct / (relevant + 1e-6)
    # F1 score for aspect (opinion) extraction
    f_aspect = 2 * p_aspect * r_aspect / (p_aspect + r_aspect + 1e-6)

    acc_s, f_s, f_absa = 0, 0, 0

    if not train_op:
        num_correct_overall = correct_count['pos']+correct_count['neg']+correct_count['neu']
        num_correct_aspect = rel_count['pos']+rel_count['neg']+rel_count['neu']
        num_total = total_count['pos']+total_count['neg']+total_count['neu']

        acc_s = num_correct_overall/(num_correct_aspect+1e-6)
       
        p_pos = correct_count['pos'] / (pred_count['pos']+1e-6)
        r_pos = correct_count['pos'] / (rel_count['pos']+1e-6)
        
        p_neg = correct_count['neg'] / (pred_count['neg']+1e-6)
        r_neg = correct_count['neg'] / (rel_count['neg']+1e-6)

        p_neu = correct_count['neu'] / (pred_count['neu']+1e-6)
        r_neu= correct_count['neu'] / (rel_count['neu']+1e-6)

        pr_s = (p_pos+p_neg+p_neu)/3.0
        re_s = (r_pos+r_neg+r_neu)/3.0

        # For calculating the F1 Score for SC, we have discussed with Ruidan at https://github.com/ruidan/IMN-E2E-ABSA/issues?q=is%3Aissue+is%3Aclosed.
        # We provide the correct formula as follow, but we still adopt the calculation in IMN to conduct a fair comparison.
        # TODO implement the correct, keep link to dicussion
        # f_pos = 2*p_pos*r_pos /(p_pos+r_pos+1e-6)
        # f_neg = 2*p_neg*r_neg /(p_neg+r_neg+1e-6)
        # f_neu = 2*p_neu*r_neu /(p_neu+r_neu+1e-6)
        # f_s = (f_pos+f_neg+f_neu)/3.0

        # F1 score for SC only (in IMN)
        f_s = 2*pr_s*re_s/(pr_s+re_s+1e-6)

        precision_absa = num_correct_overall/(predicted+1e-6 - predicted_conf)
        recall_absa = num_correct_overall/(num_total+1e-6)
        # F1 score of the end-to-end task
        f_absa = 2*precision_absa*recall_absa/(precision_absa+recall_absa+1e-6)

    return f_aspect, acc_s, f_s, f_absa



