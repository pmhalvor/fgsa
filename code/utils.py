import torch
import numpy as np
from sklearn.metrics import f1_score


def pad(batch, ignore_id=-1):
    """
    Pad batches according to largest sentence.

    A sentence in the batch has shape [6, sentence_length] and looks like:
        (
            tensor([  102,  3707, 29922,  1773,  4658, 13502,  1773,  3325,  3357, 19275,
                    3896,  3638,  3169, 10566,  8030, 30337,  2857,  3707,  4297, 24718,
                    9426, 29916, 28004,  8004, 30337, 15271,  4895, 10219,  6083,  4297,
                    26375, 20322, 26273,   103]), 
            tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 
            tensor([0, 0, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0])
            tensor([0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0])
            tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,0])
            tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0,0])
        )
    """
    longest_sentence = max([b[0].size(0) for b in batch])
    padded_ids, padded_masks = [], []
    padded_expression, padded_holder, padded_polarity, padded_target= [], [], [], []

    for b in batch:
        # this manual unpacking should not be necessary..
        ids = b[0]
        mask = b[1]
        expression = b[2]
        holder = b[3]
        polarity = b[4]
        target = b[5]
        
        padded_ids.append(
            torch.nn.functional.pad(
                ids,
                (0, longest_sentence - ids.size(0)),
                value=0  # padding token can vary between Berts
            )
        )
        padded_masks.append(
            torch.nn.functional.pad(
                mask,
                (0, longest_sentence - mask.size(0)),
                value=0  # 1 means item present, 0 means padding
            )
        )
        padded_expression.append(
            torch.nn.functional.pad(
                expression,
                (0, longest_sentence - expression.size(0)),
                value=ignore_id  # NOTE cannot pad with 0 since thats used as label O
            )
        )
        padded_holder.append(
            torch.nn.functional.pad(
                holder,
                (0, longest_sentence - holder.size(0)),
                value=ignore_id  # NOTE cannot pad with 0 since thats used as label O
            )
        )
        padded_polarity.append(
            torch.nn.functional.pad(
                polarity,
                (0, longest_sentence - polarity.size(0)),
                value=ignore_id  # NOTE cannot pad with 0 since thats used as label O
            )
        )
        padded_target.append(
            torch.nn.functional.pad(
                target,
                (0, longest_sentence - target.size(0)),
                value=ignore_id  # NOTE cannot pad with 0 since thats used as label O
            )
        )

    ids = torch.stack(padded_ids).long()
    masks = torch.stack(padded_masks).long()
    expression = torch.stack(padded_expression).long()
    holder = torch.stack(padded_holder).long()
    polarity = torch.stack(padded_polarity).long()
    target = torch.stack(padded_target).long()

    return ids, masks, expression, holder, polarity, target


def compare(tensor_1, tensor_2):
    b = torch.eq(tensor_1, tensor_2)

    for ele in b:
        assert ele
    
    return True
  

def score(true_aspect, predict_aspect, true_sentiment, predict_sentiment, train_op):
    """
    Evaluation metric used in IMN and RACL. 
    Takes batch of inputs as lists. 

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
                    # NOTE change by pmhalvor: typecast to int for pytorch batches
                    if int(true_sentiment[i][num])!=0:
                        total_count[polarity_map[int(true_sentiment[i][num])]]+=1
                     
                if predict[num] == begin:
                    match = True 
                    for j in range(num+1, len(true_seq)):
                        if true_seq[j] == inside and predict[j] == inside:
                            continue
                        elif true_seq[j] != inside and predict[j] != inside:
                            break
                        else:
                            match = False  # this is incredibly strict
                            break

                    if match:
                        correct += 1
                        if not train_op:
                            # do not count conflict examples
                            # NOTE change by pmhalvor: typecast to ints for pytorch batches
                            if int(true_sentiment[i][num])!=0:
                                rel_count[polarity_map[int(true_sentiment[i][num])]]+=1
                                # NOTE change by pmhalvor: do not assume predictions!=0 for all true labels
                                if int(predict_sentiment[i][num])!=0:
                                    pred_count[polarity_map[int(predict_sentiment[i][num])]]+=1
                                    if int(true_sentiment[i][num]) == int(predict_sentiment[i][num]):
                                        correct_count[polarity_map[int(true_sentiment[i][num])]]+=1

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
        # TODO implement the correct, keep link to discussion
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


def ez_score(true_labels, predict_labels, num_labels):
    """
    F1-score for _any_ correctly guessed label. Much more lenient than score() from RACL (above).

    Parameters:
        true_labels (torch.Tensor): batched true labels of size [batchsize, seq_len] 
        predict_labels (torch.Tensor): batched predictions size [batchsize, seq_len]
        num_labels (int): number of labels model is learning ot predict
    """

    total = 0
    
    for true, pred in zip(true_labels, predict_labels):
        total += f1_score(
            true, 
            pred, 
            labels=[e for e in range(1, num_labels)],
            average='micro',
            zero_division=1,  # set score to 1 when all labels and predictions are 0
        )
    return total/true_labels.shape[0]
