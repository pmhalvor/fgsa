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
