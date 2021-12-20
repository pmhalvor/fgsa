import torch.nn
import torch 


class Conv1DWithMasking(torch.nn.Conv1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Conv1DWithMasking, self).__init__(**kwargs)
    
    def compute_mask(self, x, mask):
        return mask



class Self_attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        PyTorch layer that implements a self-attention mechanism.
        """

        