import torch
from torch.utils.data import Dataset

class TokenStreamDataset(Dataset):
    def __init__(self, tokens, context_length):
        self.tokens = torch.as_tensor(tokens, dtype=torch.long)
        self.context_length = int(context_length)

    def __len__(self):
        return max(0, len(self.tokens) - self.context_length)

    def __getitem__(self, i):
        return (self.tokens[i:i+self.context_length],
                self.tokens[i+1:i+self.context_length+1])
