import torch.utils.data
from torchtext.datasets import Multi30k


class T5Multi30kEnDe(torch.utils.data.Dataset):
    """
    Dataset for Multi30k English-German translation task on T5.

    Structure of dataset:
    ```
    [
        {
            'src': 'translate English to German: <English sentence>',
            'tgt': '<German sentence>'
        },
        ...
    ]
    ```

    `__getitem__` returns a dictionary with keys 'src' and 'tgt'.
    """
    def __init__(self, split='train'):
        assert split in ['train', 'valid', 'test'], "Split must be one of 'train', 'valid', or 'test'"

        super(T5Multi30kEnDe, self).__init__()
        self.dataset = Multi30k(split=split, language_pair=('en', 'de'))
        self.dataset = list(self.dataset)
        self.dataset = [{'src': 'translate English to German: ' + en, 'tgt': de} for en, de in self.dataset]
        self.length = len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.length
