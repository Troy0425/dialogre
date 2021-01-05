from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class customDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.encodings = list(
            map(
                lambda data: tokenizer.encode_plus(
                    data.text_a,
                    data.text_b + '[SEP]' + data.text_c, 
                    return_tensors="pt", 
                    truncation='only_first', 
                    max_length=512, 
                    padding='max_length'
                ),
                self.data            
            )
        )
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {}
        item['input_ids'] = self.encodings[idx]['input_ids'].squeeze(0)
        item['attention_mask'] = self.encodings[idx]['attention_mask'].squeeze(0)
        item['token_type_ids'] = self.encodings[idx]['token_type_ids'].squeeze(0)
        item['labels'] = torch.FloatTensor(self.data[idx].label)
        return item

if __name__ == '__main__':
    pass