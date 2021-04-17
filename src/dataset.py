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
                    data.text_b + ' [SEP] ' + data.text_c, 
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

class multiDataset(Dataset):
    def __init__(self, data):
        self.max_length = 512
        self.data = data
        self.trigger_labels = self.create_trigger_label(self.data)
        self.encodings = list(
            map(
                lambda data: tokenizer.encode_plus(
                    data.text_a,
                    data.text_b + ' [SEP] ' + data.text_c, 
                    #return_tensors="np", 
                    truncation='only_first', 
                    max_length=512, 
                    padding='max_length',
                ),
                self.data  
            )
        )
        self.trigger_masks = self.create_trigger_mask()

    def pad_to_len(self, data, max_len):
        return data[:max_len] + [0] * max(max_len - len(data), 0)

    def create_trigger_label(self, data):
        print('create trigger label')
        trigger_labels = []
        for i in range(len(data)):
            triggers = data[i].trigger
            token = tokenizer.tokenize('[CLS] ' + data[i].text_a)
            token_len = len(token)
            trigger_label = [0] * len(token)
            for trigger in triggers:
                trigger_token = tokenizer.tokenize(trigger)
                trigger_token_len = len(trigger_token)
                for offset in range(token_len - trigger_token_len + 1):
                    if token[offset: offset + trigger_token_len] == trigger_token:
                        trigger_label[offset: offset + trigger_token_len] = [1] * trigger_token_len
            assert(len(trigger_label) == len(token))
            trigger_label = self.pad_to_len(trigger_label, self.max_length)
            assert(len(trigger_label) == self.max_length)
            trigger_labels.append(trigger_label)

        return trigger_labels

    def create_trigger_mask(self):
        print('create trigger mask')
        trigger_masks = []
        for i in range(len(self.encodings)):
            token_type_ids = self.encodings[i]['token_type_ids']
            trigger_mask = [0] * self.max_length
            for j in range(1, self.max_length):
                if token_type_ids[j + 1] == 0:
                    trigger_mask[j] = 1
                else:
                    break
            trigger_masks.append(trigger_mask)
        return trigger_masks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {}
        item['input_ids'] = torch.LongTensor(self.encodings[idx]['input_ids']).squeeze(0)
        item['attention_mask'] = torch.LongTensor(self.encodings[idx]['attention_mask']).squeeze(0)
        item['token_type_ids'] = torch.LongTensor(self.encodings[idx]['token_type_ids']).squeeze(0)
        item['labels'] = torch.FloatTensor(self.data[idx].label)
        item['trigger_labels'] = torch.LongTensor(self.trigger_labels[idx])
        item['trigger_masks'] = torch.LongTensor(self.trigger_masks[idx])
        return item

class triggerDataset(Dataset):
    def __init__(self, data):
        self.max_length = 512
        self.data = data
        self.encodings = list(
            map(
                lambda data: tokenizer.encode_plus(
                    data.text_a,
                    data.text_b + ' [SEP] ' + data.text_c + ' [SEP] '.join([''] + data.trigger), 
                    #return_tensors="np", 
                    truncation='only_first', 
                    max_length=512, 
                    padding='max_length',
                ),
                self.data  
            )
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {}
        item['input_ids'] = torch.LongTensor(self.encodings[idx]['input_ids']).squeeze(0)
        item['attention_mask'] = torch.LongTensor(self.encodings[idx]['attention_mask']).squeeze(0)
        item['token_type_ids'] = torch.LongTensor(self.encodings[idx]['token_type_ids']).squeeze(0)
        item['labels'] = torch.FloatTensor(self.data[idx].label)
        return item

if __name__ == '__main__':
    a = tokenizer.encode_plus(
        'I am troy',
        'Troy' + ' [SEP] ' + 'AAA', 
        #return_tensors="np", 
        #truncation='only_first', 
        max_length=512, 
        padding='max_length',
    )
    print(a)
    pass