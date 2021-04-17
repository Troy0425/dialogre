import random
import json
import logging
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from dataset import customDataset
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from torch import nn
import numpy as np
import os

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None, trigger=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label
        self.trigger = trigger

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class bertProcessor(DataProcessor): #bert
    def __init__(self):
        random.seed(42)
        self.D = [[], [], []]
        for sid in range(3):
            with open("data_v2/en/data/"+["train.json", "dev.json", "test.json"][sid], "r", encoding="utf8") as f:
                data = json.load(f)
            if sid == 0:
                random.shuffle(data)
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    rid = []
                    trigger = []
                    for k in range(36):
                        if k+1 in data[i][1][j]["rid"]:
                            rid += [1]
                        else:
                            rid += [0]
                    for t in data[i][1][j]["t"]:
                        trigger += [t.lower()]
                    
                    d = ['\n'.join(data[i][0]).lower(),
                         data[i][1][j]["x"].lower(),
                         data[i][1][j]["y"].lower(),
                         rid,
                         trigger]
                    self.D[sid] += [d]
        logger.info(str(len(self.D[0])) + "," + str(len(self.D[1])) + "," + str(len(self.D[2])))
        
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[0], "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[2], "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[1], "dev")

    def get_labels(self):
        """See base class."""
        return [str(x) for x in range(2)]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, d) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text_a = data[i][0]
            text_b = data[i][1]
            text_c = data[i][2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=data[i][3], text_c=text_c, trigger=data[i][4]))
            
        return examples

def count_trigger(datas):
    count_all = 0
    count_all_trigger = 0
    for t in datas:
        count_now = 0
        count_now_trigger = 0
        data_now = datas[t]
        for data in data_now:
            triggers = data.trigger
            for trigger in triggers:
                count_all += 1
                count_now += 1
                if trigger != '':
                    count_all_trigger += 1
                    count_now_trigger += 1
        print(t, ': ', count_now_trigger / count_now)
    print('Total: ', count_all_trigger / count_all)

def main():
    processor = bertProcessor()
    datas = {}
    datas['train_data'] = processor.get_train_examples('')
    datas['val_data'] = processor.get_dev_examples('')
    datas['test_data'] = processor.get_test_examples('')
    count_trigger(datas)

if __name__ == '__main__':
    main()