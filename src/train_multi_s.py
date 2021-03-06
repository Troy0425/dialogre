import random
import json
import logging
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
import torch
from dataset import customDataset, multiDataset
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

class bertsProcessor(DataProcessor): #bert_s
    def __init__(self):
        def is_speaker(a):
            a = a.split()
            return len(a) == 2 and a[0] == "speaker" and a[1].isdigit()
        
        def rename(d, x, y):
            unused = ["[unused1]", "[unused2]"]
            a = []
            if is_speaker(x):
                a += [x]
            else:
                a += [None]
            if x != y and is_speaker(y):
                a += [y]
            else:
                a += [None]
            for i in range(len(a)):
                if a[i] is None:
                    continue
                d = d.replace(a[i] + ":", unused[i] + " :")
                if x == a[i]:
                    x = unused[i]
                if y == a[i]:
                    y = unused[i]
            return d, x, y
            
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
                    d, h, t = rename('\n'.join(data[i][0]).lower(), data[i][1][j]["x"].lower(), data[i][1][j]["y"].lower())
                    d = [d,
                         h,
                         t,
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


def accuracy(out, labels):
    out = out.reshape(-1)
    out = 1 / (1 + np.exp(-out))
    return np.sum((out > 0.5) == (labels > 0.5)) / 36

def f1_eval(logits, dataset):
    def getpred(result, T1 = 0.5, T2 = 0.4):
        ret = []
        for i in range(len(result)):
            r = []
            maxl, maxj = -1, -1
            for j in range(len(result[i])):
                if result[i][j] > T1:
                    r += [j]
                if result[i][j] > maxl:
                    maxl = result[i][j]
                    maxj = j
            if len(r) == 0:
                if maxl <= T2:
                    r = [36]
                else:
                    r += [maxj]
            ret += [r]
        return ret

    def geteval(devp, data):
        correct_sys, all_sys = 0, 0
        correct_gt = 0
        
        for i in range(len(data)):
            for id in data[i]:
                if id != 36:
                    correct_gt += 1
                    if id in devp[i]:
                        correct_sys += 1

            for id in devp[i]:
                if id != 36:
                    all_sys += 1

        precision = 1 if all_sys == 0 else correct_sys/all_sys
        recall = 0 if correct_gt == 0 else correct_sys/correct_gt
        f_1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0
        return f_1

    logits = np.asarray(logits)
    logits = list(1 / (1 + np.exp(-logits)))

    labels_onehot = list(map(lambda x: x.label, dataset))
    
    labels = []
    for f in labels_onehot:
        label = []
        assert(len(f) == 36)
        for i in range(36):
            if f[i] == 1:
                label += [i]
        if len(label) == 0:
            label = [36]
        labels += [label]


    assert(len(labels) == len(logits))
    
    bestT2 = bestf_1 = 0
    for T2 in range(51):
        devp = getpred(logits, T2=T2/100.)
        print('-' * 20)
        print('pred')
        print(devp[:10])
        print('label')
        print(labels[:10])
        f_1 = geteval(devp, labels)
        if f_1 > bestf_1:
            bestf_1 = f_1
            bestT2 = T2/100.

    return bestf_1, bestT2

class multiTaskBert(torch.nn.Module):
    def __init__(self):
        super(multiTaskBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.class_dropout = nn.Dropout(0.1)
        self.class_classifier = nn.Linear(768, 36)
        self.trigger_dropout = nn.Dropout(0.1)
        self.trigger_classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooler_output = bert_output.pooler_output
        pooler_output = self.class_dropout(pooler_output)
        class_logits = self.class_classifier(pooler_output)

        last_hidden_state = bert_output.last_hidden_state
        last_hidden_state = self.trigger_dropout(last_hidden_state)
        trigger_logits = self.trigger_classifier(last_hidden_state)

        return class_logits, trigger_logits

def main():
    epochs = 50
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = bertsProcessor()

    train_data = processor.get_train_examples('')
    train_dataset = multiDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    val_data = processor.get_dev_examples('')
    val_dataset = multiDataset(val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    model = multiTaskBert()
    model = model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    class_loss_fn = nn.BCEWithLogitsLoss()
    trigger_loss_fn = nn.CrossEntropyLoss(torch.FloatTensor([1,10]).to(device))
    best_metric = 0
    output_dir = 'model/'
    trigger_weight = 0.25
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            trigger_labels = batch['trigger_labels'].to(device)
            trigger_masks = batch['trigger_masks'].to(device)
            class_logits, trigger_logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            class_loss = class_loss_fn(class_logits, labels)

            trigger_logits = trigger_logits.view(-1, 2)
            trigger_labels = torch.where(
                trigger_masks.view(-1) == 1, trigger_labels.view(-1), torch.tensor(trigger_loss_fn.ignore_index).type_as(trigger_labels)
            )
            trigger_loss = trigger_loss_fn(trigger_logits, trigger_labels)
            loss = class_loss + trigger_loss * trigger_weight

            loss.backward()
            optimizer.step()
        
        model.eval()
        eval_class_loss, eval_trigger_loss, eval_accuracy = 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        logits_all = []
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            trigger_labels = batch['trigger_labels'].to(device)
            trigger_masks = batch['trigger_masks'].to(device)
            with torch.no_grad():
                class_logits, trigger_logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                tmp_eval_class_loss = class_loss_fn(class_logits, labels)

                trigger_logits = trigger_logits.view(-1, 2)
                trigger_labels = torch.where(
                    trigger_masks.view(-1) == 1, trigger_labels.view(-1), torch.tensor(trigger_loss_fn.ignore_index).type_as(trigger_labels)
                )
                tmp_eval_trigger_loss = trigger_loss_fn(trigger_logits, trigger_labels)
            
            class_logits = class_logits.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()
            for i in range(len(class_logits)):
                logits_all += [class_logits[i]]
            
            tmp_eval_accuracy = accuracy(class_logits, labels.reshape(-1))

            eval_class_loss += tmp_eval_class_loss.mean().item()
            eval_trigger_loss += tmp_eval_trigger_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_class_loss = eval_class_loss / nb_eval_steps
        eval_trigger_loss = eval_trigger_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        
        result = {
            'eval_class_loss': eval_class_loss,
            'eval_trigger_loss': eval_trigger_loss
        }

        eval_f1, eval_T2 = f1_eval(logits_all, val_data)
        result["f1"] = eval_f1
        result["T2"] = eval_T2

        logger.info("***** Epoch {} Eval results *****".format(epoch + 1))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        logger.info("  %s = %s", 'best_f1', str(best_metric))
        
        if eval_f1 >= best_metric:
            torch.save(model.state_dict(), os.path.join(output_dir, "model_best.pt"))
            best_metric = eval_f1
        # else:
        #     if eval_accuracy >= best_metric:
        #         torch.save(model.state_dict(), os.path.join(output_dir, "model_best.pt"))
        #         best_metric = eval_accuracy

    logger.info("***** Results *****")
    logger.info("  %s = %s", 'best_f1', str(best_metric))
    model.load_state_dict(torch.load(os.path.join(output_dir, "model_best.pt")))
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))


if __name__ == '__main__':
    main()
    
