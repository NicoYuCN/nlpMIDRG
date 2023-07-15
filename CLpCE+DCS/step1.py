import csv
import torch
import torch.utils.data as tud
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import random
from torch.nn.utils.rnn import pad_sequence

TRAIN_DATA_PATH = 'data/train.csv'
DEV_DATA_PATH = 'data/valid.csv'
TOKENIZER_PATH = 'data/volab.txt'
BATCH_SIZE = 32
MAX_LEN = 230  # 输入模型的最大长度，不能超过config中n_ctx的值


# """随机数种子"""
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# random.seed(0)


def collate_fn(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据
    Returns:
    """
    input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
    #     input_sim_list, attention_sim_list = [], []
    for instance in batch_data:
        # 按照batch中的最大数据长度,对数据进行padding填充
        input_ids_temp = instance["input_ids"]
        token_type_ids_temp = instance["token_type_ids"]
        attention_mask_temp = instance["attention_mask"]

        #         attention_sim_temp = instance["attention_sim"]
        #         input_sim_temp = instance["input_sim"]

        # 将input_ids_temp和token_type_ids_temp添加到对应的list中
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        token_type_ids_list.append(torch.tensor(token_type_ids_temp, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask_temp, dtype=torch.long))

    #         input_sim_list.append(torch.tensor(input_sim_temp, dtype=torch.long))
    #         attention_sim_list.append(torch.tensor(attention_sim_temp, dtype=torch.long))

    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "token_type_ids": pad_sequence(token_type_ids_list, batch_first=True, padding_value=0),
            "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0)}


#             "attention_sim": pad_sequence(attention_sim_list, batch_first=True, padding_value=0),
#             "input_sim": pad_sequence(input_sim_list, batch_first=True, padding_value=0)}

class SummaryDataset(tud.Dataset):
    def __init__(self, data_path, tokenizer_path, max_len, name=None):
        super(SummaryDataset, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        # 内容正文和摘要分别用content_id，summary_id区分表示
        self.content_id = self.tokenizer.convert_tokens_to_ids("[Content]")
        self.summary_id = self.tokenizer.convert_tokens_to_ids("[Summary]")
        self.sep_id = self.tokenizer.sep_token_id
        self.cls_id = self.tokenizer.cls_token_id

        self.data_set = []

        data = pd.read_csv(data_path, encoding='utf-8')

        if name == "train":
            data = data[:len(data) - 6000]

        for pos in range(len(data['main'])):

            summary = data['label'][pos]
            content = data['main'][pos]

            input_ids = []
            token_type_ids = []

            summary_tokens = self.tokenizer.tokenize(summary)

            content_tokens = self.tokenizer.tokenize(content)
            # 如果正文过长，进行截断
            if len(content_tokens) > max_len - len(summary_tokens) - 3:
                content_tokens = content_tokens[:max_len - len(summary_tokens) - 3]

            input_ids.append(self.cls_id)
            token_type_ids.append(self.content_id)
            input_ids.extend(self.tokenizer.convert_tokens_to_ids(content_tokens))
            token_type_ids.extend([self.content_id] * len(content_tokens))
            input_ids.append(self.sep_id)
            token_type_ids.append(self.content_id)
            input_ids.extend(self.tokenizer.convert_tokens_to_ids(summary_tokens))
            token_type_ids.extend([self.summary_id] * len(summary_tokens))
            input_ids.append(self.sep_id)
            token_type_ids.append(self.summary_id)

            assert len(input_ids) == len(token_type_ids)
            assert len(input_ids) <= max_len

            self.data_set.append(
                {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": [1] * len(input_ids)})

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]


def collate_fn_train(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据
    Returns:
    """
    input_ids_list_1, attention_mask_list_1 = [], []
    for instance in batch_data:
        # 按照batch中的最大数据长度,对数据进行padding填充
        input_ids_temp_1 = instance["input_ids_1"]
        attention_mask_temp_1 = instance["mask_1"]
        # 将input_ids_temp和token_type_ids_temp添加到对应的list中
        input_ids_list_1.append(torch.tensor(input_ids_temp_1, dtype=torch.long))
        attention_mask_list_1.append(torch.tensor(attention_mask_temp_1, dtype=torch.long))
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids_1": pad_sequence(input_ids_list_1, batch_first=True, padding_value=0),
            "attention_mask_1": pad_sequence(attention_mask_list_1, batch_first=True, padding_value=0)}


class SimiDataset_Train(tud.Dataset):
    def __init__(self, data_path, tokenizer_path, max_len):
        super(SimiDataset_Train, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.max_len = max_len

        self.data_set = []

        data = pd.read_csv(data_path, encoding='utf-8')

        data = data[:len(data) - 6000]

        for pos in range(len(data['main'])):

            sent1 = data['label'][pos]

            tokens_1 = self.tokenizer.tokenize(sent1)
            if len(tokens_1) > self.max_len - 2:
                tokens_1 = tokens_1[:self.max_len]
            input_ids_1 = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens_1 + ['[SEP]'])
            mask_1 = [1] * len(input_ids_1)
            for k in range(2):
                self.data_set.append({"input_ids_1": input_ids_1, "mask_1": mask_1})

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]


def getdataset():

    traindataset = SummaryDataset(TRAIN_DATA_PATH, TOKENIZER_PATH, MAX_LEN, "train")
    traindataloader = tud.DataLoader(traindataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    valdataset = SummaryDataset(DEV_DATA_PATH, TOKENIZER_PATH, MAX_LEN)
    valdataloader = tud.DataLoader(valdataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    simtraindataset = SimiDataset_Train(TRAIN_DATA_PATH, TOKENIZER_PATH, 80)
    simtraindataloader = tud.DataLoader(simtraindataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn_train)

    return traindataset, traindataloader, valdataset, valdataloader, simtraindataset, simtraindataloader
