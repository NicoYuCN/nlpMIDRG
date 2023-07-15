import torch
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, BertModel, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.nn.functional as F
from step1 import getdataset   # 数据集

# 数据集
traindataset, traindataloader, valdataset, valdataloader, simtraindataset, simtraindataloader = getdataset()

N_EPOCHS = 100
LR = 5e-4
WARMUP_PROPORTION = 0.1
MAX_GRAD_NORM = 1.0
SUMMARY_ID = 2
device = 'cuda'


def calculate_loss(outputs, labels, token_type_ids, summary_id):
    """
    只计算summary部分的loss
    """
    logits = outputs[0]  # 维度:[batch_size, sequence_length, config.vocab_size]

    # 获取mask值，token_type_ids中等于summary_id的部分需要计算loss，标记为1；否则为0。
    # size:[batch_size, sequence_length]
    mask = (token_type_ids == summary_id).long()
    # 获取新的标签，size:[batch_size, sequence_length]
    labels = labels * mask
    # 对预测结果和标签进行偏移操作
    # GPT2的生成机制为通过前面的token，预测下一个token；并且labels与input_ids相同，
    # 因此input_ids中的第一个token的预测结果，实际上是标签中的第二个token，以此类推，最终仅计算sequence_length-1个token的loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # 定义损失函数CrossEntropyLoss，并且设置忽略计算loss的索引，以及返回loss的形式
    # 忽略shift_labels中为0的loss，也就是仅计算summary部分的损失值
    loss_fct = CrossEntropyLoss(ignore_index=0)

    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    return loss


def simcse_unsup_loss(y_pred):
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size, 768]
    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)

    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)

    return loss


class SimcseModel(nn.Module):
    """Simcse有监督模型定义"""

    def __init__(self):
        super(SimcseModel, self).__init__()
        self.model_config = GPT2Config.from_json_file('config/config.json')
        self.gpt = GPT2LMHeadModel(config=self.model_config)
        self.gpt.resize_token_embeddings(1300)

    #         self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=12)

    def forward(self, input_ids, attention_mask, book=None):

        if book == "sim":
            output = self.gpt(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_state = output.hidden_states[-1]  # [batch_size, seq_len, 768]
            #             last_hidden_state = self.attention(last_hidden_state, last_hidden_state, last_hidden_state)[0]
            out = last_hidden_state.permute(0, 2, 1)
            out = nn.AvgPool1d(out.size(2))(out).squeeze(2)

            return out

        else:
            output = self.gpt(input_ids=input_ids, attention_mask=attention_mask)

            return output


def run(kk):
    best_valid_loss = float('inf')
    model = SimcseModel().to(device)

    total_steps = len(traindataloader) * N_EPOCHS
    optimizer = AdamW(model.parameters(), lr=LR, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(WARMUP_PROPORTION * total_steps),
                                                num_training_steps=total_steps)

    loss_vals = []
    loss_vals_eval = []
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = []
        pbar = tqdm(traindataloader)
        pbar.set_description("[Train Epoch {}]".format(epoch))

        for batch_idx, batch_data in enumerate(pbar):

            for pos, batch_data_ in enumerate(simtraindataloader):

                if batch_idx == pos:
                    input_ids = batch_data["input_ids"].to(device)
                    token_type_ids = batch_data["token_type_ids"].to(device)
                    attention_mask = batch_data["attention_mask"].to(device)

                    input_ids_1 = batch_data_["input_ids_1"].to(device)
                    attention_mask_1 = batch_data_["attention_mask_1"].to(device)

                    model.zero_grad()

                    outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask)  # 正常
                    loss_mle = calculate_loss(outputs, input_ids, token_type_ids, SUMMARY_ID)  # 正常

                    outputs_ = model.forward(input_ids=input_ids_1, attention_mask=attention_mask_1, book="sim")
                    loss_sim = simcse_unsup_loss(outputs_)

                    loss = (1-kk)*loss_mle + kk*loss_sim
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    epoch_loss.append(loss.item())

                    optimizer.step()
                    scheduler.step()

        loss_vals.append(np.mean(epoch_loss))

        model.eval()
        epoch_loss_eval = []
        pbar = tqdm(valdataloader)
        pbar.set_description("[Eval Epoch {}]".format(epoch))

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(pbar):
                input_ids = batch_data["input_ids"].to(device)
                token_type_ids = batch_data["token_type_ids"].to(device)
                attention_mask = batch_data["attention_mask"].to(device)
                outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask)
                loss = calculate_loss(outputs, input_ids, token_type_ids, SUMMARY_ID)
                epoch_loss_eval.append(loss.item())

        valid_loss = np.mean(epoch_loss_eval)
        loss_vals_eval.append(valid_loss)

        torch.cuda.empty_cache()

    l1, = plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), loss_vals)
    l2, = plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), loss_vals_eval)
    plt.legend(handles=[l1, l2], labels=['Train loss', 'Eval loss'], loc='best')

    print("模型训练完毕！")
    return model
