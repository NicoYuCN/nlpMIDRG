# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizer
import torch.nn.functional as F
import pandas as pd
from step2 import run

device = "cuda" if torch.cuda.is_available() else 'cpu'
max_len = 230  # 输入模型的最大长度，要比config中n_ctx小
generate_max_len = 80  # 生成摘要的最大长度
repetition_penalty = 1.2
topk = 5
topp = 0.71


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1  # batch size 1 for now
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sumarize(model, content):
    tokenizer = BertTokenizer.from_pretrained('data/volab.txt')
    model.to(device)
    model.eval()

    # 对新闻正文进行预处理，并判断如果超长则进行截断
    content_tokens = tokenizer.tokenize(content)
    if len(content_tokens) > max_len - 3 - generate_max_len:
        content_tokens = content_tokens[:max_len - 3 - generate_max_len]

    # 将tokens索引化，变成模型所需格式
    content_tokens = ['[CLS]'] + content_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(content_tokens)

    # 将input_ids变成tensor
    curr_input_tensor = torch.tensor(input_ids).long().to(device)

    generated = []
    # 最多生成generate_max_len个token
    for _ in range(generate_max_len):
        outputs = model(input_ids=curr_input_tensor, attention_mask=None)
        next_token_logits = outputs[0][-1, :]  # size:[vocab size]
        # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
        for id_ in set(generated):
            next_token_logits[id_] /= repetition_penalty
        # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
        next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=topk, top_p=topp)
        # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
        if next_token.item() == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
            break
        generated.append(next_token.item())
        curr_input_tensor = torch.cat((curr_input_tensor, next_token), dim=0)

    text = tokenizer.convert_ids_to_tokens(generated)
    return text



def Result():
    model = run()

    result = pd.read_csv('data/train.csv', encoding='utf-8')
    result = result[len(result) - 6000:]

    data = []
    label = []

    for i in result['main']:
        data.append(i)

    for j in result['label']:
        label.append(j)

    ref = {}
    gt = {}

    for pos in range(len(data)):

        ls = sumarize(model, data[pos])
        sp = ""
        for y in ls:
            sp = sp + str(y) + " "

        ref.update({str(pos): [sp.strip()]})
        gt.update({str(pos): [label[pos]]})

    print("Over!")

    return model, ref, gt