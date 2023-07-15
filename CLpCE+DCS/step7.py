import torch
from transformers import BertTokenizer
import pandas as pd


device = "cuda" if torch.cuda.is_available() else 'cpu'
max_len = 230  # 输入模型的最大长度，要比config中n_ctx小
generate_max_len = 80  # 生成摘要的最大长度
repetition_penalty = 1.2
topk = 5
topp = 0.71


def sumarize(model, content, name, beam_width=5, number=1):
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

    curr_input_tensor = curr_input_tensor.reshape(1, curr_input_tensor.shape[0])

    if name == "cs":
        generated, idx = model.fast_contrastive_search(curr_input_tensor, beam_width, 0.5, 80, number=number)
    elif name == "gs":
        generated = model.greedy_search(curr_input_tensor)
    elif name == "bs":
        generated = model.beam_search(curr_input_tensor)
    elif name == "ns":
        generated = model.nucleus_sampling(curr_input_tensor)
    else:
        generated = model.topk_sampling(curr_input_tensor)
    #     generated = model.diverse_contrastive_search(curr_input_tensor)

    text = tokenizer.convert_ids_to_tokens(generated)
    return text


def Result2(net, name, beam_width=5, number=1):
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

    # ipo = []

    for pos in range(len(data)):

        ls = sumarize(net, data[pos], name, beam_width, number)
        sp = ""

        book = False
        for y in ls:

            if book:
                sp = sp + str(y) + " "

            if y == '[SEP]':
                book = True

        ref.update({str(pos): [sp.strip()]})
        gt.update({str(pos): [label[pos]]})

        # ipo.append(op)

    print("Over!")

    # return ref, gt
    # print(ipo)
    return ref, gt
