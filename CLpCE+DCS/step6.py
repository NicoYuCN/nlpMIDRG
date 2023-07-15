from torch.nn import CrossEntropyLoss

val_fct = CrossEntropyLoss(reduction='none')


import torch
from torch import nn
import torch.nn.functional as F
import random


# ========== batch version ========= #

def getidx(x, topp=3, r=0.01):

    x = x.sort(dim=-1, descending=True)

    current = 1

    result = []

    max_p = x[0][0]
    result.append(x[1][0])

    for pos, p in enumerate(x[0][1:]):
        if current == topp:
            break
        if max_p - p < r:
            result.append(x[1][pos + 1])
            current += 1

    idx = random.randint(0, current - 1)

    return result[idx], current


# ========== batch version ========= #
def ranking_fast(context_hidden, next_hidden, next_top_k_probs, alpha, beam_width, number):
    '''
        context_hidden: bsz*beam x seqlen x embed_dim
        next_hidden: bsz*beam x 1 x embed_dim
        next_top_k_probs: bsz x beam
    '''
    _, context_len, embed_dim = context_hidden.size()
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1, 2)).squeeze(-1)  # [B*K, S]
    scores, _ = torch.max(cosine_matrix, dim=-1)  # [B*K]
    next_top_k_probs = next_top_k_probs.view(-1)  # [B*K]
    scores = (1.0 - alpha) * next_top_k_probs - alpha * scores
    scores = torch.stack(torch.split(scores, beam_width))  # [B, K]

    result, current = getidx(scores[0], topp=number)
    # selected_idx = scores.max(dim=-1)[1]

    return result, current


def ContrastiveDecodingOneStepFast(
        model,
        ids,
        beam_width,
        alpha,
        past_key_values,
        last_hidden_states,
        vocab,
        logit_for_next_step,
        first_step=False,
        number=3
):
    # input_ids: [B, S]
    if first_step:
        output = model(
            input_ids=ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True
        )
        past_key_values = output.past_key_values
        last_hidden_states = output.hidden_states[-1]  # [B, S, E]
        logit_for_next_step = output.logits[:, -1, :]  # [B, V]

    bsz, seqlen, embed_dim = last_hidden_states.size()
    p = random.uniform(0, 1)

    next_probs = F.softmax(logit_for_next_step, dim=-1)
    _, top_k_ids = torch.topk(logit_for_next_step, dim=-1, k=beam_width)  # [B, K]
    top_k_probs = torch.gather(next_probs, dim=1, index=top_k_ids)  # [B, K]
    # compute new hidden

    past_key_values = enlarge_past_key_values(past_key_values, beam_width)
    output = model(
        input_ids=top_k_ids.view(-1, 1),
        attention_mask=torch.ones_like(top_k_ids.view(-1, 1)),
        past_key_values=past_key_values,
        output_hidden_states=True,
        use_cache=True,
    )
    past_key_values = output.past_key_values
    logits = output.logits[:, -1, :]  # [B*K, V]
    next_hidden = output.hidden_states[-1]  # [B*K, 1, E]
    context_hidden = last_hidden_states.unsqueeze(1).expand(-1, beam_width, -1, -1).reshape(bsz * beam_width, seqlen,
                                                                                            embed_dim)  # [B*K, S, E]

    selected_idx, current = ranking_fast(
        context_hidden,
        next_hidden,
        top_k_probs,  # [B, K]
        alpha,
        beam_width,
        number
    )  # [B]
    # prepare for the next step
    next_id = top_k_ids[range(len(top_k_ids)), selected_idx].unsqueeze(-1)  # [B, 1]
    next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), beam_width))  # [B, K, E]
    next_hidden = next_hidden[range(bsz), selected_idx, :]  # [B, E]
    last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)  # [B, S, E]
    past_key_values = select_past_key_values(past_key_values, beam_width, selected_idx)
    logits = torch.stack(torch.split(logits, beam_width))[range(bsz), selected_idx, :]  # [B, V]
    # next_id: [B, 1]

    return next_id, past_key_values, last_hidden_states, logits, current


def enlarge_past_key_values(past_key_values, beam_width):
    # from [B, num_head, seq_len, esz] to [B*K, num_head, seq_len, esz]
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            # item is the key and value matrix
            bsz, num_head, seq_len, esz = item.size()
            item = item.unsqueeze(1).expand(-1, beam_width, -1, -1, -1).reshape(bsz * beam_width, num_head, seq_len,
                                                                                esz)  # [bsz*beam, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values


def select_past_key_values(past_key_values, beam_width, selected_idx):
    '''select_idx: [B]'''
    new_key_values = []
    for layer in past_key_values:
        items = []
        for item in layer:
            bsz_and_beam, num_head, seq_len, esz = item.size()
            bsz = int(bsz_and_beam // beam_width)
            item = torch.stack(torch.split(item, beam_width, dim=0))  # [B, K, num_head, seq_len, esz]
            item = item[range(bsz), selected_idx, :, :, :]  # [B, num_head, seq_len, esz]
            items.append(item)
        new_key_values.append(items)
    return new_key_values


class SimCTGGPT(nn.Module):
    def __init__(self, model):
        super(SimCTGGPT, self).__init__()
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained('data/volab.txt')
        self.model = model

        self.vocab_size = len(self.tokenizer)
        print('The vocabulary size of the language model is {}'.format(len(self.tokenizer)))
        self.embed_dim = self.model.config.hidden_size

    def compute_logits_and_hidden_states(self, input_ids):
        # used for advanced decoding
        # input_ids: 1 x seqlen
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits
        return last_hidden_states, logits

    def forward(self, input_ids, labels):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        last_hidden_states = outputs.hidden_states[-1]
        assert last_hidden_states.size() == torch.Size([bsz, seqlen, self.embed_dim])
        return last_hidden_states, logits

    def eval_loss(self, input_ids, labels):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        mle_loss = val_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        assert mle_loss.size() == torch.Size([bsz * seqlen])
        mask_tmp = labels.masked_fill(~labels.eq(-100), 1.0)
        mask = mask_tmp.masked_fill(mask_tmp.eq(-100), 0.0)
        # sum
        mle_loss_sum = torch.sum(mle_loss)
        token_num_sum = torch.sum(mask)
        return mle_loss_sum, token_num_sum

    def save_model(self, ckpt_save_path):
        import os
        if os.path.exists(ckpt_save_path):
            pass
        else:  # recursively construct directory
            os.makedirs(ckpt_save_path, exist_ok=True)
        # save model
        self.model.save_pretrained(ckpt_save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(ckpt_save_path)

    # decoding functions
    # ------------------------------------------------------- #
    @torch.no_grad()
    def fast_contrastive_search(self, input_ids, beam_width, alpha, decoding_len, number=1,
                                end_of_sequence_token_id=5, early_stop=True):
        '''
           input_ids: prefix input; 1 x prefix_len
           decoding_len: how many tokens to generate
           beam_width: size of candidate pool during decoding
           alpha: regulates importance of model confidence and degeneration penalty
           end_of_sequence_token_id: the token id that denotes the end of generation
           early_stop: whether to use the end_of_sequence_token_id to truncate the output
        '''
        if early_stop:
            try:
                assert end_of_sequence_token_id != None
            except AssertionError:
                raise Exception('When early_stop is True, end_of_sequence_token_id cannot be None!!!')

        self.model.eval()

        # sanity check
        assert alpha >= 0. and alpha <= 1.0

        Tpkk = 0

        # fast mode
        batch_size, seqlen = input_ids.size()
        prefix_len = seqlen
        # generated = [[] for _ in range(batch_size)]
        generated = [item for item in input_ids.tolist()]
        past_key_values = None
        last_hidden_states = None
        logits = None
        for step in range(decoding_len):
            input_ids, past_key_values, last_hidden_states, logits, current = ContrastiveDecodingOneStepFast(
                self.model,
                input_ids,
                beam_width,
                alpha,
                past_key_values,
                last_hidden_states,
                self.tokenizer,
                logits,
                first_step=step == 0,
                number=number
            )
            Tpkk += current
            tokens = input_ids.squeeze(dim=-1).tolist()
            for idx, t in enumerate(tokens):
                generated[idx].append(t)

        output = generated[0]
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if len(tmp) < prefix_len:
                    tmp.append(output[idx])
                else:
                    if output[idx] != end_of_sequence_token_id:
                        tmp.append(output[idx])
                    else:
                        break
            output = tmp
        return output, Tpkk/decoding_len

    def diverse_contrastive_search(self, input_ids, sample_step=10, nucleus_p=0.5, beam_width=5, alpha=0.5,
                                   decoding_len=80,
                                   end_of_sequence_token_id=5, early_stop=True):
        '''
            sample_step:
                number of steps to decode with nucleus sampling,
                for the remaining steps we use contrastive search
            decoding_len:
                the total number of generated tokens
            beam_width:
                size of candidate pool during decoding
            alpha:
                regulates importance of model confidence and degeneration penalty
        '''
        if early_stop:
            try:
                assert end_of_sequence_token_id != None
            except AssertionError:
                raise Exception('When early_stop is True, end_of_sequence_token_id cannot be None!!!')

        contrastive_step = decoding_len - sample_step
        _, prefix_len = input_ids.size()
        # first do sample
        input_ids = self.model.generate(
            input_ids,
            do_sample=True,
            max_length=prefix_len + sample_step,
            top_p=nucleus_p,
            top_k=0)
        # then do contrastive search
        output = self.fast_contrastive_search(input_ids, beam_width, alpha, contrastive_step)
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if len(tmp) < prefix_len:
                    tmp.append(output[idx])
                else:
                    if output[idx] != end_of_sequence_token_id:
                        tmp.append(output[idx])
                    else:
                        break
            output = tmp
        return output

    def greedy_search(self, input_ids, decoding_len=80, end_of_sequence_token_id=5, early_stop=True):
        if early_stop:
            try:
                assert end_of_sequence_token_id != None
            except AssertionError:
                raise Exception('When early_stop is True, end_of_sequence_token_id cannot be None!!!')

        _, prefix_len = input_ids.size()
        output = self.model.generate(
            input_ids,
            max_length=prefix_len + decoding_len)
        output = output[0]
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if len(tmp) < prefix_len:
                    tmp.append(output[idx])
                else:
                    if output[idx] != end_of_sequence_token_id:
                        tmp.append(output[idx])
                    else:
                        break
            output = tmp
        return output

    def beam_search(self, input_ids, beam_width=5, decoding_len=80, end_of_sequence_token_id=5, early_stop=True):
        if early_stop:
            try:
                assert end_of_sequence_token_id != None
            except AssertionError:
                raise Exception('When early_stop is True, end_of_sequence_token_id cannot be None!!!')

        _, prefix_len = input_ids.size()
        output = self.model.generate(
            input_ids,
            max_length=prefix_len + decoding_len,
            num_beams=beam_width)
        output = output[0]
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if len(tmp) < prefix_len:
                    tmp.append(output[idx])
                else:
                    if output[idx] != end_of_sequence_token_id:
                        tmp.append(output[idx])
                    else:
                        break
            output = tmp
        return output

    def nucleus_sampling(self, input_ids, nucleus_p=0.71, decoding_len=80, end_of_sequence_token_id=5, early_stop=True):
        if early_stop:
            try:
                assert end_of_sequence_token_id != None
            except AssertionError:
                raise Exception('When early_stop is True, end_of_sequence_token_id cannot be None!!!')

        _, prefix_len = input_ids.size()
        output = self.model.generate(
            input_ids,
            do_sample=True,
            max_length=prefix_len + decoding_len,
            top_p=nucleus_p,
            top_k=0)
        output = output[0]
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if len(tmp) < prefix_len:
                    tmp.append(output[idx])
                else:
                    if output[idx] != end_of_sequence_token_id:
                        tmp.append(output[idx])
                    else:
                        break
            output = tmp
        return output

    def topk_sampling(self, input_ids, topk=5, decoding_len=80, end_of_sequence_token_id=5, early_stop=True):
        if early_stop:
            try:
                assert end_of_sequence_token_id != None
            except AssertionError:
                raise Exception('When early_stop is True, end_of_sequence_token_id cannot be None!!!')

        _, prefix_len = input_ids.size()
        output = self.model.generate(
            input_ids,
            do_sample=True,
            max_length=prefix_len + decoding_len,
            top_p=1.0,
            top_k=topk)
        output = output[0]
        if early_stop:
            tmp = []
            for idx in range(len(output)):
                if len(tmp) < prefix_len:
                    tmp.append(output[idx])
                else:
                    if output[idx] != end_of_sequence_token_id:
                        tmp.append(output[idx])
                    else:
                        break
            output = tmp
        return output

#
# net = SimCTGGPT(model.gpt)
