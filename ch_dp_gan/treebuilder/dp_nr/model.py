# coding: UTF-8
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from config import *
import math


class MaskedGRU(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MaskedGRU, self).__init__()
        self.rnn = nn.GRU(batch_first=True, *args, **kwargs)
        self.hidden_size = self.rnn.hidden_size

    def forward(self, padded, lengths, initial_state=None):
        # [batch*edu]
        zero_mask = lengths != 0
        lengths[lengths == 0] += 1  # in case zero length instance
        _, indices = lengths.sort(descending=True)
        _, rev_indices = indices.sort()

        # [batch*edu, max_word_seqlen, embedding]
        padded_sorted = padded[indices]
        lengths_sorted = lengths[indices]
        padded_packed = pack_padded_sequence(padded_sorted, lengths_sorted, batch_first=True)
        self.rnn.flatten_parameters()
        outputs_sorted_packed, hidden_sorted = self.rnn(padded_packed, initial_state)
        # [batch*edu, max_word_seqlen, ]
        outputs_sorted, _ = pad_packed_sequence(outputs_sorted_packed, batch_first=True)
        # [batch*edu, max_word_seqlen, output_size]
        outputs = outputs_sorted[rev_indices]
        # [batch*edu, output_size]
        hidden = hidden_sorted.transpose(1, 0).contiguous().view(outputs.size(0), -1)[rev_indices]

        outputs = outputs * zero_mask.view(-1, 1, 1).float()
        hidden = hidden * zero_mask.view(-1, 1).float()
        return outputs, hidden


class BiGRUEDUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiGRUEDUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn = MaskedGRU(input_size, hidden_size//2, bidirectional=True)
        self.token_scorer = nn.Linear(hidden_size, 1)
        self.output_size = hidden_size

    def forward(self, inputs, masks):
        lengths = masks.sum(-1)
        outputs, hidden = self.rnn(inputs, lengths)
        token_score = self.token_scorer(outputs).squeeze(-1)
        token_score[masks == 0] = -1e8
        token_score = token_score.softmax(dim=-1) * masks.float()
        weighted_sum = (outputs * token_score.unsqueeze(-1)).sum(-2)
        return hidden + weighted_sum


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size
        self.input_dense = nn.Linear(input_size, hidden_size)
        self.edu_rnn = MaskedGRU(hidden_size, hidden_size//2, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.bound_emb = nn.Embedding(2, BOUND_INFO_SIZE)
        split_in_size = hidden_size if not USE_BOUND else hidden_size + BOUND_INFO_SIZE
        self.split_rnn = MaskedGRU(split_in_size, hidden_size//2, bidirectional=True)
        self.split_rnn_norm = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadAttention(heads=HEADS, input_size=hidden_size, hidden_size=ML_ATT_HIDDEN, dropout=dropout)
        self.attn_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs, masks, edu_boundary):
        inputs = self.input_dense(inputs)
        # edu rnn
        edus, _ = self.edu_rnn(inputs, masks.sum(-1))
        edus = inputs + self.dropout(edus)
        # cnn
        edus = edus.transpose(-2, -1)
        splits = self.conv(edus).transpose(-2, -1)
        masks = torch.cat([(masks.sum(-1, keepdim=True) > 0).type_as(masks), masks], dim=1)
        lengths = masks.sum(-1)

        if USE_BOUND:
            bound_embed = self.bound_emb(edu_boundary)
            splits_bounded = torch.cat((splits, bound_embed), -1)
            outputs, hidden = self.split_rnn(splits_bounded, lengths)
        else:
            # split rnn
            outputs, hidden = self.split_rnn(splits, lengths)

        # outputs = splits + self.dropout(outputs)
        if LAYER_NORM_USE:
            outputs = self.split_rnn_norm(outputs)
        if CONTEXT_ATT:
            attn_masks = masks.unsqueeze(1).expand(masks.size(0), masks.size(1), masks.size(1)) * masks.unsqueeze(-1)
            outputs = outputs + self.attn(outputs, outputs, outputs, attn_masks)
            if LAYER_NORM_USE:
                outputs = self.attn_norm(outputs)
        return outputs, masks, hidden


class Decoder(nn.Module):
    def __init__(self, inputs_size, hidden_size):
        super(Decoder, self).__init__()
        self.input_dense = nn.Linear(inputs_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output_size = hidden_size

    def forward(self, input, state):
        return self.run_step(input, state)

    def run_batch(self, inputs, init_states, masks):
        inputs = self.input_dense(inputs) * masks.unsqueeze(-1).float()
        outputs, _ = self.rnn(inputs, init_states.unsqueeze(0))
        outputs = outputs * masks.unsqueeze(-1).float()
        return outputs

    def run_step(self, input, state):
        input = self.input_dense(input)
        self.rnn.flatten_parameters()
        output, state = self.rnn(input, state)
        return output, state


class BiaffineAttention(nn.Module):
    def __init__(self, encoder_size, decoder_size, num_labels, hidden_size):
        super(BiaffineAttention, self).__init__()
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.e_mlp = nn.Sequential(
            nn.Linear(encoder_size, hidden_size),
            nn.ReLU()
        )
        self.d_mlp = nn.Sequential(
            nn.Linear(decoder_size, hidden_size),
            nn.ReLU()
        )
        self.W_e = nn.Parameter(torch.empty(num_labels, hidden_size, dtype=torch.float))
        self.W_d = nn.Parameter(torch.empty(num_labels, hidden_size, dtype=torch.float))
        self.U = nn.Parameter(torch.empty(num_labels, hidden_size, hidden_size, dtype=torch.float))
        self.b = nn.Parameter(torch.zeros(num_labels, 1, 1, dtype=torch.float))
        nn.init.xavier_normal_(self.W_e)
        nn.init.xavier_normal_(self.W_d)
        nn.init.xavier_normal_(self.U)

    def forward(self, e_outputs, d_outputs):
        # e_outputs [batch, length_encoder, encoder_size]
        # d_outputs [batch, length_decoder, decoder_size]

        # [batch, length_encoder, hidden_size]
        e_outputs = self.e_mlp(e_outputs)
        # [batch, length_encoder, hidden_size]
        d_outputs = self.d_mlp(d_outputs)

        # [batch, num_labels, 1, length_encoder]
        out_e = (self.W_e @ e_outputs.transpose(1, 2)).unsqueeze(2)
        # [batch, num_labels, length_decoder, 1]
        out_d = (self.W_d @ d_outputs.transpose(1, 2)).unsqueeze(3)

        # [batch, 1, length_decoder, hidden_size] @ [num_labels, hidden_size, hidden_size]
        # [batch, num_labels, length_decoder, hidden_size]
        out_u = d_outputs.unsqueeze(1) @ self.U
        # [batch, num_labels, length_decoder, hidden_size] * [batch, 1, hidden_size, length_encoder]
        # [batch, num_labels, length_decoder, length_encoder]
        out_u = out_u @ e_outputs.unsqueeze(1).transpose(2, 3)
        # [batch, length_decoder, length_encoder, num_labels]
        out = (out_e + out_d + out_u + self.b).permute(0, 2, 3, 1)
        return out


class SplitAttention(nn.Module):
    def __init__(self, encoder_size, decoder_size, hidden_size):
        super(SplitAttention, self).__init__()
        self.biaffine = BiaffineAttention(encoder_size, decoder_size, 1, hidden_size)

    def forward(self, e_outputs, d_outputs, masks):
        biaffine = self.biaffine(e_outputs, d_outputs)
        attn = biaffine.squeeze(-1)
        attn[masks == 0] = -1e8
        return attn


class PartitionPtr(nn.Module):
    def __init__(self, hidden_size, dropout, word_vocab, pos_vocab, nuc_label, rel_label,
                 pretrained=None, w2v_size=None, w2v_freeze=False, pos_size=30,
                 split_mlp_size=64, nuc_mlp_size=64, rel_mlp_size=128,
                 use_gpu=False):
        super(PartitionPtr, self).__init__()
        self.use_gpu = use_gpu
        self.word_vocab = word_vocab
        self.pos_vocab = pos_vocab
        self.nuc_label = nuc_label
        self.rel_label = rel_label
        self.word_emb = word_vocab.embedding(pretrained=pretrained, freeze=w2v_freeze, use_gpu=use_gpu)  # , dim=w2v_size
        self.w2v_size = self.word_emb.weight.shape[-1]
        self.pos_emb = pos_vocab.embedding(dim=pos_size, use_gpu=use_gpu)
        self.pos_size = pos_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout

        # component
        self.edu_encoder = BiGRUEDUEncoder(self.w2v_size+self.pos_size, hidden_size)
        self.encoder = Encoder(self.edu_encoder.output_size, hidden_size, dropout)
        self.context_dense = nn.Linear(self.encoder.hidden_size, hidden_size)
        self.decoder = Decoder(self.encoder.output_size*2, hidden_size)
        self.split_attention = SplitAttention(self.encoder.output_size, self.decoder.output_size, split_mlp_size)
        self.nr_classifier = BiaffineAttention(self.encoder.output_size, self.decoder.output_size, NR_LABEL, rel_mlp_size)
        self.nr_linear = nn.Sequential(nn.Linear(NR_LABEL, 1), nn.Sigmoid())
        # Up—sampling
        self.down = nn.Sequential(nn.Conv2d(in_channel_G, out_channel_G, (ker_h_G, ker_w_G), strip_G),
                                  nn.ReLU())
        self.max_p = nn.MaxPool2d(kernel_size=(p_w_G, p_h_G), stride=p_w_G)

    def cnn_feat_ext(self, img):
        out = self.down(img)
        if MAX_POOLING:
            out = self.max_p(out)
        return out

    def forward(self, left, right, memory, state):
        return self.decode(left, right, memory, state)

    def decode(self, left, right, memory, state):
        d_input = torch.cat([memory[0, left], memory[0, right]]).view(1, 1, -1)
        d_output, state = self.decoder(d_input, state)
        masks = torch.zeros(1, 1, memory.size(1), dtype=torch.uint8)
        masks[0, 0, left+1:right] = 1
        if self.use_gpu:
            masks = masks.cuda()
        split_scores = self.split_attention(memory, d_output, masks)
        split_scores = split_scores.softmax(dim=-1)
        nr_score = self.nr_classifier(memory, d_output).softmax(dim=-1) * masks.unsqueeze(-1).float()
        split_scores = split_scores[0, 0].cpu().detach().numpy()
        nr_score = nr_score[0, 0].cpu().detach().numpy()
        return split_scores, nr_score, state

    def encode_edus(self, e_inputs):
        e_input_words, e_input_poses, e_masks, _ = e_inputs
        batch_size, max_edu_seqlen, max_word_seqlen = e_input_words.size()
        # [batch_size, max_edu_seqlen, max_word_seqlen, embedding]
        word_embedd = self.word_emb(e_input_words)
        pos_embedd = self.pos_emb(e_input_poses)
        concat_embedd = torch.cat([word_embedd, pos_embedd], dim=-1) * e_masks.unsqueeze(-1).float()
        # encode edu
        # [batch_size*max_edu_seqlen, max_word_seqlen, embedding]
        inputs = concat_embedd.view(batch_size*max_edu_seqlen, max_word_seqlen, -1)
        # [batch_size*max_edu_seqlen, max_word_seqlen]
        masks = e_masks.view(batch_size*max_edu_seqlen, max_word_seqlen)
        edu_encoded = self.edu_encoder(inputs, masks)
        # [batch_size, max_edu_seqlen, edu_encoder_output_size]
        edu_encoded = edu_encoded.view(batch_size, max_edu_seqlen, self.edu_encoder.output_size)
        e_masks = (e_masks.sum(-1) > 0).int()
        return edu_encoded, e_masks

    def _decode_batch(self, e_outputs, e_contexts, d_inputs):
        d_inputs_indices, d_masks = d_inputs
        d_outputs_masks = (d_masks.sum(-1) > 0).type_as(d_masks)

        d_init_states = self.context_dense(e_contexts)

        d_inputs = e_outputs[torch.arange(e_outputs.size(0)), d_inputs_indices.permute(2, 1, 0)].permute(2, 1, 0, 3)
        d_inputs = d_inputs.contiguous().view(d_inputs.size(0), d_inputs.size(1), -1)
        d_inputs = d_inputs * d_outputs_masks.unsqueeze(-1).float()

        d_outputs = self.decoder.run_batch(d_inputs, d_init_states, d_outputs_masks)
        return d_outputs, d_outputs_masks, d_masks

    def loss(self, e_inputs, d_inputs, grounds, n_epoch):
        e_inputs_, e_masks = self.encode_edus(e_inputs)
        e_outputs, e_outputs_masks, e_contexts = self.encoder(e_inputs_, e_masks, e_inputs[-1])
        d_outputs, d_outputs_masks, d_masks = self._decode_batch(e_outputs, e_contexts, d_inputs)

        splits_ground, nrs_ground, d_g_s, d_g_nr, l_x = grounds
        # split losses
        splits_attn = self.split_attention(e_outputs, d_outputs, d_masks)
        splits_predict_ = splits_attn.log_softmax(dim=2)
        splits_ground_ = splits_ground.view(-1)
        splits_predict = splits_predict_.view(splits_ground_.size(0), -1)
        splits_masks = d_outputs_masks.view(-1).float()
        splits_loss = F.nll_loss(splits_predict, splits_ground_, reduction="none")
        splits_loss = (splits_loss * splits_masks).sum() / splits_masks.sum()

        nr_score = self.nr_classifier(e_outputs, d_outputs)
        nr_score = nr_score.log_softmax(dim=-1) * d_masks.unsqueeze(-1).float()
        nr_score = nr_score.view(nr_score.size(0) * nr_score.size(1), nr_score.size(2), nr_score.size(3))
        target_nr_score = nr_score[torch.arange(nr_score.size(0)), splits_ground_]
        target_nr_ground = nrs_ground.view(-1)
        nr_loss = F.nll_loss(target_nr_score, target_nr_ground)

        # GAN learning
        if n_epoch > WARM_UP_EP:
            # GAN learning for global pure optimization
            # d_g_s, d_g_n, d_g_r, l_x: (batch, max_edu_num - 1)
            batch_, split_num, _ = splits_predict_.size()
            gen_g_s, gen_g_nr = d_g_s.clone(), d_g_nr.clone()
            nr_s = self.nr_linear(target_nr_score)
            nr_idx_ = 0
            for t_idx in range(batch_):
                splits_ = splits_predict_[t_idx]  # (split_num, split_num)
                act_len = min(splits_.size(0), MAX_W)
                split_ground_t = splits_ground[t_idx]
                l_x_ = l_x[t_idx]
                s_num = len(l_x_)
                nr_idx = nr_idx_
                nr_idx_ += split_num  # 分割点递增, 对应 nr 递增
                buffer_idx = []
                for idx in range(s_num):
                    if l_x_[idx] < MAX_H:
                        # S graph
                        if l_x_[idx] in buffer_idx:
                            gen_g_s[t_idx][l_x_[idx]][:act_len] = gen_g_s[t_idx][l_x_[idx]][:act_len] + splits_[idx][:act_len]
                        else:
                            buffer_idx.append(l_x_[idx])
                            gen_g_s[t_idx][l_x_[idx]][:act_len] = splits_[idx][:act_len]

                        # NR graph
                        if split_ground_t[idx] < MAX_W:
                            gen_g_nr[t_idx][l_x_[idx]][split_ground_t[idx]] = nr_s[nr_idx] * NR_LABEL
                    nr_idx += 1
            # 2-channel color
            real_img = torch.cat((d_g_s.unsqueeze(1), d_g_nr.unsqueeze(1)), 1)
            gen_img = torch.cat((gen_g_s.unsqueeze(1), gen_g_nr.unsqueeze(1)), 1)
        else:
            gen_img = real_img = None
        return splits_loss, nr_loss, gen_img, real_img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Up—sampling
        self.down = nn.Sequential(nn.Conv2d(in_channel_G, out_channel_G, (ker_h_G, ker_w_G), strip_G),
                                  nn.LeakyReLU(0.2, inplace=True))
        # max pooling
        self.max_p = nn.MaxPool2d(kernel_size=(p_w_G, p_h_G), stride=p_w_G)

        # Fully-connected layers
        c_h = (MAX_H - (ker_h_G - 1)) // p_w_G if MAX_POOLING else MAX_H - (ker_h_G - 1)
        c_w = (MAX_W - (ker_w_G - 1)) // p_h_G if MAX_POOLING else MAX_W - (ker_w_G - 1)
        down_dim = out_channel_G * c_h * c_w
        # print(down_dim, c_h, c_w)
        self.fc = nn.Sequential(
            nn.Linear(down_dim, down_dim // 2),
            nn.BatchNorm1d(down_dim // 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(down_dim // 2, 1),
            # nn.Sigmoid()
        )

    def cnn_feat_ext(self, img):
        out = self.down(img)
        if MAX_POOLING:
            out = self.max_p(out)
        return out

    def forward(self, out):
        """ (batch, colors, height, width)
            (5, 3, 20, 80)
            16 * 19 * 1 = 304
        """
        # out = self.down(img)
        # if MAX_POOLING:
        #     out = self.max_p(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_k = hidden_size // heads
        self.h = heads
        self.q_linear = nn.Linear(input_size, hidden_size)
        self.v_linear = nn.Linear(input_size, hidden_size)
        self.k_linear = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.d_k * self.h, input_size)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).float()
            scores = scores.masked_fill(mask == 0, -1e8)
        scores = F.softmax(scores, dim=-1)
        if mask is not None:
            scores = scores * mask
        scores = self.dropout(scores)
        attn = scores @ v

        # concatenate heads and put through final linear layer
        concat = attn.transpose(1, 2).contiguous().view(bs, -1, self.d_k * self.h)
        output = self.out(concat)
        if mask is not None:
            mask = (mask.squeeze(1).sum(-1, keepdim=True) > 0).float()
            output = output * mask
        return output
