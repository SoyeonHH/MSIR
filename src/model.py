from ast import Sub
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.activations import gelu, gelu_new
from transformers.models.bert.configuration_bert import BertConfig

from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler, BertModel
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from encoder import *

# class TextOnly(nn.Module):
#     def __init__(self, hp):
#         super().__init__()
#         self.hp = hp
#         hp.d_tout = hp.d_tin
#         self.text_in, self.text_out = hp.d_tin, hp.d_tout
#         self.hidden = hp.pretrain_emb
#         self.post_fusion_dim = hp.d_tfn
#         self.post_fusion_prob = hp.dropout_prj

#         self.text_enc = LanguageEmbeddingLayer(hp)

#         # define the post_fusion layers
#         self.post_fusion_dropout = nn.Dropout(p=hp.dropout_prob)
#         self.post_fusion_layer_1 = nn.Linear(self.hidden + 1, self.post_fusion_dim)
#         self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
#         self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)

#         # in TFN we are doing a regression with constrained output range: (-3, 3), hence we'll apply sigmoid to output
#         # shrink it to (0, 1), and scale\shift it back to range (-3, 3)
#         self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
#         self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
    
#     def forward(self, sentences, bert_sent, bert_sent_type, bert_sent_mask):
#         enc_word = self.text_enc(sentences, bert_sent, bert_sent_type, bert_sent_mask) # (batch_size, seq_len, emb_size)
#         text_h = enc_word[:,0,:] # (batch_size, emb_size)
#         batch_size = text_h.shape[0]

#         if text_h.is_cuda:
#             DTYPE = torch.cuda.FloatTensor
#         else:
#             DTYPE = torch.FloatTensor

#         _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_h), dim=1)
#         # fusion_tensor = torch.bmm(_text_h.unsqueeze(2), _text_h.unsqueeze(1))
#         # fusion_tensor = fusion_tensor.view(-1, (self.hidden + 1) * (self.hidden + 1), 1)
#         # fusion_tensor = torch.bmm(fusion_tensor, _text_h).unsqueeze(1).view(batch_size, -1)
#         fusion_tensor = _text_h.unsqueeze(1).view(batch_size, -1)

#         post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
#         post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped))
#         post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1))
#         post_fusion_y_3 = F.sigmoid(self.post_fusion_layer_3(post_fusion_y_2))
#         output = post_fusion_y_3 * self.output_range + self.output_shift

#         return output, text_h, fusion_tensor

# class AcousticOnly(nn.Module):
#     def __init__(self, hp):
#         super().__init__()
#         self.hp = hp
#         self.audio_in = hp.d_ain
#         self.audio_hidden = hp.d_ah
#         self.post_fusion_dim = hp.d_ah
#         self.audio_prob = hp.dropout_a
#         self.post_fusion_prob = hp.dropout_prj

#         self.audio_enc = RNNEncoder(
#             in_size=hp.d_ain,
#             hidden_size=hp.d_ah,
#             out_size=hp.d_aout,
#             num_layers=hp.n_layer,
#             dropout=hp.dropout_a if hp.n_layer > 1 else 0.0,
#             bidirectional=hp.bidirectional
#         )
#         self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
#         self.post_fusion_layer_1 = nn.Linear(self.audio_hidden + 1, self.post_fusion_dim)
#         self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
#         self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)

#         self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
#         self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
    
#     def forward(self, audio_x, a_len):
#         audio_h = self.audio_enc(audio_x, a_len)
#         batch_size = audio_h.data.shape[0]
#         if audio_h.is_cuda:
#             DTYPE = torch.cuda.FloatTensor
#         else:
#             DTYPE = torch.FloatTensor

#         _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_h), dim=1)
#         # fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _audio_h.unsqueeze(1))
#         # fusion_tensor = fusion_tensor.view(-1, (self.audio_hidden + 1) * (self.audio_hidden + 1), 1)
#         # fusion_tensor = torch.bmm(fusion_tensor, _audio_h.unsqueeze(1)).view(batch_size, -1)
#         fusion_tensor = _audio_h.unsqueeze(1).view(batch_size, -1)

#         post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
#         post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped))
#         post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1))
#         post_fusion_y_3 = F.sigmoid(self.post_fusion_layer_3(post_fusion_y_2))
#         output = post_fusion_y_3 * self.output_range + self.output_shift

#         return output, audio_h, fusion_tensor

# class VisualOnly(nn.Module):
#     def __init__(self, hp):
#         super().__init__()
#         self.hp = hp
#         self.video_in = hp.d_vin
#         self.video_hidden = hp.d_vh
#         self.post_fusion_dim = hp.d_vh
#         self.video_prob = hp.dropout_v
#         self.post_fusion_prob = hp.dropout_prj

#         self.video_enc = RNNEncoder(
#             in_size = hp.d_vin,
#             hidden_size = hp.d_vh,
#             out_size = hp.d_vout,
#             num_layers = hp.n_layer,
#             dropout = hp.dropout_v if hp.n_layer > 1 else 0.0,
#             bidirectional = hp.bidirectional
#         )
#         self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
#         self.post_fusion_layer_1 = nn.Linear(self.video_hidden + 1, self.post_fusion_dim)
#         self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
#         self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)

#         self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
#         self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
    
#     def forward(self, video_x, v_len):
#         video_h = self.video_enc(video_x, v_len)
#         batch_size = video_h.data.shape[0]
#         if video_h.is_cuda:
#             DTYPE = torch.cuda.FloatTensor
#         else:
#             DTYPE = torch.FloatTensor

#         _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_h), dim=1)
#         # fusion_tensor = torch.bmm(_video_h.unsqueeze(2), _video_h.unsqueeze(1))
#         # fusion_tensor = fusion_tensor.view(-1, (self.video_hidden + 1) * (self.video_hidden + 1), 1)
#         # fusion_tensor = torch.bmm(fusion_tensor, _video_h.unsqueeze(1)).view(batch_size, -1)
#         fusion_tensor = _video_h.unsqueeze(1).view(batch_size, -1)

#         post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
#         post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped))
#         post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1))
#         post_fusion_y_3 = F.sigmoid(self.post_fusion_layer_3(post_fusion_y_2))
#         output = post_fusion_y_3 * self.output_range + self.output_shift

#         return output, video_h, fusion_tensor

class TFN(nn.Module):
    '''
    Implements the Tensor Fusion Networks for multimodal sentiment analysis as is described in:
    Zadeh, Amir, et al. "Tensor fusion network for multimodal sentiment analysis." EMNLP 2017 Oral.
    '''
    def __init__(self, hp):
        super().__init__()
        self.hp = hp

        self.text_out = hp.d_tout
        self.audio_hidden = hp.d_ah
        self.video_hidden = hp.d_vh
        self.post_fusion_dim = hp.d_tfn
        self.post_fusion_prob = hp.dropout_prj

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.audio_hidden + 1) * (self.video_hidden  + 1), self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)

        # in TFN we are doing a regression with constrained output range: (-3, 3), hence we'll apply sigmoid to output
        # shrink it to (0, 1), and scale\shift it back to range (-3, 3)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
    
    def forward(self, audio_h, video_h, text_h):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        batch_size = audio_h.data.shape[0]

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_h), dim=1)
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_h), dim=1)
        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_h), dim=1)

        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))
        fusion_tensor = fusion_tensor.view(-1, (self.audio_hidden + 1) * (self.video_hidden + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, _text_h.unsqueeze(1)).view(batch_size, -1)

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped))
        post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1))
        post_fusion_y_3 = torch.sigmoid(self.post_fusion_layer_3(post_fusion_y_2))
        output = post_fusion_y_3 * self.output_range + self.output_shift

        return output, fusion_tensor

class MMIM(nn.Module):
    def __init__(self, hp):
        """Construct MultiMoldal InfoMax model.
        Reference:
            https://github.com/declare-lab/Multimodal-Infomax
        Args: 
            hp (dict): a dict stores training and model configurations
        """
        # Base Encoders
        super().__init__()
        self.hp = hp
        self.add_va = hp.add_va
        hp.d_tout = hp.d_tin

        self.text_enc = LanguageEmbeddingLayer(hp)
        self.visual_enc = RNNEncoder(
            in_size = hp.d_vin,
            hidden_size = hp.d_vh,
            out_size = hp.d_vout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )
        self.acoustic_enc = RNNEncoder(
            in_size = hp.d_ain,
            hidden_size = hp.d_ah,
            out_size = hp.d_aout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )

        # For MI maximization
        self.mi_tv = MMILB(
            x_size = hp.d_tout,
            y_size = hp.d_vout,
            mid_activation = hp.mmilb_mid_activation,
            last_activation = hp.mmilb_last_activation
        )

        self.mi_ta = MMILB(
            x_size = hp.d_tout,
            y_size = hp.d_aout,
            mid_activation = hp.mmilb_mid_activation,
            last_activation = hp.mmilb_last_activation
        )

        if hp.add_va:
            self.mi_va = MMILB(
                x_size = hp.d_vout,
                y_size = hp.d_aout,
                mid_activation = hp.mmilb_mid_activation,
                last_activation = hp.mmilb_last_activation
            )

        dim_sum = hp.d_aout + hp.d_vout + hp.d_tout

        # CPC MI bound
        self.cpc_zt = CPC(
            x_size = hp.d_tout, # to be predicted
            y_size = hp.d_prjh,
            n_layers = hp.cpc_layers,
            activation = hp.cpc_activation
        )
        self.cpc_zv = CPC(
            x_size = hp.d_vout,
            y_size = hp.d_prjh,
            n_layers = hp.cpc_layers,
            activation = hp.cpc_activation
        )
        self.cpc_za = CPC(
            x_size = hp.d_aout,
            y_size = hp.d_prjh,
            n_layers = hp.cpc_layers,
            activation = hp.cpc_activation
        )

        # Trimodal Settings
        self.fusion_prj = SubNet(
            in_size = dim_sum,
            hidden_size = hp.d_prjh,
            n_class = hp.n_class,
            dropout = hp.dropout_prj
        )
            
    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, y=None, mem=None):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        For Bert input, the length of text is "seq_len + 2"
        """
        enc_word = self.text_enc(sentences, bert_sent, bert_sent_type, bert_sent_mask) # (batch_size, seq_len, emb_size)
        text = enc_word[:,0,:] # (batch_size, emb_size)

        acoustic = self.acoustic_enc(acoustic, a_len)
        visual = self.visual_enc(visual, v_len)

        if y is not None:
            lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual, labels=y, mem=mem['tv'])
            lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic, labels=y, mem=mem['ta'])
            # for ablation use
            if self.add_va:
                lld_va, va_pn, H_va = self.mi_va(x=visual, y=acoustic, labels=y, mem=mem['va'])
        else:
            lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual)
            lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic)
            if self.add_va:
                lld_va, va_pn, H_va = self.mi_va(x=visual, y=acoustic)


        # Linear proj and pred
        fusion, preds = self.fusion_prj(torch.cat([text, acoustic, visual], dim=1))

        nce_t = self.cpc_zt(text, fusion)
        nce_v = self.cpc_zv(visual, fusion)
        nce_a = self.cpc_za(acoustic, fusion)
        
        nce = nce_t + nce_v + nce_a

        pn_dic = {'tv':tv_pn, 'ta':ta_pn, 'va': va_pn if self.add_va else None}
        lld = lld_tv + lld_ta + (lld_va if self.add_va else 0.0)
        H = H_tv + H_ta + (H_va if self.add_va else 0.0)

        return lld, nce, preds, pn_dic, H
    
"""
MAG_Bert reference: https://github.com/WasifurRahman/BERT_multimodal_transformer
"""

def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {
    "gelu": gelu,
    "relu": torch.nn.functional.relu,
    "gelu_new": gelu_new,
    "mish": mish,
}


BertLayerNorm = torch.nn.LayerNorm

class MAG_BertModel(BertPreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.MAG = MAG(
            config.hidden_size,
            multimodal_config.beta_shift,
            multimodal_config.dropout_prob,
        )

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        # Early fusion with MAG
        fused_embedding = self.MAG(embedding_output, visual, acoustic)

        encoder_outputs = self.encoder(
            fused_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs, fused_embedding


class MAG_BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert, self.H = MAG_BertModel(config, multimodal_config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        outputs = self.bert(
            input_ids,
            visual,
            acoustic,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[
            2:
        ]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs, self.H