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

        # define the pre-fusion subnetworks
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
        self.text_enc = RNNEncoder(
            in_size = hp.d_tin,
            hidden_size = hp.d_th,
            out_size = hp.d_tout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_prj if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.audio_hidden + 1) * (self.video_hidden  + 1), self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)

        # in TFN we are doing a regression with constrained output range: (-3, 3), hence we'll apply sigmoid to output
        # shrink it to (0, 1), and scale\shift it back to range (-3, 3)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
    
    def forward(self, audio_x, video_x, text_x, a_len, v_len, t_len):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        audio_h = self.acoustic_enc(audio_x, a_len)
        video_h = self.visual_enc(video_x, v_len)
        text_h = self.text_enc(text_x, t_len)
        batch_size = audio_h.data.shape[0]

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
            device = torch.device('cuda:1')
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE).to(device), requires_grad=False), audio_h), dim=1)
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE).to(device), requires_grad=False), video_h), dim=1)
        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE).to(device), requires_grad=False), text_h), dim=1)

        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))
        fusion_tensor = fusion_tensor.view(-1, (self.audio_hidden + 1) * (self.video_hidden + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, _text_h.unsqueeze(1)).view(batch_size, -1)

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped))
        post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1))
        post_fusion_y_3 = torch.sigmoid(self.post_fusion_layer_3(post_fusion_y_2))
        output = post_fusion_y_3 * self.output_range + self.output_shift

        return output, fusion_tensor


class Text(nn.Module):

    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        # define encoder
        self.text_enc = RNNEncoder(
            in_size = hp.d_tin,
            hidden_size = hp.d_th,
            out_size = hp.d_tout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_prj if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )
        
        # define MLP layers
        self.mlp_dropout = nn.Dropout(p=hp.dropout_prj)
        self.mlp_layer_1 = nn.Linear(hp.d_tout + 1, 128)
        self.mlp_layer_2 = nn.Linear(128, 64)
        self.mlp_layer_3 = nn.Linear(64, 1)

        # in TFN we are doing a regression with constrained output range: (-3, 3), hence we'll apply sigmoid to output
        # shrink it to (0, 1), and scale\shift it back to range (-3, 3)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
    
    def forward(self, text, t_len):
        ''' text: tensor of shape (batch_size, sequence_len, text_in) '''
        text_h = self.text_enc(text, t_len)
        batch_size = text_h.data.shape[0]

        if text_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor
        
        device = torch.device('cuda:1')
        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE).to(device), requires_grad=False), text_h), dim=1)
        tensor = _text_h.unsqueeze(1).view(batch_size, -1)

        mlp_dropped = self.mlp_dropout(tensor)
        mlp_y_1 = F.relu(self.mlp_layer_1(mlp_dropped))
        mlp_y_2 = F.relu(self.mlp_layer_2(mlp_y_1))
        mlp_y_3 = torch.sigmoid(self.mlp_layer_3(mlp_y_2))
        output = mlp_y_3 * self.output_range + self.output_shift

        return output, tensor


class Visual(nn.Module):

    def __init__(self, hp) -> None:
        super().__init__()
        self.hp = hp

        # define encoder
        self.visual_enc = RNNEncoder(
            in_size = hp.d_vin,
            hidden_size = hp.d_vh,
            out_size = hp.d_vout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )

        # define MLP layers
        self.mlp_dropout = nn.Dropout(p=hp.dropout_v)
        self.mlp_layer_1 = nn.Linear(hp.d_vout + 1, 32)
        self.mlp_layer_2 = nn.Linear(32, 32)
        self.mlp_layer_3 = nn.Linear(32, 1)

        # in TFN we are doing a regression with constrained output range: (-3, 3), hence we'll apply sigmoid to output
        # shrink it to (0, 1), and scale\shift it back to range (-3, 3)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
    
    def forward(self, visual, v_len):
        ''' visual: tensor of shape (batch_size, visual_in) '''
        video_h = self.visual_enc(visual, v_len)
        batch_size = video_h.data.shape[0]

        if video_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor
        
        device = torch.device('cuda:1')
        _visual_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE).to(device), requires_grad=False), video_h), dim=1)
        tensor = _visual_h.unsqueeze(1).view(batch_size, -1)

        mlp_dropped = self.mlp_dropout(tensor)
        mlp_y_1 = F.relu(self.mlp_layer_1(mlp_dropped))
        mlp_y_2 = F.relu(self.mlp_layer_2(mlp_y_1))
        mlp_y_3 = torch.sigmoid(self.mlp_layer_3(mlp_y_2))
        output = mlp_y_3 * self.output_range + self.output_shift

        return output, tensor


class Acoustic(nn.Module):

    def __init__(self, hp) -> None:
        super().__init__()
        self.hp = hp

        # define encoder
        self.acoustic_enc = RNNEncoder(
            in_size = hp.d_ain,
            hidden_size = hp.d_ah,
            out_size = hp.d_aout,
            num_layers = hp.n_layer,
            dropout = hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional = hp.bidirectional
        )

        # define MLP layers
        self.mlp_dropout = nn.Dropout(p=hp.dropout_a)
        self.mlp_layer_1 = nn.Linear(hp.d_aout + 1, 32)
        self.mlp_layer_2 = nn.Linear(32, 32)
        self.mlp_layer_3 = nn.Linear(32, 1)

        # in TFN we are doing a regression with constrained output range: (-3, 3), hence we'll apply sigmoid to output
        # shrink it to (0, 1), and scale\shift it back to range (-3, 3)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)
    
    def forward(self, audio, a_len):
        ''' visual: tensor of shape (batch_size, visual_in) '''
        audio_h = self.acoustic_enc(audio, a_len)
        batch_size = audio_h.data.shape[0]

        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor
        
        device = torch.device('cuda:1')
        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE).to(device), requires_grad=False), audio_h), dim=1)
        tensor = _audio_h.unsqueeze(1).view(batch_size, -1)

        mlp_dropped = self.mlp_dropout(tensor)
        mlp_y_1 = F.relu(self.mlp_layer_1(mlp_dropped))
        mlp_y_2 = F.relu(self.mlp_layer_2(mlp_y_1))
        mlp_y_3 = torch.sigmoid(self.mlp_layer_3(mlp_y_2))
        output = mlp_y_3 * self.output_range + self.output_shift

        return output, tensor