import torch
from torch import nn
from embeddings import FeatureEmbedding
from vanilla_transformers import TransformerEncoder, TransformerEncoderLayer
from TRNmodule import RelationModule


class video_encoder(nn.Module):
    def __init__(self,
                 num_class,
                 seq_len=5,
                 num_clips=10,
                 visual_input_dim=2304,
                 d_model=512,
                 dim_feedforward=2048,
                 nhead=8,
                 num_layers=6,
                 dropout=0.1,
                 data_set='epic',
                 modalities = ['rgb']
                 ):

        super(video_encoder, self).__init__()
        self.num_class = num_class
        self.seq_len = seq_len
        self.num_clips = num_clips
        self.visual_input_dim = visual_input_dim
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.presnetation_dim = 1024
        self.dataset = data_set
        self.modalities = modalities
        print("Building Transformer with {}-D, {} heads, and {} layers".format(self.d_model,
                                                                               self.nhead,
                                                                               self.num_layers))
        self._create_model()



    def _create_model(self):
        if self.dataset == 'epic':
            self.feature_embedding = FeatureEmbedding(self.seq_len,
                                                      1,
                                                      self.d_model * len(self.modalities),
                                                      self.d_model,
                                                      False).cuda()
        else:
            self.feature_embedding = FeatureEmbedding(self.seq_len,
                                                      1,
                                                      self.d_model,
                                                      self.d_model,
                                                      True)
        self.aggregation = {}
        for modal_idx in self.modalities:
            self.aggregation[modal_idx] = RelationModule(self.visual_input_dim, self.d_model, self.num_clips)
        self.aggregation = nn.ModuleDict(self.aggregation)
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model,
                                                nhead=self.nhead,
                                                dim_feedforward=self.dim_feedforward,
                                                dropout=self.dropout)
        self.dropout_layer1 = nn.Dropout(0.5)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=self.num_layers)


    def representation(self, inputs):
        x = torch.zeros((inputs.size(0), self.seq_len ,self.d_model * len(self.modalities))).cuda()
        for modal_idx in range(len(self.modalities)):
            x[:,:,modal_idx * self.d_model:(modal_idx + 1) * self.d_model ] = self.aggregation[self.modalities[modal_idx]](inputs[:,modal_idx,:,:,:].reshape(inputs.size(0),self.seq_len,self.num_clips,-1))
        
        x = self.feature_embedding(x)
        x, _ = self.transformer_encoder(x)
        return x

    def forward(self, inputs, encode_label=False):

        if encode_label:
            with torch.no_grad():
                encode_space = self.representation(inputs.view(inputs.size(0), len(self.modalities), self.seq_len, self.num_clips, -1))[self.seq_len // 2,:, :]
                encode_space = encode_space.transpose(0, 1).contiguous()
                encode_space = encode_space.reshape(inputs.size(0),1,  -1)
                return encode_space

        else:
            x_out = self.representation(inputs.view(inputs.size(0), len(self.modalities),self.seq_len, self.num_clips, -1))

            return x_out


