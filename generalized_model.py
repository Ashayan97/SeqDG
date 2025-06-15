import torch
from torch import nn
from embeddings import FeatureEmbedding
from vanilla_transformers import TransformerEncoder, TransformerEncoderLayer
from TRNmodule import RelationModule


class final_model(nn.Module):
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
                 ):

        super(final_model, self).__init__()
        self.aggregation = None
        self.representation_layer1 = None
        self.dropout_layer = None
        self.representation_layer2 = None
        self.relu = None
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
        print("Building Transformer with {}-D, {} heads, and {} layers".format(self.d_model,
                                                                               self.nhead,
                                                                               self.num_layers))
        self._create_model()

    def _create_model(self):
        # self.dropout_layer = nn.Dropout(self.dropout)
        self.feature_embedding = FeatureEmbedding(self.seq_len,
                                                  1,
                                                  self.d_model,
                                                  self.d_model,
                                                  False)

        encoder_layer = TransformerEncoderLayer(d_model=self.d_model,
                                                nhead=self.nhead,
                                                dim_feedforward=self.dim_feedforward,
                                                dropout=self.dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Classifier
        self.fc_verb = nn.Linear(self.d_model, self.num_class[0])
        self.fc_noun = nn.Linear(self.d_model, self.num_class[1])



    def set_representation(self, aggregation):#, representation_layer1, dropout_layer, representation_layer2, relu):
        self.aggregation = aggregation
        # self.representation_layer1 = representation_layer1
        # self.dropout_layer = dropout_layer
        # self.representation_layer2 = representation_layer2
        # self.relu = relu


    def representation(self, inputs):
        x = self.aggregation(inputs)
        # x = self.dropout_layer(x)
        # x = self.representation_layer1(x)
        # x = self.relu(x)
        # x = self.representation_layer2(x)
        return x


    def forward(self, inputs):
        with torch.no_grad():
            x_out = self.representation(inputs.view(inputs.size(0), self.seq_len, self.num_clips, -1))

        x_out = self.feature_embedding(x_out)
        x_out, _ = self.transformer_encoder(x_out)
        x_out = x_out.transpose(0, 1).contiguous()


        # classifier output
        y_classifier_noun = self.fc_noun(x_out[:, -2, :])
        y_classifier_verb = self.fc_verb(x_out[:, -1, :])

        output_classifier = (y_classifier_verb, y_classifier_noun)

        return output_classifier
