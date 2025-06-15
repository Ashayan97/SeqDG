import torch
from torch import nn
from embeddings import FeatureEmbedding
from video_encoder import video_encoder
from vanilla_transformers import TransformerDecoderLayer, TransformerDecoder, TransformerEncoderLayer, \
    TransformerEncoder
from torch_geometric.nn import PositionalEncoding


class General_Encoder_Decoder(nn.Module):
    def __init__(self,
                 num_class,
                 seq_len=5,
                 num_clips=10,
                 visual_input_dim=2304,
                 d_model=768,
                 dim_feedforward=2048,
                 nhead=8,
                 num_layers=6,
                 dropout=0.1,
                 classification_mode='center',
                 dataset='epic',
                 modalities = ['rgb']):
        super(General_Encoder_Decoder, self).__init__()
        self.num_class = num_class
        self.seq_len = seq_len
        self.num_clips = num_clips
        self.visual_input_dim = visual_input_dim
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.num_layers = num_layers
        self.pos = torch.tensor([i for i in range(self.seq_len)])
        self.dropout = dropout
        self.classification_mode = classification_mode
        self.dataset = dataset
        self.modalities = modalities
        print("Building Transformer with {}-D, {} heads, and {} layers".format(self.d_model,
                                                                               self.nhead,
                                                                               self.num_layers))
        self._create_model()

    def _create_model(self):
        self.video_encoder = video_encoder(self.num_class,
                                           seq_len=self.seq_len,
                                           num_clips=self.num_clips,
                                           visual_input_dim=self.visual_input_dim,
                                           d_model=self.d_model,
                                           dim_feedforward=self.dim_feedforward,
                                           nhead=self.nhead,
                                           num_layers=self.num_layers,
                                           dropout=self.dropout,
                                           data_set=self.dataset,
                                           modalities=self.modalities,
                                           )
        self.video_encoder.cuda()
        # Video decoder to reconstruct the video features
        self.video_decoder_layer = TransformerDecoderLayer(d_model=self.d_model, nhead=self.nhead)

        self.decoder_pose_video = PositionalEncoding(self.d_model)

        # self.decoder_pose_video = FeatureEmbedding(self.seq_len,
        #                                            1,
        #                                            768,
        #                                            768,
        #                                            True)

        self.video_decoder = TransformerDecoder(self.video_decoder_layer, num_layers=self.num_layers)

        self.video_decoder_linear_layer = nn.Linear(self.d_model, self.d_model)
        # Text encoder to provide embeddings space it is pre trained
        self.text_position_feature = FeatureEmbedding(self.seq_len,
                                                      1,
                                                      768,
                                                      768,
                                                      True)

        self.text_encoder_layer = TransformerEncoderLayer(d_model=768,
                                                          nhead=self.nhead,
                                                          dim_feedforward=self.dim_feedforward,
                                                          dropout=self.dropout)

        self.text_encoder = TransformerEncoder(self.text_encoder_layer, num_layers=self.num_layers)

        # Text decoder to reconstruct tokens of action
        self.decoder_pose_text = PositionalEncoding(768)

        self.text_decoder_layer = TransformerDecoderLayer(d_model=768, nhead=self.nhead)

        self.text_decoder = TransformerDecoder(self.text_decoder_layer, num_layers=self.num_layers)

        # self.dropout_layer = nn.Dropout(self.dropout)

        # Classifier
        if self.dataset == 'epic':
            self.text_decoder_layer_prediction = nn.Linear(768, self.num_class[0] + self.num_class[1])

            self.fc_verb = nn.Linear(self.d_model, self.num_class[0])
            self.fc_noun = nn.Linear(self.d_model, self.num_class[1])
        else:
            self.text_decoder_layer_prediction = nn.Linear(768, self.num_class)
            self.action = nn.Linear(self.d_model, self.num_class)

    # def set_encoder(self, feature_embedding, transformer_encoder):
    #     self.video_encoder = video_encoder(self.num_class,
    #                                        seq_len=self.seq_len,
    #                                        num_clips=self.num_clips,
    #                                        visual_input_dim=self.visual_input_dim,
    #                                        d_model=self.d_model,
    #                                        dim_feedforward=self.dim_feedforward,
    #                                        nhead=self.nhead,
    #                                        num_layers=self.num_layers,
    #                                        dropout=self.dropout)
    #     self.video_encoder.set_encoder(transformer_encoder, feature_embedding)

    def masking(self, feature_vector, mask_location):
        f = feature_vector.clone()
        f[:, mask_location, :] = f[:, mask_location, :] * 0  # [batch size][window size][feature_size]
        return f

    def visual_masking(self, feature_vector, mask_location):
        f = torch.zeros(feature_vector.shape)
        for i in range(f.shape[1]):
            if i != mask_location:
                f[:, i, :] = feature_vector[:, i, :]


    def forward(self, visual_inputs, text_inputs):
        # videos features
        if self.training:
            # video_features
            x_video = self.video_encoder(visual_inputs)
            x_video = x_video.transpose(0, 1).contiguous()

            # text features
            x_text = text_inputs

            # classifier output
            if self.dataset == 'epic':
                range_index = -2
                if self.classification_mode == 'all':
                    output_verb_av = self.fc_verb(x_video[:, :-2, :]).transpose(1, 2).contiguous()
                    output_noun_av = self.fc_noun(x_video[:, :-2, :]).transpose(1, 2).contiguous()
                    output_verb_ve = self.fc_verb(x_video[:, -2, :]).unsqueeze(2)
                    output_noun_no = self.fc_noun(x_video[:, -1, :]).unsqueeze(2)
                    y_classifier_verb = torch.cat([output_verb_av, output_verb_ve], dim=2)
                    y_classifier_noun = torch.cat([output_noun_av, output_noun_no], dim=2)
                else:
                    y_classifier_noun = self.fc_noun(x_video[:, -2, :])
                    y_classifier_verb = self.fc_verb(x_video[:, -1, :])
                output_classifier = (y_classifier_verb, y_classifier_noun)

            else:
                range_index = -1
                if self.classification_mode == 'all':
                    output = self.action(x_video).transpose(1, 2).contiguous()
                else:
                    output = self.action(x_video[:, -1, :])
                output_classifier = output

            # video decoder output
            # import pdb;
            # pdb.set_trace()
            # with torch.no_grad():
            x_video_masked = self.masking(x_video[:, :range_index, :], self.seq_len // 2)

            x_video_masked = x_video_masked.cuda() + self.decoder_pose_video(self.pos.cuda())
            # x_video_masked = self.decoder_pose_video(x_video_masked.cuda())
            # x_video_masked = x_video_masked.transpose(0, 1).contiguous()
            y_video = self.video_decoder(x_video_masked, x_text)

            # text output decoder
            # with torch.no_grad():
            x_text_masked = self.masking(x_text, self.seq_len // 2)
            x_text_masked = x_text_masked.cuda() + self.decoder_pose_text(self.pos.cuda())
            y_text = self.text_decoder(x_text_masked, x_video[:, :range_index, :])
            y_text = self.text_decoder_layer_prediction(y_text)

            return y_text[:, self.seq_len // 2, :], y_video[:, self.seq_len // 2, :], output_classifier

        else:
            if self.dataset == 'epic':
                x_video = self.video_encoder(visual_inputs)
                x_video = x_video.transpose(0, 1).contiguous()
                y_classifier_noun = self.fc_noun(x_video[:, -2, :])
                y_classifier_verb = self.fc_verb(x_video[:, -1, :])
                output_classifier = (y_classifier_verb, y_classifier_noun)
                return output_classifier
            else:
                x_video = self.video_encoder(visual_inputs)
                x_video = x_video.transpose(0, 1).contiguous()
                output_action = self.action(x_video[:, -1, :])
                return output_action
