import torch
from torch import nn
from torch.nn.init import normal_


class FeatureEmbedding(nn.Module):
    def __init__(self, seq_len, num_clips, visual_input_dim, d_model, embed_actions):
        super(FeatureEmbedding, self).__init__()
        self.seq_len = seq_len
        self.num_clips = num_clips
        self.visual_input_dim = visual_input_dim
        self.visual_projection = nn.Linear(visual_input_dim, d_model)
        self.visual_relu = nn.ReLU()
        self.num_cls_embeddings = 1 if embed_actions else 2
        self.positional_embedding = nn.Parameter(torch.empty((1, seq_len + self.num_cls_embeddings, d_model), requires_grad=True))
        normal_(self.positional_embedding, std=0.001)
        # When there is no audio (EGTEA), there is no need for modality embeddings
        # as there are only visual inputs, so there is no need for discrimination
        # between visual/audio inputs.
        if not embed_actions:
            self.verb_embedding = nn.Parameter(torch.empty((1, 1, d_model), requires_grad=True))
            normal_(self.verb_embedding, std=0.001)
            self.noun_embedding = nn.Parameter(torch.empty((1, 1, d_model), requires_grad=True))
            normal_(self.noun_embedding, std=0.001)
        else:
            self.action_embedding = nn.Parameter(torch.empty((1, 1, d_model), requires_grad=True))
            normal_(self.action_embedding, std=0.001)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout_v = nn.Dropout(p=0.5)
        self.dropout_a = nn.Dropout(p=0.5)
        self.embed_actions = embed_actions

    def forward(self, inputs):
        # Project audio and visual features to a lower dim
        vis_embed = self.dropout_v(inputs[:, :self.seq_len * self.num_clips, :self.visual_input_dim])
        vis_embed = self.visual_projection(vis_embed)
        vis_embed = self.visual_relu(vis_embed)

        # Tag visual inputs with positional embeddings
        vis_embed = vis_embed + \
                    self.positional_embedding[:, :-self.num_cls_embeddings, :].repeat_interleave(self.num_clips, dim=1)
        if not self.embed_actions:
            # Tag verb/noun embeddings with positional embeddings
            verb_embed = self.verb_embedding + self.positional_embedding[:, -2, :]
            noun_embed = self.noun_embedding + self.positional_embedding[:, -1, :]
            verb_embed = verb_embed.expand(vis_embed.shape[0], -1, -1)
            noun_embed = noun_embed.expand(vis_embed.shape[0], -1, -1)
        else:
            # Tag action embedding with positional embeddings
            action_embed = self.action_embedding + self.positional_embedding[:, -1, :]
            action_embed = action_embed.expand(vis_embed.shape[0], -1, -1)

        if not self.embed_actions:
            seq = torch.cat([vis_embed, verb_embed, noun_embed], 1)
        else:
            seq = torch.cat([vis_embed, action_embed], 1)
        seq = self.dropout(seq)
        seq = seq.transpose(0, 1).contiguous()
        return seq
