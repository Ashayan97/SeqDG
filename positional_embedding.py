# import torch
# import torch.nn as nn
#
#
# # TODO Fix it, it is a raw implementation of chatgpt, yes I am smart but I need idea :)
# class PositionalEmbeddingDecoder(nn.Module):
#     def __init__(self, seq_len, visual_input_dim):
#         super(PositionalEmbeddingDecoder, self).__init__()
#         self.window = seq_len
#         self.dim = visual_input_dim
#
#         # Creating the positional embeddings for the maximum sequence length
#         self.positional_embeddings = nn.Embedding(max_sequence_length, embedding_dim)
#
#     def forward(self, masked_input, total_input):
#         """
#         masked_input: Tensor of shape (batch_size, embedding_dim)
#         total_input: Tensor of shape (batch_size, sequence_length, embedding_dim)
#         """
#
#         # Get the batch size
#         batch_size = masked_input.size(0)
#
#         # Get the sequence length
#         sequence_length = total_input.size(1)
#
#         # Expand the positional embeddings to match the total input sequence length
#         positional_embeddings = self.positional_embeddings(
#             torch.arange(sequence_length, device=total_input.device)).unsqueeze(0).expand(batch_size, -1, -1)
#
#         # Add positional embeddings to the masked input
#         masked_input_with_positional = masked_input.unsqueeze(1) + positional_embeddings
#
#         # Process the rest of the decoder with the total input
#         # ... Your decoder logic here ...
#
#         return masked_input_with_positional
