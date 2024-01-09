import torch

def mean_pooling(model_output, attention_mask):
  """
  Perform mean pooling.
  :param model_output:    model output last hidden state.
  :param attention_mask:  attention mask.
  """
  model_output = torch.from_numpy(model_output)
  attention_mask = torch.from_numpy(attention_mask)
  token_embeddings = model_output #First element of model_output contains all token embeddings
  input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
  return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
