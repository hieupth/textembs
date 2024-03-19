import argparse
import numpy as np
from enum import Enum
from pathlib import Path
from pydantic import BaseModel
from typing import Optional
from tokenizers import Tokenizer as HFTokenizer


class Proto(Enum):
  """
  Define available tokenizer protocols.
  """
  HF = 'hf'       # hugging-face tokenizer.
  LOCAL = 'local' # local file tokenizer.


class Tokenizer:
  """
  This class wraps native tokenizer into a simple callable class.
  """

  class Config(BaseModel):
    """
    Tokenizer configuration.
    """
    proto: Proto = Proto.HF       # tokenizer protocol.
    model: str = 'bert'           # model name or path.
    token: Optional[str] = None   # authentication token if necessary.

  def __init__(self, config: Config = Config(), *args, **kwds) -> None:
    """
    Class constructor.
    :param model: model name or path.
    :param token: authentication token.
    :param args:  additional arguments.
    :param kwds:  additional keyword arguments.
    """
    if config.proto == Proto.HF:
      self.tokenizer = HFTokenizer.from_pretrained(config.model, auth_token=config.token)
    else:
      uri = Path(config.model).expanduser().resolve()
      uri = uri.joinpath('tokenizer_config.json') if uri.is_dir() else uri
      self.tokenizer = HFTokenizer.from_file(str(uri))
    self.tokenizer.enable_padding()

  def __call__(self, inputs: list[str], type = np.int64, *args, **kwds) -> dict:
    """
    Encode inputs.
    :param inputs:  input strings.
    :param args:    additional arguments.
    :param kwds:    additional keyword arguments.
    """
    x = self.tokenizer.encode_batch(inputs, False, False)
    # Create result dict.
    z = {'token_ids': list(), 'token_type_ids': list(), 'attention_mask': list()}
    # Fill result dict.
    for i in x:
      z['token_ids'].append(i.ids)
      z['token_type_ids'].append(i.type_ids)
      z['attention_mask'].append(i.attention_mask)
    # Return result.
    return {
      'token_ids': np.array(z['token_ids'], dtype=type),
      'token_type_ids': np.array(z['token_type_ids'], dtype=type),
      'attention_mask': np.array(z['attention_mask'], dtype=type)
    }


# Testing.
if __name__ == '__main__':
  # Define args parser.
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--proto', default='hf')
  arg_parser.add_argument('--model', default='vinai/phobert-base-v2')
  arg_parser.add_argument('--token', default=None)
  arg_parser.add_argument('--test', default='testcases/vi.txt')
  # Parse args.
  args = arg_parser.parse_args()
  # Tokenizer
  config = {'proto': args.proto, 'model': args.model, 'token': args.token}
  tokenizer = Tokenizer(Tokenizer.Config(**config))
  # Test
  with open(args.test) as file:
    test_cases = file.readlines()
  tokenized = tokenizer(test_cases)
  # Result
  print({'meta': {k: {'shape': v.shape} for k, v in tokenized.items()}, 'data': tokenized})
