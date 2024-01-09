import os
import json
import ovmsclient
import numpy as np
from typing import List
from pydantic import BaseModel
from textembserve.tokenizers import *
from textembserve import wordsegmenters
from textembserve import postprocessing
from fastapi import FastAPI, HTTPException


def load_tokenizers():
  """
  Load tokenizers from config.
  """
  tokenizers = dict()
  # Read config file.
  with open(os.getenv("TOKENIZER_CONF_FILE", "/tokenizer_config.json"), "r") as file:
    conf = json.load(file)
  # Create tokenizer.
  for k, v in conf.items():
    wseg = wordsegmenters.create_word_segmenter(v.get("word_segmenter"))
    tok = Tokenizer(model=v.get("tokenizer_model"), wordsegmenter=wseg)
    tokenizers.update({k: tok})
  return tokenizers

# Make tokenizers.
TOKENIZERS = load_tokenizers()
# Make openvino http client.
client = ovmsclient.make_http_client(os.getenv("OVMS_URL", "ovms:8080"))
# Make fastapi app.
app = FastAPI()

class Messages(BaseModel):
  """
  This class is used to handle list of messages.
  """
  messages: List[str]

@app.post("/encode/{model}")
async def encode(model: str, messages: Messages):
  """
  Encode messages into embs.
  :param model:     model name.
  :param messages:  messages as json.
  """
  assert model in TOKENIZERS, HTTPException(status_code=404)
  metadata = client.get_model_metadata(model)
  tokenizer = TOKENIZERS[model]
  tokenized = tokenizer.encode(messages.messages)
  inputs = dict()
  for k in metadata["inputs"].keys():
    inputs.update({k: np.array(tokenized[k], np.int64)})
  res = client.predict(model_name=model, inputs=inputs)
  res = postprocessing.mean_pooling(np.asarray(res), np.asarray(tokenized["attention_mask"])).squeeze().tolist()
  return json.dumps(res)