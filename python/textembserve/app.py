import os
import json
import ovmsclient
from fastapi import FastAPI, HTTPException
from textembserve.tokenizers import *
from textembserve import wordsegmenters
from pydantic import BaseModel
from typing import List


def load_tokenizers():
  tokenizers = dict()
  #
  with open(os.getenv("TOKENIZER_CONF_FILE", "/tokenizer_config.json"), "r") as file:
    conf = json.load(file)
  #
  for k, v in conf.items():
    wseg = wordsegmenters.create_word_segmenter(v.get("word_segmenter"))
    tok = Tokenizer(model=v.get("tokenizer_model"), wordsegmenter=wseg)
    tokenizers.update({k: tok})
  #
  return tokenizers
#
TOKENIZERS = load_tokenizers()

#
client = ovmsclient.make_http_client(os.getenv("OVMS_URL", "http://ovms:8080"))
#
app = FastAPI()

class Messages(BaseModel):
  messages: List[str]

@app.post("/encode/{model}")
async def encode(model: str, messages: Messages):
  assert model in TOKENIZERS, HTTPException(status_code=404)
  tokenizer = TOKENIZERS[model]
  tokenized = tokenizer.encode(messages.messages)
  return tokenized