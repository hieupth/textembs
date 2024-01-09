from .base import WordSegmenter
from .vncorenlp import VnCoreNLPWordSegmenter


def create_word_segmenter(name: str):
  if name is None:
    return WordSegmenter()
  else:
    name = name.lower()
    if name == "vncorenlp":
      return VnCoreNLPWordSegmenter()
    else:
      return WordSegmenter()