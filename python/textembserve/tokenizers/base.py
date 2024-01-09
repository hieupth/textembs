from textembserve import textembserve
from textembserve.wordsegmenters import WordSegmenter


class Tokenizer:
  """
  This is base class of all inherited tokenizers.
  """

  def __init__(self, model: str = None, wseg = None, **kwargs) -> None:
    """
    Class constructor.
    :param model:   model name or path.
    :param wseg:    word segmenter to be used.
    :param kwargs:  additional keyword arguments.
    """
    self._tokenizer = textembserve.RustTokenizer(model)
    self._wordsegmenter = WordSegmenter() if wseg is None else wseg

  def encode(self, message):
    """
    Encode message string(s) into tokens.
    :param message: input string(s).
    :return:        tokens.
    """
    return self._tokenizer.encode(self._wordsegmenter(message))
