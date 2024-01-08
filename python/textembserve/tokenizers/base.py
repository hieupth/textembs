from textembserve import textembserve
from textembserve.wordsegmenters import WordSegmenter


class Tokenizer:

  def __init__(
      self, 
      model: str = "vinai/phobert-base-v2", 
      wordsegmenter = WordSegmenter(), 
      **kwargs
      ) -> None:
    """
    Class constructor.
    :param model:         model name or path.
    :param wordsegmenter: word segmenter to be used.
    :param kwargs:        additional keyword arguments.
    """
    self._tokenizer = textembserve.Rustokenizer(model)
    self._wordsegmenter = wordsegmenter

  def encode(self, message):
    """
    Encode message string(s) into tokens.
    :param message: input string(s).
    :return:        tokens.
    """
    return self._tokenizer.encode(self._wordsegmenter(message))
