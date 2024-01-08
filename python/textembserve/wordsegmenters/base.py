
from typing import Any


class WordSegmenter:
  """
  This is base class of all inherited word segmenters.
  """

  def _segment(self, message: str) -> str:
    """
    Perform word segment on a message string.
    :param message: input string.
    :return:        segmented string.
    """
    return message

  def __call__(self, message, *args: Any, **kwds: Any) -> Any:
    """
    Perform word segment as a callable object.
    :param message: input strings.
    :param args:    additional arguments.
    :param kwds:    additional keyword arguments.
    """
    if isinstance(message, list):
      return [self._segment(x) for x in message]
    elif isinstance(message, str):
      return [self._segment(message)]
    else:
      return AssertionError("Word segment can only work with strings!")