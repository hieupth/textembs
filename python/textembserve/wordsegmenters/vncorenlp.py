import os
import py_vncorenlp
from . import WordSegmenter


class VnCoreNLPWordSegmenter(WordSegmenter):
  """
  This class perform Vietnamese word segmentation via VnCoreNLP lib.
  """

  def __init__(self) -> None:
    """
    Class constructor.
    """
    # Automatically download VnCoreNLP components from the original repository
    # and save them in some local working folder.
    dir = os.getenv("VNCORENLP_DIR", os.path.expanduser("~/.vncorenlp"))
    os.makedirs(dir, exist_ok=True)
    py_vncorenlp.download_model(save_dir=dir)
    # Load the word and sentence segmentation component.
    self._segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=dir)
  
  def _segment(self, message: str) -> str:
    """
    Perform word segment on a message string.
    :param message: input string.
    :return:        segmented string.
    """
    return self._segmenter.word_segment(message)