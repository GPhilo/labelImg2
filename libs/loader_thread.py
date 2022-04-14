from PySide6.QtCore import QThread, Signal
import mmocr.datasets.pipelines # necessary even if unused explicitly
from mmocr.apis.inference import init_detector, model_inference
import numpy as np

class LoaderThread(QThread):
  model_loaded = Signal()

  def __init__(self, config_path, ckpt_path, parent=None):
    super().__init__(parent)
    self.config_path = config_path
    self.ckpt_path = ckpt_path
    self.model = None
  
  def run(self):
    self.model = init_detector(config=self.config_path, checkpoint=self.ckpt_path)
    _ = model_inference(self.model, np.zeros([200,200,3], np.uint8))
    self.model_loaded.emit()
