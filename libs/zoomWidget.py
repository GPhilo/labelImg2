from PySide6.QtGui import *
from PySide6.QtCore import *
from PySide6.QtWidgets import *


class ZoomWidget(QSpinBox):

    def __init__(self, value=100):
        super(ZoomWidget, self).__init__()
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.setRange(1, 500)
        self.setSuffix(' %')
        self.setValue(value)
        self.setToolTip(u'Zoom Level')
        self.setStatusTip(self.toolTip())
        self.setAlignment(Qt.AlignCenter)
        self.setInputMethodHints(Qt.ImhDigitsOnly)
        
        self.setAttribute(Qt.WA_InputMethodEnabled, False)

    def minimumSizeHint(self):
        height = super(ZoomWidget, self).minimumSizeHint().height()
        fm = QFontMetrics(self.font())
        width = fm.horizontalAdvance(str(self.maximum()))
        return QSize(width, height)
