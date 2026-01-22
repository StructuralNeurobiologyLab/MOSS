#!/usr/bin/env python3
"""
Real-time loss plot widget using PyQtGraph.

Displays training loss over time for visualization in the wizard sidebar.
Shows individual points (faded) with a smoothed trend line.
"""

from collections import deque
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import pyqtSlot, Qt

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    pg = None


class LossPlotWidget(QWidget):
    """Real-time loss plot for training visualization."""

    def __init__(self, max_points: int = 500, smoothing_window: int = 50, parent=None):
        super().__init__(parent)
        self.max_points = max_points
        self.smoothing_window = smoothing_window
        self.loss_history = deque(maxlen=max_points)
        self.batch_history = deque(maxlen=max_points)
        self.smoothed_history = deque(maxlen=max_points)
        self._ema_value = None  # Exponential moving average

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        if not PYQTGRAPH_AVAILABLE:
            # Fallback if pyqtgraph not installed
            label = QLabel("pyqtgraph not installed\npip install pyqtgraph")
            label.setStyleSheet("color: #888; font-size: 10px;")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            self.plot_widget = None
            self.raw_curve = None
            self.smooth_curve = None
            return

        # Configure pyqtgraph for dark theme
        pg.setConfigOptions(antialias=True)

        # Create plot widget with transparent background
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('transparent')
        self.plot_widget.setLabel('left', 'Loss', color='#888')
        self.plot_widget.setLabel('bottom', 'Batch', color='#888')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)

        # Style the axes
        self.plot_widget.getAxis('left').setTextPen('#888')
        self.plot_widget.getAxis('bottom').setTextPen('#888')
        self.plot_widget.getAxis('left').setPen('#444')
        self.plot_widget.getAxis('bottom').setPen('#444')

        # Raw data points - faded green dots
        self.raw_curve = self.plot_widget.plot(
            pen=None,  # No line connecting dots
            symbol='o',
            symbolSize=3,
            symbolBrush=pg.mkBrush(76, 175, 80, 60),  # Green with low alpha
            symbolPen=None
        )

        # Smoothed trend line - solid green
        self.smooth_curve = self.plot_widget.plot(
            pen=pg.mkPen(color='#4CAF50', width=2)
        )

        layout.addWidget(self.plot_widget)
        self.setMinimumHeight(150)
        self.setMaximumHeight(200)

    def _update_ema(self, value: float) -> float:
        """Update exponential moving average."""
        alpha = 2.0 / (self.smoothing_window + 1)
        if self._ema_value is None:
            self._ema_value = value
        else:
            self._ema_value = alpha * value + (1 - alpha) * self._ema_value
        return self._ema_value

    @pyqtSlot(float, int)
    def add_point(self, loss: float, batch: int):
        """Add a new loss point to the plot."""
        if self.raw_curve is None:
            return

        self.loss_history.append(loss)
        self.batch_history.append(batch)

        # Update smoothed value
        smoothed = self._update_ema(loss)
        self.smoothed_history.append(smoothed)

        # Update plots
        batches = list(self.batch_history)
        self.raw_curve.setData(batches, list(self.loss_history))
        self.smooth_curve.setData(batches, list(self.smoothed_history))

    @pyqtSlot(int, int, float, float)
    def add_epoch_point(self, epoch: int, total_epochs: int, train_loss: float, val_loss: float):
        """Add a point using epoch-based progress signal format."""
        if self.raw_curve is None:
            return

        self.loss_history.append(train_loss)
        self.batch_history.append(epoch)

        # Update smoothed value
        smoothed = self._update_ema(train_loss)
        self.smoothed_history.append(smoothed)

        # Update plots
        batches = list(self.batch_history)
        self.raw_curve.setData(batches, list(self.loss_history))
        self.smooth_curve.setData(batches, list(self.smoothed_history))

    def clear(self):
        """Clear the plot."""
        self.loss_history.clear()
        self.batch_history.clear()
        self.smoothed_history.clear()
        self._ema_value = None
        if self.raw_curve is not None:
            self.raw_curve.setData([], [])
        if self.smooth_curve is not None:
            self.smooth_curve.setData([], [])

