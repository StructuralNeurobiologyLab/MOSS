#!/usr/bin/env python3
"""
Training wizard window with QStackedWidget for multi-step workflow.
"""

import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QStackedWidget, QListWidget, QListWidgetItem,
    QFrame, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QSettings
from PyQt6.QtGui import QFont, QColor, QBrush

from .dpi_scaling import scaled, scaled_font, scaled_window_size, center_on_screen
from .wizard_pages.setup_page import SetupPage
from .wizard_pages.interactive_training_page import InteractiveTrainingPage
from .wizard_pages.reslice_page import ReslicePage
from .wizard_pages.prediction_page import PredictionPage
from .wizard_pages.voting_page import VotingPage
from .wizard_pages.cleanup_page import CleanupPage
from .wizard_pages.finish_page import FinishPage


class TrainingWizard(QMainWindow):
    """Training wizard window with step-by-step workflow."""

    # Signal emitted when wizard is closed
    wizard_closed = pyqtSignal()

    # Step indices
    STEP_SETUP = 0
    STEP_TRAINING = 1
    STEP_RESLICE = 2
    STEP_PREDICTION = 3
    STEP_VOTING = 4
    STEP_CLEANUP = 5
    STEP_FINISH = 6

    STEP_NAMES = [
        "Setup",
        "Training",
        "Reslicing",
        "Prediction",
        "Voting",
        "Cleanup",
        "Finish"
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = {}
        self.visited_pages = set()  # Track which pages have been visited
        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        self.setWindowTitle("MOSS - Training Wizard")

        # Use scaled window size
        win_w, win_h = scaled_window_size(900, 700)
        self.setMinimumSize(win_w, win_h)
        center_on_screen(self)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar with step list (no splitter - fixed width)
        sidebar = self._create_sidebar()
        main_layout.addWidget(sidebar)

        # Content area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        margin = scaled(20)
        content_layout.setContentsMargins(margin, margin, margin, margin)

        # Stacked widget for pages
        self.stack = QStackedWidget()
        content_layout.addWidget(self.stack)

        # Create pages
        self.setup_page = SetupPage()
        self.training_page = InteractiveTrainingPage()
        self.reslice_page = ReslicePage()
        self.prediction_page = PredictionPage()
        self.voting_page = VotingPage()
        self.cleanup_page = CleanupPage()
        self.finish_page = FinishPage()

        self.stack.addWidget(self.setup_page)
        self.stack.addWidget(self.training_page)
        self.stack.addWidget(self.reslice_page)
        self.stack.addWidget(self.prediction_page)
        self.stack.addWidget(self.voting_page)
        self.stack.addWidget(self.cleanup_page)
        self.stack.addWidget(self.finish_page)

        # Navigation buttons
        nav_layout = QHBoxLayout()

        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(self._go_back)
        nav_layout.addWidget(self.back_btn)

        nav_layout.addStretch()

        self.skip_btn = QPushButton("Skip")
        self.skip_btn.clicked.connect(self._skip_step)
        nav_layout.addWidget(self.skip_btn)

        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self._go_next)
        pad_v = scaled(8)
        pad_h = scaled(20)
        radius = scaled(4)
        self.next_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #2196F3;
                color: white;
                padding: {pad_v}px {pad_h}px;
                border-radius: {radius}px;
            }}
            QPushButton:hover {{
                background-color: #1976D2;
            }}
            QPushButton:disabled {{
                background-color: #BDBDBD;
            }}
        """)
        nav_layout.addWidget(self.next_btn)

        content_layout.addLayout(nav_layout)

        main_layout.addWidget(content_widget, 1)  # Stretch factor 1 to fill space

        # Start at setup page
        self._update_ui()

    def _create_sidebar(self) -> QWidget:
        """Create the sidebar with step list."""
        sidebar = QFrame()
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: none;
            }
        """)
        sidebar.setMinimumWidth(scaled(180))
        sidebar.setMaximumWidth(scaled(250))

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)

        # Title
        title = QLabel("Workflow Steps")
        title.setFont(scaled_font(12, QFont.Weight.Bold))
        pad = scaled(15)
        title.setStyleSheet(f"color: white; padding: {pad}px;")
        layout.addWidget(title)

        # Step list with scaled padding
        item_pad_v = scaled(12)
        item_pad_h = scaled(15)
        self.step_list = QListWidget()
        self.step_list.setStyleSheet(f"""
            QListWidget {{
                background-color: transparent;
                border: none;
                outline: none;
            }}
            QListWidget::item {{
                color: #888888;
                padding: {item_pad_v}px {item_pad_h}px;
                border-left: 3px solid transparent;
            }}
            QListWidget::item:selected {{
                color: white;
                background-color: #3d3d3d;
                border-left: 3px solid #2196F3;
            }}
            QListWidget::item:hover {{
                background-color: #353535;
            }}
        """)

        for name in self.STEP_NAMES:
            item = QListWidgetItem(name)
            self.step_list.addItem(item)

        self.step_list.currentRowChanged.connect(self._on_step_clicked)
        layout.addWidget(self.step_list)

        layout.addStretch()

        # Multi-user session section
        session_label = QLabel("Multi-User Session")
        session_label.setStyleSheet(f"""
            QLabel {{
                color: #aaaaaa;
                font-size: {scaled(11)}px;
                padding: {scaled(10)}px {scaled(15)}px {scaled(5)}px {scaled(15)}px;
            }}
        """)
        layout.addWidget(session_label)

        btn_style = f"""
            QPushButton {{
                color: #cccccc;
                background-color: #3d3d3d;
                border: none;
                padding: {scaled(8)}px {scaled(15)}px;
                text-align: left;
                margin: 2px {scaled(10)}px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: #4d4d4d;
                color: white;
            }}
            QPushButton:disabled {{
                color: #666666;
                background-color: #2d2d2d;
            }}
        """

        self.session_create_btn = QPushButton("Create Session")
        self.session_create_btn.setStyleSheet(btn_style)
        self.session_create_btn.setToolTip("Create a session for collaborative training")
        self.session_create_btn.clicked.connect(self._create_session)
        self.session_create_btn.setEnabled(False)  # Disabled until project loaded
        layout.addWidget(self.session_create_btn)

        self.session_join_btn = QPushButton("Join Session")
        self.session_join_btn.setStyleSheet(btn_style)
        self.session_join_btn.setToolTip("Join an existing session by code")
        self.session_join_btn.clicked.connect(self._join_session)
        self.session_join_btn.setEnabled(False)  # Disabled until project loaded
        layout.addWidget(self.session_join_btn)

        self.session_status_label = QLabel("")
        self.session_status_label.setStyleSheet(f"""
            QLabel {{
                color: #4CAF50;
                font-weight: bold;
                padding: {scaled(5)}px {scaled(15)}px;
            }}
        """)
        self.session_status_label.setVisible(False)
        layout.addWidget(self.session_status_label)

        self.session_disconnect_btn = QPushButton("Disconnect")
        self.session_disconnect_btn.setStyleSheet(btn_style.replace("#3d3d3d", "#5d3d3d").replace("#4d4d4d", "#6d4d4d"))
        self.session_disconnect_btn.setToolTip("Leave the current session")
        self.session_disconnect_btn.clicked.connect(self._disconnect_session)
        self.session_disconnect_btn.setVisible(False)
        layout.addWidget(self.session_disconnect_btn)

        # Session state
        self._session_client = None
        self._is_session_host = False

        layout.addSpacing(scaled(10))

        # Loss plot section
        loss_label = QLabel("Training Loss")
        loss_label.setStyleSheet(f"""
            QLabel {{
                color: #aaaaaa;
                font-size: {scaled(11)}px;
                padding: {scaled(10)}px {scaled(15)}px {scaled(5)}px {scaled(15)}px;
            }}
        """)
        layout.addWidget(loss_label)
        self._loss_label = loss_label
        self._loss_label.setVisible(False)  # Hidden until training starts

        from .widgets import LossPlotWidget
        self.loss_plot = LossPlotWidget(max_points=500)
        self.loss_plot.setVisible(False)  # Hidden until training starts
        layout.addWidget(self.loss_plot)

        layout.addSpacing(scaled(10))

        # Close button
        btn_pad = scaled(15)
        close_btn = QPushButton("Close Wizard")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                color: #888888;
                background-color: transparent;
                border: none;
                padding: {btn_pad}px;
                text-align: left;
            }}
            QPushButton:hover {{
                color: white;
                background-color: #353535;
            }}
        """)
        close_btn.clicked.connect(self._close_wizard)
        layout.addWidget(close_btn)

        return sidebar

    def connect_signals(self):
        """Connect page signals."""
        # Setup page signals - auto-advance when project loaded
        self.setup_page.project_loaded.connect(self._on_project_loaded)

        # Training page signals
        self.training_page.training_complete.connect(self._on_training_complete)
        self.training_page.training_started.connect(self._on_training_started)
        self.training_page.training_stopped.connect(self._on_training_stopped)

        # Reslice page signals
        self.reslice_page.reslice_complete.connect(self._on_reslice_complete)
        self.reslice_page.busy_changed.connect(self._on_reslice_busy_changed)

        # Prediction page signals
        self.prediction_page.prediction_complete.connect(self._on_prediction_complete)
        self.prediction_page.busy_changed.connect(self._on_prediction_busy_changed)

        # Voting page signals
        self.voting_page.voting_complete.connect(self._on_voting_complete)

        # Cleanup page signals
        self.cleanup_page.cleanup_complete.connect(self._on_cleanup_complete)
        self.cleanup_page.busy_changed.connect(self._on_cleanup_busy_changed)

        # Finish page signals
        self.finish_page.open_annotation_tool.connect(self._on_open_annotation_tool)

        # Multi-user architecture locking
        self.setup_page.architecture_locked.connect(self.training_page.lock_architecture)
        self.setup_page.architecture_unlocked.connect(self.training_page.unlock_architecture)
        self.training_page.architecture_changed.connect(self.setup_page.set_current_architecture)

    def _on_reslice_busy_changed(self, busy: bool):
        """Handle reslice busy state change."""
        # Disable Next/Skip/Back buttons while reslicing
        self.next_btn.setEnabled(not busy)
        self.skip_btn.setEnabled(not busy)
        self.back_btn.setEnabled(not busy)

    def _on_prediction_busy_changed(self, busy: bool):
        """Handle prediction busy state change."""
        # Disable Next/Skip/Back buttons while predicting
        self.next_btn.setEnabled(not busy)
        self.skip_btn.setEnabled(not busy)
        self.back_btn.setEnabled(not busy)

    def _on_cleanup_busy_changed(self, busy: bool):
        """Handle cleanup busy state change."""
        # Disable Next/Skip/Back buttons while cleaning up
        self.next_btn.setEnabled(not busy)
        self.skip_btn.setEnabled(not busy)
        self.back_btn.setEnabled(not busy)

    def _on_training_started(self):
        """Handle training started signal."""
        # Show loss plot and connect signal
        self._loss_label.setVisible(True)
        self.loss_plot.setVisible(True)
        self.loss_plot.clear()

        # Connect loss updates from training page
        try:
            self.training_page.loss_updated.connect(self.loss_plot.add_point)
        except TypeError:
            pass  # Already connected

    def _on_training_stopped(self):
        """Handle training stopped signal."""
        # Disconnect loss updates
        try:
            self.training_page.loss_updated.disconnect(self.loss_plot.add_point)
        except TypeError:
            pass  # Already disconnected or never connected

        # Keep the plot visible so user can see final results
        # It will be cleared on next training start

    def _on_step_clicked(self, row: int):
        """Handle step list click."""
        # Allow going to any visited page or current/previous pages
        current = self.stack.currentIndex()
        if row <= current or row in self.visited_pages:
            print(f"Navigating to page {row} ({self.STEP_NAMES[row]})")
            self.stack.setCurrentIndex(row)
            self._update_ui()

            # If navigating to Training page, ensure multi-user state is passed
            if row == self.STEP_TRAINING:
                self._pass_multi_user_state()

    def _go_back(self):
        """Go to previous step."""
        current = self.stack.currentIndex()
        if current > 0:
            new_page = current - 1
            self.stack.setCurrentIndex(new_page)
            self._update_ui()

            # If navigating to Training page, ensure multi-user state is passed
            if new_page == self.STEP_TRAINING:
                self._pass_multi_user_state()

    def _pass_multi_user_state(self):
        """Pass multi-user state from setup page to training page."""
        server, client, is_relay_host = self.setup_page.get_multi_user_state()
        if server or client:
            print(f"[Wizard] Passing multi-user state to training page (server={server is not None}, client={client is not None}, relay_host={is_relay_host})")
            self.training_page.set_multi_user_state(server, client, is_relay_host)

    def _go_next(self):
        """Go to next step."""
        current = self.stack.currentIndex()

        # Validate current step
        if current == self.STEP_SETUP:
            valid, msg = self.setup_page.validate()
            if not valid:
                QMessageBox.warning(self, "Validation Error", msg)
                return
            # Get config and pass to subsequent pages
            self.config = self.setup_page.get_config()
            self._ensure_project_dir()
            self._propagate_config()

        if current < self.STEP_FINISH:
            self.stack.setCurrentIndex(current + 1)
            self._update_ui()

    def _skip_step(self):
        """Skip current step."""
        current = self.stack.currentIndex()
        if current < self.STEP_FINISH:
            self.stack.setCurrentIndex(current + 1)
            self._update_ui()

    def _update_ui(self):
        """Update UI based on current step."""
        current = self.stack.currentIndex()

        # Mark current page as visited
        self.visited_pages.add(current)

        # Save current page index for resume functionality
        if hasattr(self, 'config') and self.config.get('project_dir'):
            # Normalize path to ensure consistent key
            project_dir = os.path.normpath(self.config['project_dir'])
            settings = QSettings("MOSS", "SegmentationSuite")
            settings.setValue(f"page_index_{project_dir}", current)
            # Also save visited pages
            settings.setValue(f"visited_pages_{project_dir}", list(self.visited_pages))

        # Update step list - show visited pages as clickable
        for i in range(self.step_list.count()):
            item = self.step_list.item(i)
            if i < current:
                # Previous steps - show checkmark
                item.setText(f"✓ {self.STEP_NAMES[i]}")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEnabled)
                item.setForeground(QBrush(QColor(100, 200, 100)))  # Green
            elif i == current:
                # Current step
                item.setText(f"● {self.STEP_NAMES[i]}")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEnabled)
                item.setForeground(QBrush(QColor(255, 255, 255)))  # White
            elif i in self.visited_pages:
                # Previously visited future step - clickable
                item.setText(f"◆ {self.STEP_NAMES[i]}")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEnabled)
                item.setForeground(QBrush(QColor(100, 150, 200)))  # Blue
            else:
                # Unvisited future steps - greyed out and disabled
                item.setText(f"○ {self.STEP_NAMES[i]}")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setForeground(QBrush(QColor(80, 80, 80)))  # Dark grey

        # Update step list selection
        self.step_list.setCurrentRow(current)

        # Update back button
        self.back_btn.setEnabled(current > 0)

        # Update next/skip buttons
        if current == self.STEP_FINISH:
            self.next_btn.setVisible(False)
            self.skip_btn.setVisible(False)
        else:
            self.next_btn.setVisible(True)
            self.skip_btn.setVisible(current > self.STEP_SETUP)

        # Update next button text
        if current == self.STEP_SETUP:
            self.next_btn.setText("Start Workflow")
        elif current == self.STEP_VOTING:
            self.next_btn.setText("Cleanup Masks")
        elif current == self.STEP_CLEANUP:
            self.next_btn.setText("View Results")
        else:
            self.next_btn.setText("Next")

    def _ensure_project_dir(self):
        """Create project directory if it doesn't exist."""
        project_dir = self.config.get('project_dir', '')
        if project_dir and not os.path.exists(project_dir):
            os.makedirs(project_dir, exist_ok=True)

    def _propagate_config(self):
        """Pass config to all pages."""
        self.training_page.set_config(self.config)
        self.reslice_page.set_config(self.config)
        self.prediction_page.set_config(self.config)
        self.voting_page.set_config(self.config)
        self.cleanup_page.set_config(self.config)
        self.finish_page.set_config(self.config)

        # Pass multi-user state from setup to training page
        server, client, is_relay_host = self.setup_page.get_multi_user_state()
        if server or client:
            self.training_page.set_multi_user_state(server, client, is_relay_host)

    def _on_project_loaded(self):
        """Handle project loaded - restore to last page or advance to training."""
        valid, msg = self.setup_page.validate()
        if not valid:
            QMessageBox.warning(self, "Validation Error", msg)
            return

        # Enable session buttons now that project is loaded
        self.session_create_btn.setEnabled(True)
        self.session_join_btn.setEnabled(True)

        # Get config first
        self.config = self.setup_page.get_config()
        self._ensure_project_dir()
        self._propagate_config()

        # Check if there's a saved page index for this project
        # Normalize path to ensure consistent key
        project_dir = os.path.normpath(self.config['project_dir'])
        settings = QSettings("MOSS", "SegmentationSuite")
        settings_key = f"page_index_{project_dir}"
        saved_page = settings.value(settings_key, None)

        # Restore visited pages (clear first to avoid carrying over from previous project)
        self.visited_pages = set()
        visited_key = f"visited_pages_{project_dir}"
        saved_visited = settings.value(visited_key, None)
        if saved_visited:
            try:
                self.visited_pages = set(int(p) for p in saved_visited)
                print(f"Restored visited pages: {self.visited_pages}")
            except (TypeError, ValueError):
                self.visited_pages = set()

        print(f"Looking for saved page with key: {settings_key}")
        print(f"Found saved_page: {saved_page}")

        if saved_page is not None:
            saved_page = int(saved_page)
            # Restore to saved page (but at least go to training)
            target_page = max(self.STEP_TRAINING, saved_page)
            print(f"Resuming project at page {target_page} ({self.STEP_NAMES[target_page]})")
            self.stack.setCurrentIndex(target_page)
            self._update_ui()
        else:
            # New project - go to training
            print("No saved page found, starting at Training")
            self.stack.setCurrentIndex(self.STEP_TRAINING)
            self._update_ui()

    def _on_training_complete(self, checkpoint_path: str):
        """Handle training completion."""
        self.config['checkpoint_path'] = checkpoint_path
        self._propagate_config()

    def _on_reslice_complete(self, reslice_dirs: dict):
        """Handle reslice completion."""
        self.config['reslice_dirs'] = reslice_dirs
        self.config['raw_images_dir'] = self.reslice_page.input_dir.text()
        self._propagate_config()

    def _on_prediction_complete(self, prediction_dirs: dict):
        """Handle prediction completion."""
        self.config['prediction_dirs'] = prediction_dirs
        self._propagate_config()

    def _on_voting_complete(self, heatmap_dir: str):
        """Handle voting completion."""
        self.config['heatmap_dir'] = heatmap_dir
        self._propagate_config()
        # Auto-advance to cleanup page
        self.stack.setCurrentIndex(self.STEP_CLEANUP)
        self._update_ui()

    def _on_cleanup_complete(self, output_dir: str):
        """Handle cleanup completion."""
        self.config['binary_masks_dir'] = output_dir
        self._propagate_config()
        # Auto-advance to finish page
        self.stack.setCurrentIndex(self.STEP_FINISH)
        self._update_ui()

    def _on_open_annotation_tool(self, images_dir: str, masks_dir: str):
        """Handle request to open annotation tool."""
        import subprocess
        try:
            subprocess.Popen(['annotation-tool'])
        except FileNotFoundError:
            QMessageBox.warning(
                self,
                "Annotation Tool Not Found",
                "The annotation-tool command was not found.\n"
                "Please ensure it is installed and in your PATH."
            )

    # =========================================================================
    # Multi-User Session Management
    # =========================================================================

    def _create_session(self):
        """Create a multi-user session."""
        from PyQt6.QtWidgets import QInputDialog
        try:
            from .network import SyncClient, DEFAULT_RELAY_URL
        except ImportError as e:
            QMessageBox.warning(
                self, "Missing Dependency",
                f"Multi-user networking requires the 'websockets' library.\n"
                f"Install with: pip install websockets\n\nError: {e}"
            )
            return

        if not DEFAULT_RELAY_URL:
            QMessageBox.warning(
                self, "Relay Not Configured",
                "No relay server configured.\n\n"
                "Please set up relay_config.txt in the network folder.\n"
                "See SETUP_GUIDE.md in relay_server/ for instructions."
            )
            return

        # Get display name
        default_name = os.environ.get('USER', os.environ.get('USERNAME', 'user'))
        name, ok = QInputDialog.getText(
            self, "Create Session",
            "Enter your display name:",
            text=default_name + "_host"
        )
        if not ok or not name.strip():
            return

        # Create client and connect
        self._session_client = SyncClient(parent=self)
        self._session_client.display_name = name.strip()
        self._session_client.room_created.connect(self._on_room_created)
        self._session_client.disconnected.connect(self._on_session_disconnected)
        self._session_client.error.connect(self._on_session_error)
        self._session_client.user_list_updated.connect(self._on_user_list_updated)
        self._session_client.sync_status.connect(self._on_sync_status)

        self._session_client.create_relay_room(name.strip(), DEFAULT_RELAY_URL)
        self.session_create_btn.setEnabled(False)
        self.session_join_btn.setEnabled(False)
        self._is_session_host = True

    def _join_session(self):
        """Join an existing session."""
        from PyQt6.QtWidgets import QInputDialog
        try:
            from .network import SyncClient, DEFAULT_RELAY_URL
        except ImportError as e:
            QMessageBox.warning(
                self, "Missing Dependency",
                f"Multi-user networking requires the 'websockets' library.\n"
                f"Install with: pip install websockets\n\nError: {e}"
            )
            return

        if not DEFAULT_RELAY_URL:
            QMessageBox.warning(
                self, "Relay Not Configured",
                "No relay server configured.\n\n"
                "Please set up relay_config.txt in the network folder.\n"
                "See SETUP_GUIDE.md in relay_server/ for instructions."
            )
            return

        # Get room code
        code, ok = QInputDialog.getText(
            self, "Join Session",
            "Enter 6-character session code:"
        )
        if not ok or not code.strip():
            return

        code = code.strip().upper()
        if len(code) != 6:
            QMessageBox.warning(self, "Invalid Code", "Session code must be 6 characters.")
            return

        # Get display name
        default_name = os.environ.get('USER', os.environ.get('USERNAME', 'user'))
        name, ok = QInputDialog.getText(
            self, "Join Session",
            "Enter your display name:",
            text=default_name
        )
        if not ok or not name.strip():
            return

        # Create client and connect
        self._session_client = SyncClient(parent=self)
        self._session_client.display_name = name.strip()
        self._session_client.room_joined.connect(self._on_room_joined)
        self._session_client.disconnected.connect(self._on_session_disconnected)
        self._session_client.error.connect(self._on_session_error)
        self._session_client.user_list_updated.connect(self._on_user_list_updated)
        self._session_client.sync_status.connect(self._on_sync_status)
        self._session_client.architecture_received.connect(self._on_architecture_received)

        self._session_client.connect_relay(code, name.strip(), DEFAULT_RELAY_URL)
        self.session_create_btn.setEnabled(False)
        self.session_join_btn.setEnabled(False)
        self._is_session_host = False

    def _disconnect_session(self):
        """Disconnect from the current session."""
        if self._session_client:
            self._session_client.disconnect()
            self._session_client = None
        self._is_session_host = False
        self._update_session_ui(connected=False)
        # Disable multi-user on training page
        self.training_page.disable_multi_user()
        # Unlock architecture
        self.training_page.unlock_architecture()

    def _on_room_created(self, room_code: str):
        """Handle room created."""
        print(f"[Wizard] Room created: {room_code}")
        self.session_status_label.setText(f"Session: {room_code}")
        self._update_session_ui(connected=True)
        # Enable multi-user on training page as host
        self.training_page.set_multi_user_state(None, self._session_client, is_relay_host=True)
        # Lock architecture - host's current architecture becomes the session architecture
        current_arch = self.training_page.current_architecture
        self.training_page.lock_architecture(current_arch)
        print(f"[Wizard] Locked architecture to: {current_arch}")

    def _on_room_joined(self, room_code: str):
        """Handle room joined."""
        print(f"[Wizard] Joined room: {room_code}")
        self.session_status_label.setText(f"Session: {room_code}")
        self._update_session_ui(connected=True)
        # Enable multi-user on training page as joinee
        self.training_page.set_multi_user_state(None, self._session_client, is_relay_host=False)
        # Architecture will be locked when architecture_received signal fires

    def _on_architecture_received(self, architecture: str):
        """Handle architecture received from host - lock to that architecture."""
        print(f"[Wizard] Received session architecture: {architecture}")
        self.training_page.lock_architecture(architecture)

    def _on_session_disconnected(self):
        """Handle disconnection."""
        print("[Wizard] Disconnected from session")
        self._update_session_ui(connected=False)
        self.training_page.disable_multi_user()
        self.training_page.unlock_architecture()

    def _on_session_error(self, error: str):
        """Handle session error."""
        print(f"[Wizard] Session error: {error}")
        QMessageBox.warning(self, "Session Error", error)
        self._update_session_ui(connected=False)

    def _on_user_list_updated(self, users: list):
        """Handle user list update."""
        count = len(users)
        if count > 1:
            self.session_status_label.setText(
                f"{self.session_status_label.text().split(' (')[0]} ({count} users)"
            )

    def _on_sync_status(self, status: str):
        """Handle sync status."""
        print(f"[Wizard] Sync: {status}")

    def _update_session_ui(self, connected: bool):
        """Update session UI based on connection state."""
        self.session_create_btn.setVisible(not connected)
        self.session_join_btn.setVisible(not connected)
        self.session_status_label.setVisible(connected)
        self.session_disconnect_btn.setVisible(connected)
        if not connected:
            self.session_create_btn.setEnabled(True)
            self.session_join_btn.setEnabled(True)

    def _close_wizard(self):
        """Close the wizard and return to welcome page."""
        # Disconnect from session if connected
        if self._session_client:
            self._session_client.disconnect()
            self._session_client = None
        # Emit signal to return to welcome page (when embedded in launcher)
        self.wizard_closed.emit()

    def closeEvent(self, event):
        """Handle window close (when running standalone)."""
        # Disconnect from session if connected
        if self._session_client:
            self._session_client.disconnect()
        self.wizard_closed.emit()
        event.accept()


def main():
    """Run the training wizard standalone."""
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = TrainingWizard()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
