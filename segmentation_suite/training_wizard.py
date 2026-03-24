#!/usr/bin/env python3
"""
Training wizard window with QStackedWidget for multi-step workflow.
"""

import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QStackedWidget, QListWidget, QListWidgetItem,
    QFrame, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QSettings
from PyQt6.QtGui import QFont, QColor, QBrush

from .dpi_scaling import scaled, scaled_font, scaled_window_size, center_on_screen
from .widgets.subproject_panel import SubprojectPanel
from .wizard_pages.interactive_training_page import InteractiveTrainingPage
from .wizard_pages.finish_page import FinishPage
from .wizard_pages.home_page import HomePage
from .wizard_pages.segmentation_combined_page import SegmentationCombinedPage


class TrainingWizard(QMainWindow):
    """Training wizard window with step-by-step workflow."""

    # Signal emitted when wizard is closed
    wizard_closed = pyqtSignal()

    # Step indices - Simplified workflow (Setup merged into Home)
    STEP_HOME = 0
    STEP_TRAINING = 1  # Ground truth / interactive training
    STEP_SEGMENTATION = 2  # Combined: MOSS or LSD 2D
    # STEP_PROOFREADING — Work in progress, hidden for now
    STEP_EXPORT = 3

    STEP_NAMES = [
        "Home",
        "Ground Truth",
        "Segmentation",
        "Export"
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
        self.home_page = HomePage()
        self.training_page = InteractiveTrainingPage()
        self.segmentation_page = SegmentationCombinedPage()
        self.export_page = FinishPage()

        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.training_page)
        self.stack.addWidget(self.segmentation_page)
        self.stack.addWidget(self.export_page)

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

        # Start at home page
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

        # Subproject panel (hidden until project loaded)
        self.subproject_panel = SubprojectPanel()
        self.subproject_panel.subproject_changed.connect(self._on_subproject_changed)
        layout.addWidget(self.subproject_panel)

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

        self.session_btn = QPushButton("Multi-User Session")
        self.session_btn.setStyleSheet(btn_style)
        self.session_btn.setToolTip("Create or join a collaborative training session")
        self.session_btn.clicked.connect(self._open_session_dialog)
        self.session_btn.setEnabled(False)  # Disabled until project loaded
        layout.addWidget(self.session_btn)

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
        self._aggregation_server = None
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
        # Home page signals - now includes project loading
        self.home_page.project_loaded.connect(self._on_project_loaded)
        self.home_page.start_training.connect(lambda: self._go_to_page(self.STEP_TRAINING))
        self.home_page.start_3d_segmentation.connect(lambda: self._go_to_page(self.STEP_SEGMENTATION))

        # Training page signals
        self.training_page.training_complete.connect(self._on_training_complete)
        self.training_page.training_started.connect(self._on_training_started)
        self.training_page.training_stopped.connect(self._on_training_stopped)

        # Segmentation page signals
        self.segmentation_page.segmentation_complete.connect(self._on_segmentation_complete)
        self.segmentation_page.busy_changed.connect(self._on_segmentation_busy_changed)

    def _on_segmentation_busy_changed(self, busy: bool):
        """Handle segmentation busy state change."""
        self.next_btn.setEnabled(not busy)
        self.skip_btn.setEnabled(not busy)
        self.back_btn.setEnabled(not busy)

    def _on_segmentation_complete(self, output_path: str):
        """Handle segmentation completion."""
        self.config['segmentation_output'] = output_path
        self._propagate_config()
        self.home_page.refresh()
        # Auto-advance to export (proofreading is WIP, skipped for now)
        self.stack.setCurrentIndex(self.STEP_EXPORT)
        self._update_ui()

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

    def _go_back(self):
        """Go to previous step."""
        current = self.stack.currentIndex()
        if current > 0:
            new_page = current - 1
            self.stack.setCurrentIndex(new_page)
            self._update_ui()

    def _go_next(self):
        """Go to next step."""
        current = self.stack.currentIndex()

        # On Home page, ensure project is loaded before advancing
        if current == self.STEP_HOME:
            if not self.home_page.project_dir:
                QMessageBox.warning(self, "No Project",
                    "Please load or create a project before continuing.")
                return
            # Get config from home page
            self.config = self.home_page.get_config()
            self._ensure_project_dir()
            self._propagate_config()

        if current < self.STEP_EXPORT:
            self.stack.setCurrentIndex(current + 1)
            self._update_ui()

    def _skip_step(self):
        """Skip current step."""
        current = self.stack.currentIndex()
        if current < self.STEP_EXPORT:
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
        if current == self.STEP_EXPORT:
            self.next_btn.setVisible(False)
            self.skip_btn.setVisible(False)
        else:
            self.next_btn.setVisible(True)
            self.skip_btn.setVisible(current > self.STEP_HOME)

        # Show subproject panel only on Ground Truth page
        if hasattr(self, 'subproject_panel'):
            self.subproject_panel.setVisible(
                current == self.STEP_TRAINING
                and self.subproject_panel._project_dir is not None
            )

        # Update next button text based on simplified workflow
        if current == self.STEP_HOME:
            self.next_btn.setText("Start Workflow")
        elif current == self.STEP_TRAINING:
            self.next_btn.setText("Proceed to Segmentation")
        elif current == self.STEP_SEGMENTATION:
            self.next_btn.setText("Export")
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
        self.segmentation_page.set_config(self.config)
        self.export_page.set_config(self.config)

    def _on_project_loaded(self):
        """Handle project loaded from home page."""
        # Enable session buttons now that project is loaded
        self.session_btn.setEnabled(True)

        # Get config from home page
        self.config = self.home_page.get_config()
        self._ensure_project_dir()
        self._propagate_config()

        # Refresh subproject panel
        project_dir = self.config.get('project_dir', '')
        if project_dir:
            self.subproject_panel.set_project(project_dir)

        # Check if there's a saved page index for this project
        project_dir = os.path.normpath(self.config['project_dir'])
        settings = QSettings("MOSS", "SegmentationSuite")
        settings_key = f"page_index_{project_dir}"
        saved_page = settings.value(settings_key, None)

        # Restore visited pages
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

        # Always stay on Home page when loading/resuming a project
        # Let the user navigate to other pages manually via sidebar or action buttons
        print("Staying on Home page - user can navigate manually")
        self.stack.setCurrentIndex(self.STEP_HOME)
        self._update_ui()

    def _on_subproject_changed(self, subproject_name: str):
        """Handle subproject switch from the panel."""
        print(f"[Wizard] Switching to subproject: {subproject_name}")
        self.training_page.switch_subproject(subproject_name)

    def _on_training_complete(self, checkpoint_path: str):
        """Handle training completion."""
        self.config['checkpoint_path'] = checkpoint_path
        self._propagate_config()
        self.home_page.refresh()

    def _go_to_page(self, page_index: int):
        """Navigate to a specific page."""
        print(f"[_go_to_page] Navigating to page {page_index} ({self.STEP_NAMES[page_index] if page_index < len(self.STEP_NAMES) else 'unknown'})")
        if 0 <= page_index < self.stack.count():
            self.stack.setCurrentIndex(page_index)
            self._update_ui()
        else:
            print(f"[_go_to_page] Invalid page index: {page_index}")

    # =========================================================================
    # Multi-User Session Management
    # =========================================================================

    def _open_session_dialog(self):
        """Open unified multi-user session dialog."""
        from PyQt6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QRadioButton,
            QButtonGroup, QLabel, QLineEdit, QDialogButtonBox,
            QGroupBox, QComboBox
        )
        from PyQt6.QtCore import Qt
        from .project_config import has_subprojects, list_subprojects, get_active_subproject

        try:
            from .network import SyncClient, AggregationServer, get_local_ip, DEFAULT_RELAY_URL
        except ImportError as e:
            QMessageBox.warning(
                self, "Missing Dependency",
                f"Multi-user networking requires the 'websockets' library.\n"
                f"Install with: pip install websockets\n\nError: {e}"
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Multi-User Session")
        dialog.setMinimumWidth(420)
        dlayout = QVBoxLayout(dialog)

        # --- Role: Host or Join ---
        role_group = QGroupBox("Role")
        role_layout = QVBoxLayout(role_group)
        role_btn_group = QButtonGroup(dialog)
        host_radio = QRadioButton("Host a session (others connect to you)")
        join_radio = QRadioButton("Join an existing session")
        host_radio.setChecked(True)
        role_btn_group.addButton(host_radio)
        role_btn_group.addButton(join_radio)
        role_layout.addWidget(host_radio)
        role_layout.addWidget(join_radio)
        dlayout.addWidget(role_group)

        # --- Mode: LAN or Relay ---
        mode_group = QGroupBox("Connection Mode")
        mode_layout = QVBoxLayout(mode_group)
        mode_btn_group = QButtonGroup(dialog)
        lan_radio = QRadioButton("Local Network (LAN) — no internet needed")
        relay_radio = QRadioButton("Relay Server (Internet) — uses room codes")
        lan_radio.setChecked(True)  # LAN is default
        mode_btn_group.addButton(lan_radio)
        mode_btn_group.addButton(relay_radio)
        mode_layout.addWidget(lan_radio)
        mode_layout.addWidget(relay_radio)

        # Relay status hint
        relay_hint = QLabel("")
        relay_hint.setStyleSheet("color: #888; font-size: 11px; padding-left: 20px;")
        if DEFAULT_RELAY_URL:
            relay_hint.setText(f"Relay configured: {DEFAULT_RELAY_URL.split('//')[1].split('/')[0]}")
        else:
            relay_hint.setText("No relay configured (see network/relay_config.txt)")
        mode_layout.addWidget(relay_hint)
        dlayout.addWidget(mode_group)

        # --- Host settings: architecture + subproject (only visible when hosting) ---
        host_settings_group = QGroupBox("Session Settings (Host)")
        host_settings_layout = QVBoxLayout(host_settings_group)

        # Architecture selection
        arch_label = QLabel("Architecture (all participants will use this):")
        host_settings_layout.addWidget(arch_label)
        arch_combo = QComboBox()
        arch_id_map = self.training_page._arch_id_to_name  # {id: display_name}
        arch_ids = []
        for arch_id, display_name in arch_id_map.items():
            arch_combo.addItem(display_name)
            arch_ids.append(arch_id)
        # Select current architecture
        current_arch = self.training_page.current_architecture
        if current_arch in arch_id_map:
            idx = arch_combo.findText(arch_id_map[current_arch])
            if idx >= 0:
                arch_combo.setCurrentIndex(idx)
        host_settings_layout.addWidget(arch_combo)

        # Subproject selection
        sp_label = QLabel("Subproject (all participants will annotate for this):")
        host_settings_layout.addWidget(sp_label)
        sp_combo = QComboBox()
        project_dir = self.config.get('project_dir', '')
        subproject_names = []
        if project_dir and has_subprojects(project_dir):
            subproject_names = list_subprojects(project_dir)
            for sp_name in subproject_names:
                sp_combo.addItem(sp_name)
            active_sp = get_active_subproject(project_dir)
            if active_sp:
                idx = sp_combo.findText(active_sp)
                if idx >= 0:
                    sp_combo.setCurrentIndex(idx)
        else:
            sp_combo.addItem("(no subprojects)")
            sp_combo.setEnabled(False)
        host_settings_layout.addWidget(sp_combo)

        host_settings_layout.addWidget(QLabel(
            "Note: Only the host trains. Joinees annotate and send crops to the host."
        ))
        dlayout.addWidget(host_settings_group)

        # --- Join address/code input (only visible when joining) ---
        join_group = QGroupBox("Connection Details")
        join_layout = QVBoxLayout(join_group)
        address_label = QLabel("Host address (IP or IP:port):")
        address_input = QLineEdit()
        address_input.setPlaceholderText("e.g. 192.168.1.5 or 192.168.1.5:8765")
        join_layout.addWidget(address_label)
        join_layout.addWidget(address_input)
        join_group.setVisible(False)
        dlayout.addWidget(join_group)

        # --- Display name ---
        name_group = QGroupBox("Display Name")
        name_layout = QVBoxLayout(name_group)
        default_name = os.environ.get('USER', os.environ.get('USERNAME', 'user'))
        name_input = QLineEdit(default_name)
        name_layout.addWidget(name_input)
        dlayout.addWidget(name_group)

        # --- Buttons ---
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        dlayout.addWidget(buttons)

        # --- Dynamic UI updates ---
        def update_ui():
            is_hosting = host_radio.isChecked()
            is_joining = join_radio.isChecked()
            host_settings_group.setVisible(is_hosting)
            join_group.setVisible(is_joining)
            if is_joining and lan_radio.isChecked():
                address_label.setText("Host address (IP or IP:port):")
                address_input.setPlaceholderText("e.g. 192.168.1.5 or 192.168.1.5:8765")
            elif is_joining and relay_radio.isChecked():
                address_label.setText("6-character session code:")
                address_input.setPlaceholderText("e.g. ABC123")
            address_input.clear()

        host_radio.toggled.connect(update_ui)
        join_radio.toggled.connect(update_ui)
        lan_radio.toggled.connect(update_ui)
        relay_radio.toggled.connect(update_ui)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        name = name_input.text().strip()
        if not name:
            return

        is_host = host_radio.isChecked()
        is_lan = lan_radio.isChecked()

        # Get host settings
        selected_arch = arch_ids[arch_combo.currentIndex()] if arch_ids else current_arch
        selected_sp = sp_combo.currentText() if subproject_names else None

        if is_host:
            # Switch to selected subproject before starting session
            if selected_sp and selected_sp != self.training_page._active_subproject:
                self.training_page.switch_subproject(selected_sp)

            if is_lan:
                self._host_lan_session(name, selected_arch, selected_sp)
            else:
                self._host_relay_session(name, DEFAULT_RELAY_URL, selected_arch, selected_sp)
        else:
            if is_lan:
                address = address_input.text().strip()
                if not address:
                    QMessageBox.warning(self, "Missing Address", "Please enter the host address.")
                    return
                self._join_lan_session(address, name)
            else:
                code = address_input.text().strip().upper()
                if not code or len(code) != 6:
                    QMessageBox.warning(self, "Invalid Code", "Session code must be 6 characters.")
                    return
                self._join_relay_session(code, name, DEFAULT_RELAY_URL)

    def _host_lan_session(self, name: str, architecture: str = None, subproject: str = None):
        """Host a LAN session — starts local aggregation server + host client."""
        from .network import AggregationServer, SyncClient, get_local_ip

        arch = architecture or self.training_page.current_architecture
        try:
            self._aggregation_server = AggregationServer(parent=self)
            self._aggregation_server.server_started.connect(self._on_lan_server_started)
            self._aggregation_server.error.connect(self._on_session_error)

            self._aggregation_server.start(port=8765, architecture=arch)
            print(f"[Wizard] Started LAN server with architecture: {arch}")

            # Create a host client that connects to our own server
            local_ip = get_local_ip()
            self._session_client = SyncClient(parent=self)
            self._session_client.display_name = name
            self._session_client.connected.connect(lambda: self._on_lan_host_connected(local_ip, arch, subproject))
            self._session_client.disconnected.connect(self._on_session_disconnected)
            self._session_client.error.connect(self._on_session_error)
            self._session_client.user_list_updated.connect(self._on_user_list_updated)
            self._session_client.sync_status.connect(self._on_sync_status)

            self._session_client.connect_direct("127.0.0.1", 8765, name)
            self._is_session_host = True
            self.session_btn.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start LAN session:\n{e}")

    def _on_lan_server_started(self, connection_string: str):
        """Handle LAN server started."""
        print(f"[Wizard] LAN server started: {connection_string}")

    def _on_lan_host_connected(self, local_ip: str, architecture: str = None, subproject: str = None):
        """Handle host client connected to own LAN server."""
        print(f"[Wizard] LAN host connected")
        self.session_status_label.setText(f"LAN: {local_ip}:8765")
        self._update_session_ui(connected=True)
        self.training_page.set_multi_user_state(None, self._session_client, is_relay_host=True)
        arch = architecture or self.training_page.current_architecture
        self.training_page.lock_architecture(arch)
        self._lock_subproject(subproject)

    def _host_relay_session(self, name: str, relay_url: str, architecture: str = None, subproject: str = None):
        """Host a relay session — creates a room on the relay server."""
        from .network import SyncClient

        if not relay_url:
            QMessageBox.warning(
                self, "Relay Not Configured",
                "No relay server configured.\n\n"
                "Please set up relay_config.txt in the network folder.\n"
                "See SETUP_GUIDE.md in relay_server/ for instructions."
            )
            return

        self._session_client = SyncClient(parent=self)
        self._session_client.display_name = name
        self._session_client.room_created.connect(lambda code: self._on_room_created(code, architecture, subproject))
        self._session_client.disconnected.connect(self._on_session_disconnected)
        self._session_client.error.connect(self._on_session_error)
        self._session_client.user_list_updated.connect(self._on_user_list_updated)
        self._session_client.sync_status.connect(self._on_sync_status)

        self._session_client.create_relay_room(name, relay_url)
        self.session_btn.setEnabled(False)
        self._is_session_host = True

    def _join_lan_session(self, host_address: str, name: str):
        """Join a LAN session by direct IP connection."""
        from .network import SyncClient

        try:
            if ':' in host_address:
                host_ip, port_str = host_address.split(':')
                port = int(port_str)
            else:
                host_ip = host_address
                port = 8765
        except ValueError:
            QMessageBox.warning(self, "Invalid Address",
                "Please enter a valid address (e.g., 192.168.1.5 or 192.168.1.5:8765)")
            return

        self._session_client = SyncClient(parent=self)
        self._session_client.display_name = name
        self._session_client.connected.connect(self._on_lan_client_connected)
        self._session_client.disconnected.connect(self._on_session_disconnected)
        self._session_client.error.connect(self._on_session_error)
        self._session_client.user_list_updated.connect(self._on_user_list_updated)
        self._session_client.sync_status.connect(self._on_sync_status)
        self._session_client.architecture_received.connect(self._on_architecture_received)

        self._session_client.connect_direct(host_ip, port, name)
        self.session_btn.setEnabled(False)
        self._is_session_host = False

    def _on_lan_client_connected(self):
        """Handle successful LAN client connection."""
        print("[Wizard] LAN client connected")
        self.session_status_label.setText("LAN: Connected")
        self._update_session_ui(connected=True)
        self.training_page.set_multi_user_state(None, self._session_client, is_relay_host=False)
        # Lock subproject panel — joinee uses whatever subproject the host chose
        self._lock_subproject(self.training_page._active_subproject)
        # Lock training for joinees — only the host trains
        self.training_page.lock_training()

    def _join_relay_session(self, code: str, name: str, relay_url: str):
        """Join a relay session by room code."""
        from .network import SyncClient

        if not relay_url:
            QMessageBox.warning(
                self, "Relay Not Configured",
                "No relay server configured.\n\n"
                "Please set up relay_config.txt in the network folder.\n"
                "See SETUP_GUIDE.md in relay_server/ for instructions."
            )
            return

        self._session_client = SyncClient(parent=self)
        self._session_client.display_name = name
        self._session_client.room_joined.connect(self._on_room_joined)
        self._session_client.disconnected.connect(self._on_session_disconnected)
        self._session_client.error.connect(self._on_session_error)
        self._session_client.user_list_updated.connect(self._on_user_list_updated)
        self._session_client.sync_status.connect(self._on_sync_status)
        self._session_client.architecture_received.connect(self._on_architecture_received)

        self._session_client.connect_relay(code, name, relay_url)
        self.session_btn.setEnabled(False)
        self._is_session_host = False

    def _disconnect_session(self):
        """Disconnect from the current session."""
        if self._session_client:
            self._session_client.disconnect()
            self._session_client = None
        # Stop LAN server if we were hosting
        if hasattr(self, '_aggregation_server') and self._aggregation_server:
            self._aggregation_server.stop()
            self._aggregation_server = None
        self._is_session_host = False
        self._update_session_ui(connected=False)
        # Disable multi-user on training page
        self.training_page.disable_multi_user()
        # Unlock architecture, subproject, and training
        self.training_page.unlock_architecture()
        self.training_page.unlock_training()
        self._unlock_subproject()

    def _lock_subproject(self, subproject_name: str = None):
        """Lock the subproject panel during a multi-user session."""
        if subproject_name:
            print(f"[Wizard] Locking subproject to: {subproject_name}")
        self.subproject_panel.setEnabled(False)

    def _unlock_subproject(self):
        """Unlock the subproject panel after disconnecting."""
        self.subproject_panel.setEnabled(True)

    def _on_room_created(self, room_code: str, architecture: str = None, subproject: str = None):
        """Handle room created."""
        print(f"[Wizard] Room created: {room_code}")
        self.session_status_label.setText(f"Session: {room_code}")
        self._update_session_ui(connected=True)
        # Enable multi-user on training page as host
        self.training_page.set_multi_user_state(None, self._session_client, is_relay_host=True)
        arch = architecture or self.training_page.current_architecture
        self.training_page.lock_architecture(arch)
        self._lock_subproject(subproject)
        print(f"[Wizard] Locked architecture to: {arch}")

    def _on_room_joined(self, room_code: str):
        """Handle room joined."""
        print(f"[Wizard] Joined room: {room_code}")
        self.session_status_label.setText(f"Session: {room_code}")
        self._update_session_ui(connected=True)
        # Enable multi-user on training page as joinee
        self.training_page.set_multi_user_state(None, self._session_client, is_relay_host=False)
        # Lock subproject + training for joinee
        self._lock_subproject(self.training_page._active_subproject)
        self.training_page.lock_training()
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
        self.session_btn.setVisible(not connected)
        self.session_status_label.setVisible(connected)
        self.session_disconnect_btn.setVisible(connected)
        if not connected:
            self.session_btn.setEnabled(True)

    def _close_wizard(self):
        """Close the wizard and return to welcome page."""
        # Disconnect from session if connected
        if self._session_client:
            self._session_client.disconnect()
            self._session_client = None
        if self._aggregation_server:
            self._aggregation_server.stop()
            self._aggregation_server = None
        # Emit signal to return to welcome page (when embedded in launcher)
        self.wizard_closed.emit()

    def closeEvent(self, event):
        """Handle window close (when running standalone)."""
        # Disconnect from session if connected
        if self._session_client:
            self._session_client.disconnect()
        if self._aggregation_server:
            self._aggregation_server.stop()
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
