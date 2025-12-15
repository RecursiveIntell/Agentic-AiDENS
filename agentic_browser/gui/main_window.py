"""
Main window for Agentic Browser GUI.

Provides the primary application interface.
"""

import sys
import json
import subprocess
from typing import Optional
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QListWidget,
    QListWidgetItem, QStatusBar, QSplitter, QFrame,
    QMessageBox,
)
from PySide6.QtCore import Qt, QProcess, Signal, Slot, QTimer
from PySide6.QtGui import QFont

from ..settings_store import SettingsStore, get_settings
from ..providers import Provider, PROVIDER_DISPLAY_NAMES

from .settings_dialog import SettingsDialog


class StepItem(QFrame):
    """Custom widget for displaying a step in the log."""
    
    def __init__(self, step: int, action: str, args: str, risk: str, parent=None):
        super().__init__(parent)
        self.step = step
        self.action = action
        self.args_str = args
        self.risk = risk
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        
        # Step number
        step_label = QLabel(f"#{self.step}")
        step_label.setFixedWidth(35)
        step_label.setStyleSheet("font-weight: bold; color: #666;")
        layout.addWidget(step_label)
        
        # Action
        action_label = QLabel(self.action)
        action_label.setFixedWidth(100)
        action_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        layout.addWidget(action_label)
        
        # Args summary
        args_text = self.args_str[:60] + "..." if len(self.args_str) > 60 else self.args_str
        args_label = QLabel(args_text)
        args_label.setStyleSheet("color: #444;")
        layout.addWidget(args_label, 1)
        
        # Risk badge
        risk_colors = {"low": "#28a745", "medium": "#ffc107", "high": "#dc3545"}
        color = risk_colors.get(self.risk.lower(), "#888")
        risk_label = QLabel(self.risk.upper())
        risk_label.setFixedWidth(60)
        risk_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        risk_label.setStyleSheet(f"background: {color}; color: white; border-radius: 3px; padding: 2px;")
        layout.addWidget(risk_label)
        
        # Status indicator
        self.status_label = QLabel("⏳")
        self.status_label.setFixedWidth(25)
        layout.addWidget(self.status_label)
    
    def set_result(self, success: bool, message: str = ""):
        """Update the step result."""
        if success:
            self.status_label.setText("✅")
            self.status_label.setToolTip(message)
        else:
            self.status_label.setText("❌")
            self.status_label.setToolTip(message)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.store = SettingsStore()
        self._process: Optional[QProcess] = None
        self._step_items: dict[int, StepItem] = {}
        self._current_step = 0
        self._output_buffer = ""
        
        self._setup_ui()
        self._connect_signals()
        self._update_status_bar()
    
    def _setup_ui(self):
        """Set up the main window UI."""
        self.setWindowTitle("Agentic Browser")
        self.setMinimumSize(900, 700)
        
        # Load window size from settings
        settings = self.store.settings
        self.resize(settings.window_width, settings.window_height)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # Header with goal input
        header_layout = QHBoxLayout()
        
        goal_label = QLabel("Goal:")
        goal_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        header_layout.addWidget(goal_label)
        
        self.goal_edit = QLineEdit()
        self.goal_edit.setPlaceholderText("Enter your goal, e.g., 'Open example.com and tell me the title'")
        self.goal_edit.setStyleSheet("font-size: 14px; padding: 8px;")
        header_layout.addWidget(self.goal_edit, 1)
        
        self.run_btn = QPushButton("▶ Run")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background: #28a745;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 8px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #218838;
            }
            QPushButton:disabled {
                background: #6c757d;
            }
        """)
        header_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton("⬛ Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: #dc3545;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 8px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #c82333;
            }
            QPushButton:disabled {
                background: #6c757d;
            }
        """)
        header_layout.addWidget(self.stop_btn)
        
        self.settings_btn = QPushButton("⚙ Settings")
        self.settings_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 8px 16px;
            }
        """)
        header_layout.addWidget(self.settings_btn)
        
        layout.addLayout(header_layout)
        
        # Splitter for log and result
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Step log
        log_frame = QFrame()
        log_frame.setStyleSheet("background: white; border: 1px solid #ddd; border-radius: 4px;")
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(0, 0, 0, 0)
        
        log_header = QLabel("Step Log")
        log_header.setStyleSheet("font-weight: bold; padding: 8px; background: #f8f9fa; border-bottom: 1px solid #ddd;")
        log_layout.addWidget(log_header)
        
        self.step_list = QListWidget()
        self.step_list.setStyleSheet("border: none;")
        log_layout.addWidget(self.step_list)
        
        splitter.addWidget(log_frame)
        
        # Result area
        result_frame = QFrame()
        result_frame.setStyleSheet("background: white; border: 1px solid #ddd; border-radius: 4px;")
        result_layout = QVBoxLayout(result_frame)
        result_layout.setContentsMargins(0, 0, 0, 0)
        
        result_header = QLabel("Output")
        result_header.setStyleSheet("font-weight: bold; padding: 8px; background: #f8f9fa; border-bottom: 1px solid #ddd;")
        result_layout.addWidget(result_header)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("border: none; font-size: 13px; padding: 8px; font-family: monospace;")
        self.result_text.setPlaceholderText("Output will appear here...")
        result_layout.addWidget(self.result_text)
        
        splitter.addWidget(result_frame)
        splitter.setSizes([400, 200])
        
        layout.addWidget(splitter, 1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.provider_label = QLabel()
        self.status_bar.addPermanentWidget(self.provider_label)
        
        self.step_count_label = QLabel("Steps: 0")
        self.status_bar.addPermanentWidget(self.step_count_label)
    
    def _connect_signals(self):
        """Connect UI signals."""
        self.run_btn.clicked.connect(self._on_run)
        self.stop_btn.clicked.connect(self._on_stop)
        self.settings_btn.clicked.connect(self._on_settings)
        self.goal_edit.returnPressed.connect(self._on_run)
    
    def _update_status_bar(self):
        """Update the status bar with current settings."""
        settings = self.store.settings
        try:
            provider = Provider(settings.provider)
            provider_name = PROVIDER_DISPLAY_NAMES.get(provider, settings.provider)
        except ValueError:
            provider_name = settings.provider
        
        model = settings.model or "default"
        self.provider_label.setText(f"Provider: {provider_name} | Model: {model}")
    
    def _on_run(self):
        """Start the agent as a subprocess."""
        goal = self.goal_edit.text().strip()
        if not goal:
            QMessageBox.warning(self, "No Goal", "Please enter a goal first.")
            self.goal_edit.setFocus()
            return
        
        # Validate settings
        settings = self.store.settings
        provider_config = settings.get_provider_config()
        is_valid, error = provider_config.validate()
        
        if not is_valid:
            result = QMessageBox.question(
                self,
                "Configuration Required",
                f"{error}\n\nWould you like to open settings?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if result == QMessageBox.StandardButton.Yes:
                self._on_settings()
            return
        
        # Clear previous run
        self.step_list.clear()
        self._step_items.clear()
        self._current_step = 0
        self.result_text.clear()
        self.step_count_label.setText("Steps: 0")
        self._output_buffer = ""
        
        # Update UI state
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.goal_edit.setEnabled(False)
        self.settings_btn.setEnabled(False)
        
        # Build command
        cmd = [
            sys.executable, "-m", "agentic_browser", "run",
            goal,
            "--profile", settings.profile_name,
            "--max-steps", str(settings.max_steps),
            "--model-endpoint", provider_config.endpoint,
            "--model", provider_config.effective_model,
            "--auto-approve",  # Always auto-approve in GUI mode for now
        ]
        
        if settings.headless:
            cmd.append("--headless")
        
        # Add API key as environment variable
        env = dict(subprocess.os.environ)
        if provider_config.api_key:
            env["AGENTIC_BROWSER_API_KEY"] = provider_config.api_key
        
        # Start process
        self._process = QProcess(self)
        self._process.setProcessEnvironment(self._create_env(env))
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.readyReadStandardError.connect(self._on_stderr)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)
        
        # Start the command
        self._process.start(cmd[0], cmd[1:])
        
        self.status_bar.showMessage("Agent running...")
    
    def _create_env(self, env_dict: dict) -> "QProcessEnvironment":
        """Create QProcessEnvironment from dict."""
        from PySide6.QtCore import QProcessEnvironment
        qenv = QProcessEnvironment.systemEnvironment()
        for key, value in env_dict.items():
            qenv.insert(key, value)
        return qenv
    
    def _on_stop(self):
        """Stop the agent process."""
        if self._process:
            self._process.terminate()
            # Give it a moment to terminate gracefully
            QTimer.singleShot(2000, self._force_kill)
            self.status_bar.showMessage("Stopping agent...")
    
    def _force_kill(self):
        """Force kill the process if still running."""
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()
    
    def _on_stdout(self):
        """Handle stdout from the process."""
        if not self._process:
            return
        
        data = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self._output_buffer += data
        
        # Parse lines for step information
        lines = self._output_buffer.split("\n")
        self._output_buffer = lines[-1]  # Keep incomplete line
        
        for line in lines[:-1]:
            self._parse_output_line(line)
    
    def _on_stderr(self):
        """Handle stderr from the process."""
        if not self._process:
            return
        
        data = self._process.readAllStandardError().data().decode("utf-8", errors="replace")
        self.result_text.append(f"[stderr] {data}")
    
    def _parse_output_line(self, line: str):
        """Parse a line of output to extract step info."""
        line = line.strip()
        if not line:
            return
        
        # Append to result text
        self.result_text.append(line)
        self.result_text.ensureCursorVisible()
        
        # Try to detect step patterns from rich output
        # Look for patterns like "Step 1:" or action names
        if "Step" in line and ":" in line:
            try:
                # Extract step number
                import re
                match = re.search(r"Step\s*(\d+)", line)
                if match:
                    self._current_step = int(match.group(1))
                    self.step_count_label.setText(f"Steps: {self._current_step}")
            except:
                pass
        
        # Look for action patterns
        action_patterns = ["goto", "click", "type", "press", "scroll", "extract", "done"]
        line_lower = line.lower()
        for action in action_patterns:
            if f"action: {action}" in line_lower or f"→ {action}" in line_lower:
                self._add_step_item(self._current_step or 1, action, line, "low")
                break
    
    def _add_step_item(self, step: int, action: str, details: str, risk: str):
        """Add a step to the log."""
        if step in self._step_items:
            return  # Already added
            
        item = StepItem(step, action, details, risk)
        list_item = QListWidgetItem()
        list_item.setSizeHint(item.sizeHint())
        
        self.step_list.addItem(list_item)
        self.step_list.setItemWidget(list_item, item)
        self.step_list.scrollToBottom()
        
        self._step_items[step] = item
    
    def _on_finished(self, exit_code: int, exit_status):
        """Handle process finished."""
        # Reset UI state
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.goal_edit.setEnabled(True)
        self.settings_btn.setEnabled(True)
        
        if exit_code == 0:
            self.status_bar.showMessage("Agent completed successfully!", 5000)
        else:
            self.status_bar.showMessage(f"Agent finished with exit code {exit_code}", 5000)
        
        self._process = None
    
    def _on_error(self, error):
        """Handle process error."""
        self.result_text.append(f"\n[Error] Process error: {error}")
        self.status_bar.showMessage("Agent error occurred", 5000)
    
    def _on_settings(self):
        """Open settings dialog."""
        dialog = SettingsDialog(self)
        if dialog.exec():
            self._update_status_bar()
    
    def closeEvent(self, event):
        """Handle window close."""
        # Save window size
        self.store.update(
            window_width=self.width(),
            window_height=self.height(),
        )
        
        # Stop process if running
        if self._process and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.terminate()
            self._process.waitForFinished(3000)
            if self._process.state() != QProcess.ProcessState.NotRunning:
                self._process.kill()
        
        event.accept()


def run_gui():
    """Run the GUI application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Agentic Browser")
    app.setStyle("Fusion")
    
    # Apply styling
    app.setStyleSheet("""
        QMainWindow {
            background: #f5f5f5;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-top: 12px;
            padding-top: 12px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
    """)
    
    window = MainWindow()
    window.show()
    
    return app.exec()
