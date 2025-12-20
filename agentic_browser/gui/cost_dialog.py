from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget, 
    QTableWidgetItem, QHeaderView, QLabel, QMessageBox, QInputDialog
)
from PySide6.QtCore import Qt
from ..cost import load_prices, save_prices, calculate_cost

class CostDialog(QDialog):
    """Dialog for managing model costs."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cost Settings")
        self.resize(600, 400)
        
        self.layout = QVBoxLayout(self)
        
        # Explanation
        info = QLabel("Set estimated costs per 1M tokens (USD).")
        info.setStyleSheet("color: #888; font-style: italic;")
        self.layout.addWidget(info)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Model Name", "Input Cost ($)", "Output Cost ($)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.layout.addWidget(self.table)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        add_btn = QPushButton("Add Model")
        add_btn.clicked.connect(self.add_model)
        
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_model)
        
        save_btn = QPushButton("Save && Close")
        save_btn.clicked.connect(self.save_and_close)
        save_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold;")
        
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(save_btn)
        
        self.layout.addLayout(btn_layout)
        
        self.load_data()
        
    def load_data(self):
        """Load prices into table."""
        prices = load_prices()
        self.table.setRowCount(len(prices))
        
        for i, (model, (inp, out)) in enumerate(sorted(prices.items())):
            self.table.setItem(i, 0, QTableWidgetItem(model))
            self.table.setItem(i, 1, QTableWidgetItem(str(inp)))
            self.table.setItem(i, 2, QTableWidgetItem(str(out)))
            
    def add_model(self):
        """Add a new row."""
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem("new-model"))
        self.table.setItem(row, 1, QTableWidgetItem("1.00"))
        self.table.setItem(row, 2, QTableWidgetItem("1.00"))
        self.table.editItem(self.table.item(row, 0))
        
    def remove_model(self):
        """Remove selected row."""
        current = self.table.currentRow()
        if current >= 0:
            self.table.removeRow(current)
            
    def save_and_close(self):
        """Save table data to disk."""
        prices = {}
        try:
            for row in range(self.table.rowCount()):
                model_item = self.table.item(row, 0)
                input_item = self.table.item(row, 1)
                output_item = self.table.item(row, 2)
                
                if model_item and input_item and output_item:
                    model = model_item.text().strip()
                    if not model:
                        continue
                        
                    inp = float(input_item.text())
                    out = float(output_item.text())
                    prices[model] = (inp, out)
            
            save_prices(prices)
            self.accept()
            
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Costs must be valid numbers.")
