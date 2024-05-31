from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QComboBox, QLabel, QPushButton
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import script
import plotly.io as pio
from fpdf import FPDF
import tempfile
import sys

df = pd.read_excel("/Users/sarthakmishra/Documents/Code/Lechler_3D_Nozzel/results/111.xlsx")


class CustomComboBox(QComboBox):
    def showPopup(self):
        self.view().setMinimumWidth(self.view().sizeHintForColumn(0))
        super().showPopup()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("GUI APP")

        # Create a central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create a layout
        layout = QVBoxLayout(central_widget)
        color_scales = ["Blackbody", "Bluered", "Blues", "Cividis", "Earth", "Electric", "Greens", "Greys", "Hot", "Jet", "Picnic", "Portland", "Rainbow", "RdBu", "Reds", "Viridis", "YlGnBu", "YlOrRd"]

        # Create a label for the dropdown
        self.label = QLabel("Colorscale:", self)
        layout.addWidget(self.label)

        # Create a dropdown button
        self.dropdown = CustomComboBox(self)
        for color_scale in color_scales:
            self.dropdown.addItem(color_scale)
        self.dropdown.currentIndexChanged.connect(self.update_figure)
        layout.addWidget(self.dropdown)

        # Create a QWebEngineView
        self.web_view = QWebEngineView()

        # Add the QWebEngineView to the layout
        layout.addWidget(self.web_view)

        # Create a button for generating and saving a 2D heatmap
        self.button = QPushButton("Generate 2D Heatmap", self)
        self.button.clicked.connect(self.generate_heatmap)
        layout.addWidget(self.button)

        # Create a button for changing the view figure to the discrete distribution
        self.discrete_button = QPushButton("Show Discrete Distribution", self)
        self.discrete_button.clicked.connect(self.show_discrete_distribution)
        layout.addWidget(self.discrete_button)
        
        # Add this in your __init__ method, after the existing button definitions
        self.distribution_button = QPushButton("Show 3D Distribution", self)
        self.distribution_button.clicked.connect(self.update_figure)
        layout.addWidget(self.distribution_button)
        
        # Create a button for generating the PDF report
        self.report_button = QPushButton("Generate Report", self)
        self.report_button.clicked.connect(self.generate_report)
        layout.addWidget(self.report_button)

        # Update the figure in the QWebEngineView
        self.update_figure()

    def generate_heatmap(self):
        # Code for generating and saving a 2D heatmap goes here
        heatmap_fig = script.csv_to_2D_Heatmap(df)

        # Create a temporary HTML file with the Plotly figure
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        temp_file.close()
        pio.write_html(heatmap_fig, file=temp_file.name)

        # Load the HTML file into the existing QWebEngineView
        self.web_view.load(QUrl.fromLocalFile(temp_file.name))
        
        
    def show_discrete_distribution(self):
        # Generate the discrete distribution
        discrete_fig = script.csv_to_discrete_distribution(20 , df )
        
        # Create a temporary HTML file with the Plotly figure
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        temp_file.close()
        pio.write_html(discrete_fig, file=temp_file.name)

        # Load the HTML file into the QWebEngineView
        self.web_view.load(QUrl.fromLocalFile(temp_file.name))
        
        
    def generate_report(self):
        # Generate the 3D distribution and save it as an image
        fig_3d = script.csv_to_3D_Distribution(20,df, "Jet")
        pio.write_image(fig_3d, "3d_plot.png")

        # Generate the 2D heatmap and save it as an image
        fig_2d = script.csv_to_2D_Heatmap(df, "Jet")
        pio.write_image(fig_2d, "2d_heatmap.png")

        # Create a new PDF file
        pdf = FPDF()

        # Add the 3D distribution to the PDF file
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        #pdf.cell(200, 10, txt="3D Distribution", ln=True, align='C')
        pdf.image("3d_plot.png", x=50, y=50, w=100)

        # Add the 2D heatmap to the PDF file
        pdf.set_font("Arial", size=12)
        #pdf.cell(200, 10, txt="2D Heatmap", ln=True, align='C')
        pdf.image("2d_heatmap.png", x=50, y=150, w=100)

        # Save the PDF file
        pdf.output("/Users/sarthakmishra/Documents/Code/Lechler_3D_Nozzel/report.pdf", "F")

    def update_figure(self):
        # Get the selected colorscale
        colorscale = self.dropdown.currentText()

        # Create a Plotly figure with the selected colorscale
        fig = script.csv_to_3D_Distribution(20, df, colorscale=colorscale)

        # Create a temporary HTML file with the Plotly figure
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        temp_file.close()
        pio.write_html(fig, file=temp_file.name)

        # Load the HTML file into the QWebEngineView
        self.web_view.load(QUrl.fromLocalFile(temp_file.name))
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())