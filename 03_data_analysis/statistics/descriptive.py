#!/usr/bin/env python3
from abc import ABC, abstractmethod

class LatexTableGenerator(ABC):
    """Abstract base class for LaTeX table generators"""
    
    def __init__(self, style='academic'):
        self.style = style
    
    @abstractmethod
    def generate_table(self, data, **kwargs):
        """Generate LaTeX table from data"""
        pass
    
    def _generate_table_header(self, caption, label=None):
        """Generate common table header"""
        label_str = f"\\label{{{label}}}" if label else ""
        return f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
{label_str}
"""
    
    def _generate_table_footer(self):
        """Generate common table footer"""
        return "\\end{table}"
    

