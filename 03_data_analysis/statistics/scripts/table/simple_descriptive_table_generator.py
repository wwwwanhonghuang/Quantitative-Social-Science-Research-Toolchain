#!/usr/bin/env python3
from latex_table_generator import LatexTableGenerator

class SimpleDescriptiveTableGenerator(LatexTableGenerator):
    """Generate descriptive statistics tables"""
    
    def generate_table(self, stats_data, caption="Descriptive Statistics", label=None):
        """Generate descriptive statistics table"""
        
        header = self._generate_table_header(caption, label)
        
        table_content = """\\begin{tabular}{lrrrrrr}
\\toprule
Variable & N & Mean & Std. Dev. & Min & Median & Max \\\\
\\midrule
"""
        
        for var, stats in stats_data.items():
            table_content += f"{var} & {stats['n']} & {stats['mean']:.3f} & {stats['std']:.3f} & {stats['min']:.3f} & {stats['median']:.3f} & {stats['max']:.3f} \\\\\n"
        
        table_content += """\\bottomrule
\\end{tabular}"""
        
        footer = self._generate_table_footer()
        
        return header + table_content + footer
    
    def generate_minimal_table(self, stats_data, caption="Key Variables"):
        """Alternative implementation - minimal version"""
        header = self._generate_table_header(caption)
        
        table_content = """\\begin{tabular}{lrrr}
\\toprule
Variable & Mean & Std. Dev. & N \\\\
\\midrule
"""
        
        for var, stats in stats_data.items():
            table_content += f"{var} & {stats['mean']:.3f} & {stats['std']:.3f} & {stats['n']} \\\\\n"
        
        table_content += """\\bottomrule
\\end{tabular}"""
        
        footer = self._generate_table_footer()
        
        return header + table_content + footer