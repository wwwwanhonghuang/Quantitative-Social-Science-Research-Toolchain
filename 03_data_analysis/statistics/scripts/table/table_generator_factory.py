#!/usr/bin/env python3
from descriptive_table_generator import DescriptiveTableGenerator
from regression_table_generator import RegressionTableGenerator

class TableGeneratorFactory:
    """Factory for creating table generators"""
    
    @staticmethod
    def create_generator(table_type, style='academic'):
        """Create appropriate table generator"""
        generators = {
            'descriptive': DescriptiveTableGenerator,
            'regression': RegressionTableGenerator,
            # Add new generators here as they are created
        }
        
        generator_class = generators.get(table_type)
        if not generator_class:
            raise ValueError(f"Unknown table type: {table_type}")
        
        return generator_class(style)
    
    @staticmethod
    def list_available_generators():
        """List all available table generator types"""
        return {
            'descriptive': 'Descriptive statistics table',
            'regression': 'Regression results table',
        }