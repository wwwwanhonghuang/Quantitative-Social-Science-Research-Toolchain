#!/usr/bin/env python3
"""
Table Generation Controller - Facade for generating LaTeX tables from saved data
"""

import argparse
import json
import os
from pathlib import Path
from table_generator_factory import TableGeneratorFactory

class TableGenerationController:
    """Facade/Controller for table generation operations"""
    
    def __init__(self, stats_base_dir="results/stats", tables_base_dir="results/tables"):
        self.stats_base_dir = Path(stats_base_dir)
        self.tables_base_dir = Path(tables_base_dir)
        self.tables_base_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_table(self, data_file, strategy_name, output_file=None, **kwargs):
        """
        Generate table using specified strategy
        
        Args:
            data_file: Path to JSON data file or just filename (searches in stats_base_dir)
            strategy_name: Name of table generation strategy (e.g., 'descriptive', 'regression')
            output_file: Output path for LaTeX file (optional)
            **kwargs: Additional arguments (caption, label, style, etc.)
        """
        # Resolve data file path
        data_path = self._resolve_data_file(data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data
        data = self._load_data(data_path)
        
        # Create generator
        generator = TableGeneratorFactory.create_generator(strategy_name, kwargs.get('style', 'academic'))
        
        # Generate table
        caption = kwargs.get('caption', f"{strategy_name.title()} Results")
        label = kwargs.get('label')
        latex_table = generator.generate_table(data, caption=caption, label=label)
        
        # Determine output path
        if output_file is None:
            output_path = self._generate_output_path(data_path, strategy_name)
        else:
            output_path = Path(output_file)
        
        # Save table
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex_table)
        
        return {
            'success': True,
            'input_data': str(data_path),
            'output_table': str(output_path),
            'strategy_used': strategy_name,
            'caption': caption
        }
    
    def batch_generate_tables(self, config_file):
        """Generate multiple tables from configuration file"""
        with open(config_file, 'r') as f:
            configs = json.load(f)
        
        results = []
        for config in configs:
            try:
                result = self.generate_table(**config)
                results.append(result)
                print(f"✓ Generated: {result['output_table']}")
            except Exception as e:
                results.append({
                    'success': False,
                    'error': str(e),
                    'config': config
                })
                print(f"❌ Failed: {config['data_file']} -> {e}")
        
        return results
    
    def _resolve_data_file(self, data_file):
        """Resolve data file path"""
        data_path = Path(data_file)
        if not data_path.is_absolute():
            # Try in stats base directory
            stats_path = self.stats_base_dir / data_path
            if stats_path.exists():
                return stats_path
            # Try with .json extension
            json_path = self.stats_base_dir / f"{data_path.stem}.json"
            if json_path.exists():
                return json_path
        return data_path
    
    def _load_data(self, data_path):
        """Load data from JSON file"""
        with open(data_path, 'r') as f:
            return json.load(f)
    
    def _generate_output_path(self, data_path, strategy_name):
        """Generate output path based on input data and strategy"""
        stem = data_path.stem.replace('_stats', '').replace('_results', '')
        filename = f"{stem}_{strategy_name}_table.tex"
        return self.tables_base_dir / filename
    
    def list_available_strategies(self):
        """List all available table generation strategies"""
        return TableGeneratorFactory.list_available_generators()
    
    def preview_strategy(self, strategy_name):
        """Preview what a strategy can do"""
        strategies = self.list_available_strategies()
        if strategy_name not in strategies:
            return f"Unknown strategy: {strategy_name}"
        
        # Could add more detailed preview information here
        return f"Strategy: {strategy_name}\nDescription: {strategies[strategy_name]}"

def main():
    """Command-line interface for table generation controller"""
    parser = argparse.ArgumentParser(description='Table Generation Controller')
    
    # Single table generation
    parser.add_argument('--data-file', help='Input data file (JSON)')
    parser.add_argument('--strategy', help='Table generation strategy name')
    parser.add_argument('--output', help='Output LaTeX file path')
    parser.add_argument('--caption', help='Table caption')
    parser.add_argument('--label', help='Table label for referencing')
    parser.add_argument('--style', default='academic', help='Table style')
    
    # Batch operations
    parser.add_argument('--batch-config', help='JSON config file for batch generation')
    
    # Information commands
    parser.add_argument('--list-strategies', action='store_true', 
                       help='List available table generation strategies')
    parser.add_argument('--preview-strategy', help='Preview a specific strategy')
    
    # Configuration
    parser.add_argument('--stats-dir', default='results/stats', 
                       help='Base directory for statistics files')
    parser.add_argument('--tables-dir', default='results/tables', 
                       help='Base directory for output tables')
    
    args = parser.parse_args()
    
    controller = TableGenerationController(args.stats_dir, args.tables_dir)
    
    # Handle different operations
    if args.list_strategies:
        strategies = controller.list_available_strategies()
        print("Available Table Generation Strategies:")
        for name, desc in strategies.items():
            print(f"  {name}: {desc}")
    
    elif args.preview_strategy:
        preview = controller.preview_strategy(args.preview_strategy)
        print(preview)
    
    elif args.batch_config:
        print(f"Batch generating tables from: {args.batch_config}")
        results = controller.batch_generate_tables(args.batch_config)
        print(f"Completed: {len([r for r in results if r['success']])} successful, "
              f"{len([r for r in results if not r['success']])} failed")
    
    elif args.data_file and args.strategy:
        result = controller.generate_table(
            data_file=args.data_file,
            strategy_name=args.strategy,
            output_file=args.output,
            caption=args.caption,
            label=args.label,
            style=args.style
        )
        
        if result['success']:
            print(f"✓ Table generated successfully!")
            print(f"   Input:  {result['input_data']}")
            print(f"   Output: {result['output_table']}")
            print(f"   Strategy: {result['strategy_used']}")
            print(f"   Caption: {result['caption']}")
        else:
            print(f"❌ Table generation failed")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()