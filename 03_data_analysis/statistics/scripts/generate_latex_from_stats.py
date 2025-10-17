#!/usr/bin/env python3
"""
Enhanced LaTeX table generator supporting JSON configuration files
"""

import argparse
import sys
import os
import json

# Add the tables directory to Python path so we can import the controller
sys.path.append(os.path.join(os.path.dirname(__file__), 'tables'))

from table_controller import TableGenerationController

def load_config(config_file):
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX tables from statistics data using various input methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input Methods:
  1. JSON config file (recommended):
     python generate_latex_from_stats.py --config config/tables/descriptive.json
  
  2. Single table with parameters:
     python generate_latex_from_stats.py --data-file stats.json --strategy descriptive
  
  3. Batch generation:
     python generate_latex_from_stats.py --batch-config config/tables/batch.json

Examples:
  # Using JSON config file
  python generate_latex_from_stats.py --config config/tables/descriptive.json
  
  # Single table with inline parameters  
  python generate_latex_from_stats.py --data-file descriptive_stats.json --strategy descriptive --output paper/descriptives.tex
  
  # Batch processing
  python generate_latex_from_stats.py --batch-config config/tables/paper_tables.json
        """
    )
    
    # Input method groups (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--config', help='JSON configuration file for single table')
    input_group.add_argument('--batch-config', help='JSON configuration file for batch table generation')
    input_group.add_argument('--data-file', help='Input data file for single table')
    
    # Parameters for single table mode (only used with --data-file)
    parser.add_argument('--strategy', help='Table generation strategy name (required with --data-file)')
    parser.add_argument('--output', '-o', help='Output LaTeX file path')
    parser.add_argument('--caption', help='Table caption')
    parser.add_argument('--label', help='Table label for referencing')
    parser.add_argument('--style', default='academic', help='Table style')
    
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
    
    # Initialize controller
    controller = TableGenerationController(args.stats_dir, args.tables_dir)
    
    # Handle information commands first
    if args.list_strategies:
        strategies = controller.list_available_strategies()
        print("Available Table Generation Strategies:")
        for name, desc in strategies.items():
            print(f"  {name}: {desc}")
        return
    
    if args.preview_strategy:
        preview = controller.preview_strategy(args.preview_strategy)
        print(preview)
        return
    
    # Handle different input methods
    if args.config:
        # Single table from JSON config file
        print(f"📁 Using configuration file: {args.config}")
        config = load_config(args.config)
        result = controller.generate_table(**config)
        
    elif args.batch_config:
        # Batch generation from JSON config file
        print(f"📚 Batch generating tables from: {args.batch_config}")
        results = controller.batch_generate_tables(args.batch_config)
        
        # Summary
        successful = len([r for r in results if r['success']])
        failed = len([r for r in results if not r['success']])
        
        print(f"✅ Batch generation completed: {successful} successful, {failed} failed")
        
        if failed > 0:
            print("\nFailed tables:")
            for result in results:
                if not result['success']:
                    print(f"  ❌ {result['config'].get('data_file', 'Unknown')}: {result['error']}")
        
        return
    
    elif args.data_file:
        # Single table with command-line parameters
        if not args.strategy:
            print("❌ Error: --strategy is required when using --data-file")
            sys.exit(1)
            
        result = controller.generate_table(
            data_file=args.data_file,
            strategy_name=args.strategy,
            output_file=args.output,
            caption=args.caption,
            label=args.label,
            style=args.style
        )
    
    # Output results for single table generation
    if 'result' in locals() and result['success']:
        print(f"✅ LaTeX table generated successfully!")
        print(f"   📊 Input:  {result['input_data']}")
        print(f"   📄 Output: {result['output_table']}")
        print(f"   🎯 Strategy: {result['strategy_used']}")
        if 'caption' in result:
            print(f"   📝 Caption: {result['caption']}")
    elif 'result' in locals():
        print(f"❌ Table generation failed")
        if 'error' in result:
            print(f"   💥 Error: {result['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()