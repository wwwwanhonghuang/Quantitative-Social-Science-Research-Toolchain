#!/usr/bin/env python3
import argparse
import os
from gliner import GLiNER

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--save-path', required=True)
    args = parser.parse_args()
    
    print(f"Downloading {args.model_name}...")
    model = GLiNER.from_pretrained(args.model_name)
    model.save_pretrained(args.save_path)
    print(f"✓ Model saved to {args.save_path}")

if __name__ == "__main__":
    main()