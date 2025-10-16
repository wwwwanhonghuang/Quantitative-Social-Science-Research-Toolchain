#!/usr/bin/env python3
import os
import sys
from gliner import GLiNER

def main():
    model_path = 'models/gliner-multi'
    
    if not os.path.exists(model_path):
        print('❌ Model not found. Run make download-model first')
        sys.exit(1)
    
    try:
        model = GLiNER.from_pretrained(model_path, load_tokenizer=False)
        print('✅ Model loaded successfully')
    except Exception as e:
        print(f'❌ Error loading model: {e}')
        sys.exit(1)

if __name__ == "__main__":
    main()