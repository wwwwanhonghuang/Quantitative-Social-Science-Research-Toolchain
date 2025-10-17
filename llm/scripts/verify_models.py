#!/usr/bin/env python3
"""Verify all downloaded models and test inference"""

import os
import sys
import time
from pathlib import Path

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def check_file_exists(path):
    """Check if file/directory exists and return size"""
    if os.path.exists(path):
        if os.path.isdir(path):
            size = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
            return True, size
        else:
            return True, os.path.getsize(path)
    return False, 0

def format_size(bytes):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024.0
    return f"{bytes:.1f}TB"

def verify_sentence_transformer(model_path):
    """Verify Sentence Transformer model"""
    print("1. Sentence Transformer (all-MiniLM-L6-v2)")
    print("   " + "-" * 55)
    
    path = os.path.join(model_path, "sentence-transformer")
    exists, size = check_file_exists(path)
    
    if not exists:
        print(f"   ❌ NOT FOUND: {path}")
        return False
    
    print(f"   ✓ Files found ({format_size(size)})")
    
    try:
        from sentence_transformers import SentenceTransformer
        start = time.time()
        model = SentenceTransformer(path)
        load_time = time.time() - start
        
        # Test inference
        start = time.time()
        embeddings = model.encode(["This is a test sentence."])
        inference_time = time.time() - start
        
        print(f"   ✓ Model loaded in {load_time:.2f}s")
        print(f"   ✓ Test inference: {inference_time:.3f}s")
        print(f"   ✓ Embedding dimension: {len(embeddings[0])}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return False

def verify_gliner(model_path):
    """Verify GLiNER model"""
    print("\n2. GLiNER (multi-v2.1)")
    print("   " + "-" * 55)
    
    # Try multiple possible paths
    possible_paths = [
        os.path.join(model_path, "gliner_multi-v2.1"),
        os.path.join(model_path, "gliner"),
        model_path
    ]
    
    found_path = None
    for path in possible_paths:
        if os.path.exists(path) and any(f.endswith('.bin') or f.endswith('.safetensors') 
                                       for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))):
            found_path = path
            break
    
    if not found_path:
        print(f"   ❌ NOT FOUND in any of:")
        for p in possible_paths:
            print(f"      - {p}")
        return False
    
    exists, size = check_file_exists(found_path)
    print(f"   ✓ Files found ({format_size(size)})")
    print(f"   ✓ Path: {found_path}")
    
    # Check for essential files
    essential_files = ['config.json']
    model_files = [f for f in os.listdir(found_path) if f.endswith(('.bin', '.safetensors'))]
    
    if model_files:
        print(f"   ✓ Model weights: {', '.join(model_files)}")
    else:
        print(f"   ⚠️  No model weights found (.bin or .safetensors)")
    
    for file in essential_files:
        if os.path.exists(os.path.join(found_path, file)):
            print(f"   ✓ {file} found")
        else:
            print(f"   ⚠️  {file} missing")
    
    return True

def verify_summarizer(model_path):
    """Verify Summarizer model"""
    print("\n3. Summarizer (DistilBART-CNN)")
    print("   " + "-" * 55)
    
    path = os.path.join(model_path, "distilbart-cnn")
    exists, size = check_file_exists(path)
    
    if not exists:
        print(f"   ❌ NOT FOUND: {path}")
        return False
    
    print(f"   ✓ Files found ({format_size(size)})")
    
    try:
        from transformers import BartForConditionalGeneration, BartTokenizer
        start = time.time()
        tokenizer = BartTokenizer.from_pretrained(path)
        model = BartForConditionalGeneration.from_pretrained(path)
        load_time = time.time() - start
        
        # Test inference
        test_text = "This is a test article. " * 10
        start = time.time()
        inputs = tokenizer(test_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs["input_ids"], max_length=50, num_beams=2)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        inference_time = time.time() - start
        
        print(f"   ✓ Model loaded in {load_time:.2f}s")
        print(f"   ✓ Test inference: {inference_time:.2f}s")
        print(f"   ✓ Sample output: {summary[:50]}...")
        return True
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return False

def verify_classifier(model_path):
    """Verify Classifier model"""
    print("\n4. Classifier (DistilBERT)")
    print("   " + "-" * 55)
    
    path = os.path.join(model_path, "distilbert")
    exists, size = check_file_exists(path)
    
    if not exists:
        print(f"   ❌ NOT FOUND: {path}")
        return False
    
    print(f"   ✓ Files found ({format_size(size)})")
    
    try:
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        start = time.time()
        tokenizer = DistilBertTokenizer.from_pretrained(path)
        model = DistilBertForSequenceClassification.from_pretrained(path)
        load_time = time.time() - start
        
        # Test inference
        test_text = "This is a test sentence."
        start = time.time()
        inputs = tokenizer(test_text, return_tensors="pt")
        outputs = model(**inputs)
        inference_time = time.time() - start
        
        print(f"   ✓ Model loaded in {load_time:.2f}s")
        print(f"   ✓ Test inference: {inference_time:.3f}s")
        print(f"   ✓ Number of labels: {model.config.num_labels}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return False

def main():
    print_section("Model Verification")
    
    # Get model path from environment or use default
    model_path = os.environ.get('MODEL_PATH', 'models')
    
    if not os.path.exists(model_path):
        print(f"❌ Models directory not found: {model_path}")
        print(f"\nPlease run: make download-all-models")
        sys.exit(1)
    
    print(f"Model directory: {model_path}\n")
    
    results = {
        "Sentence Transformer": verify_sentence_transformer(model_path),
        "GLiNER": verify_gliner(model_path),
        "Summarizer": verify_summarizer(model_path),
        "Classifier": verify_classifier(model_path),
    }
    
    # Summary
    print_section("Verification Summary")
    
    total = len(results)
    passed = sum(results.values())
    
    for name, status in results.items():
        status_icon = "✓" if status else "❌"
        print(f"   {status_icon} {name}")
    
    print(f"\n   Result: {passed}/{total} models verified")
    
    if passed == total:
        print("\n   🎉 All models verified successfully!")
        sys.exit(0)
    else:
        print(f"\n   ⚠️  {total - passed} model(s) failed verification")
        print("\n   Run 'make download-all-models' to download missing models")
        sys.exit(1)

if __name__ == "__main__":
    main()