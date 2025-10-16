#!/usr/bin/env python3
import sys
from transformers import T5ForConditionalGeneration, T5Tokenizer

def summarize_file(input_path, output_path, chunk_size=400):
    # Load model & tokenizer
    tokenizer = T5Tokenizer.from_pretrained("models/t5-small")
    model = T5ForConditionalGeneration.from_pretrained("models/t5-small")

    # Read input
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Simple chunking
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    summaries = []
    for chunk in chunks:
        inputs = tokenizer("summarize: " + chunk,
                           return_tensors="pt",
                           max_length=512,
                           truncation=True)
        summary_ids = model.generate(**inputs, max_length=150)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    # Combine chunk summaries
    final_summary = "\n".join(summaries)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_summary)
    print(f"✓ Summary saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python summarize_paper.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    summarize_file(input_file, output_file)
