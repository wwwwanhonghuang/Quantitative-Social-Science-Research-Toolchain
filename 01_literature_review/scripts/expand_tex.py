#!/usr/bin/env python3
import os
import re
import sys

def expand_tex(main_tex_path, output_path):
    base_dir = os.path.dirname(main_tex_path)
    expanded_lines = []

    input_pattern = re.compile(r'\\(input|include)\{(.+?)\}')

    with open(main_tex_path, "r", encoding="utf-8") as f:
        for line in f:
            match = input_pattern.search(line)
            if match:
                cmd, rel_path = match.groups()
                rel_path = rel_path if rel_path.endswith(".tex") else rel_path + ".tex"
                section_path = os.path.join(base_dir, rel_path)
                if os.path.exists(section_path):
                    with open(section_path, "r", encoding="utf-8") as s:
                        expanded_lines.append(f"% Begin {rel_path}\n")
                        expanded_lines.extend(s.readlines())
                        expanded_lines.append(f"% End {rel_path}\n")
                else:
                    expanded_lines.append(f"% File {rel_path} not found!\n")
            else:
                expanded_lines.append(line)

    with open(output_path, "w", encoding="utf-8") as out:
        out.writelines(expanded_lines)

    print(f"✓ Expanded LaTeX saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python expand_tex.py <main.tex> <output.txt>")
        sys.exit(1)
    expand_tex(sys.argv[1], sys.argv[2])
