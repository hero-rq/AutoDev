import os
import re
import shutil
import tiktoken
import subprocess
import io
import sys
import traceback
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def compile_latex(latex_code, compile=True, output_filename="output.pdf", timeout=30):
    latex_code = latex_code.replace(
    r"\documentclass{article}",
    "\\documentclass{article}\n\\usepackage{amsmath, amssymb, graphicx, hyperref, xcolor, algorithm, algpseudocode}"
)
    dir_path = "research_dir/tex"
    tex_file_path = os.path.join(dir_path, "temp.tex")
    
    with open(tex_file_path, "w") as f:
        f.write(latex_code)

    if not compile:
        return "Compilation successful"

    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "temp.tex"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            cwd=dir_path
        )
        return f"Compilation successful: {result.stdout.decode('utf-8')}"
    except subprocess.TimeoutExpired:
        return "[ERROR]: Compilation timed out."
    except subprocess.CalledProcessError as e:
        return f"[ERROR]: Compilation failed: {e.stderr.decode('utf-8')}"

def count_tokens(messages, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    return sum([len(enc.encode(m["content"])) for m in messages])

def remove_figures():
    for _file in os.listdir("."):
        if _file.startswith("Figure_") and _file.endswith(".png"):
            os.remove(_file)

def remove_directory(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)

def save_to_file(location, filename, data):
    filepath = os.path.join(location, filename)
    with open(filepath, 'w') as f:
        f.write(data)

def clip_tokens(messages, model="gpt-4o", max_tokens=100000):
    enc = tiktoken.encoding_for_model(model)
    total_tokens = sum([len(enc.encode(m["content"])) for m in messages])
    if total_tokens <= max_tokens:
        return messages
    clipped_tokens = enc.encode(" ".join(m["content"] for m in messages))[-max_tokens:]
    clipped_messages = [{"role": m["role"], "content": enc.decode(clipped_tokens)} for m in messages]
    return clipped_messages

def extract_prompt(text, word):
    pattern = rf"```{word}(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return "\n".join(matches).strip()
