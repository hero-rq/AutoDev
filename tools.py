import time
import os
import re
import io
import sys
import traceback
import numpy as np
import concurrent.futures
from pypdf import PdfReader
from datasets import load_dataset, load_dataset_builder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from semanticscholar import SemanticScholar

class DatasetSearcher:
    def __init__(self, min_likes=3, min_downloads=50):
        self.min_likes = min_likes
        self.min_downloads = min_downloads
        self.datasets = load_dataset("nkasmanoff/huggingface-datasets")["train"]
        self.filtered_datasets = self._filter_datasets()
        self.vectorizer = TfidfVectorizer()
        self.dataset_vectors = self.vectorizer.fit_transform([d["description"] for d in self.filtered_datasets])

    def _filter_datasets(self):
        return [
            d for d in self.datasets 
            if d['likes'] and d['likes'] >= self.min_likes and d['downloads'] and d['downloads'] >= self.min_downloads
        ]

    def search(self, query, top_n=10):
        query_vector = self.vectorizer.transform([query])
        scores = linear_kernel(query_vector, self.dataset_vectors).flatten()
        top_indices = scores.argsort()[-top_n:][::-1]
        return [self.filtered_datasets[i] for i in top_indices]

class PaperSearcher:
    def __init__(self):
        self.scholar = SemanticScholar(retry=False)

    def search_papers(self, query, top_n=10):
        return self.scholar.search_paper(query, limit=top_n, min_citation_count=3, open_access_pdf=True)

class CodeExecutor:
    @staticmethod
    def execute(code, timeout=60, max_output_len=1000):
        output_capture = io.StringIO()
        sys.stdout = output_capture

        def run():
            try:
                exec(code, {})
            except Exception as e:
                output_capture.write(f"[ERROR]: {str(e)}\n")
                traceback.print_exc(file=output_capture)

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run)
                future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return "[ERROR]: Execution timed out. Reduce code complexity."
        finally:
            sys.stdout = sys.__stdout__

        return output_capture.getvalue()[:max_output_len]
