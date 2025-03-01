import time
import os
import json
import re
import random
import logging
from copy import deepcopy
from pathlib import Path
from contextlib import contextmanager
import sys

from inference import query_model

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

logging.basicConfig(level=logging.WARNING)

class AutomatedCodeRefinement:
    def __init__(self, model, openai_api_key=None, max_attempts=3):
        self.model = model
        self.openai_api_key = openai_api_key
        self.max_attempts = max_attempts

    def refine_code(self, code_snippet, error_message):
        system_prompt = """
        You are an AI-powered code refinement agent.
        Your goal is to analyze the provided code and error message,
        then generate an improved version of the code that resolves the issue.
        Ensure correctness, readability, and maintainability.
        Output the refined code wrapped in ```python.
        """
        for _ in range(self.max_attempts):
            response = query_model(
                model_str=self.model,
                system_prompt=system_prompt,
                prompt=f"Error: {error_message}\n\nCode:\n{code_snippet}",
                openai_api_key=self.openai_api_key
            )
            fixed_code = self.extract_code(response)
            if fixed_code:
                return fixed_code
        return None

    @staticmethod
    def extract_code(response):
        match = re.search(r"```python(.*?)```", response, re.DOTALL)
        return match.group(1).strip() if match else None

class MLESolver:
    def __init__(self, model, openai_api_key=None, project_description="", max_steps=10):
        self.model = model
        self.openai_api_key = openai_api_key
        self.project_description = project_description
        self.max_steps = max_steps
        self.best_code = None
        self.best_score = 0

    def generate_initial_code(self):
        system_prompt = """
        You are an AI-powered machine learning engineer.
        Your task is to generate a Python script that aligns with the project requirements.
        Ensure correctness, efficiency, and best coding practices.
        Output the script wrapped in ```python.
        """
        response = query_model(
            model_str=self.model,
            system_prompt=system_prompt,
            prompt=self.project_description,
            openai_api_key=self.openai_api_key
        )
        return self.extract_code(response)

    def evaluate_code(self, code_snippet):
        system_prompt = """
        You are an AI-powered reviewer assessing the quality of a machine learning script.
        Provide a score from 0 to 1 based on correctness, efficiency, and alignment with the project description.
        Output only a numeric score.
        """
        response = query_model(
            model_str=self.model,
            system_prompt=system_prompt,
            prompt=f"Code:\n{code_snippet}\n\nProject Description:\n{self.project_description}",
            openai_api_key=self.openai_api_key
        )
        try:
            return float(response.strip())
        except ValueError:
            return 0

    def optimize_code(self):
        initial_code = self.generate_initial_code()
        if not initial_code:
            return "Failed to generate initial code."

        for _ in range(self.max_steps):
            score = self.evaluate_code(initial_code)
            if score > self.best_score:
                self.best_code = initial_code
                self.best_score = score

            refinement_agent = AutomatedCodeRefinement(self.model, self.openai_api_key)
            refined_code = refinement_agent.refine_code(initial_code, "Improve performance and correctness.")
            if refined_code:
                initial_code = refined_code

        return self.best_code
