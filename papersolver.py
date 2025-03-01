import random
import re
import os
from inference import query_model

class ResearchPaperGenerator:
    def __init__(self, model, openai_api_key=None, project_topic="", max_steps=10):
        self.model = model
        self.openai_api_key = openai_api_key
        self.project_topic = project_topic
        self.max_steps = max_steps
        self.best_paper = None
        self.best_score = 0

    def generate_initial_paper(self):
        system_prompt = """
        You are an AI-powered research assistant.
        Your task is to generate a well-structured research paper based on the provided topic.
        Ensure clarity, scientific accuracy, and completeness.
        Output the research paper wrapped in ```latex.
        """
        response = query_model(
            model_str=self.model,
            system_prompt=system_prompt,
            prompt=self.project_topic,
            openai_api_key=self.openai_api_key
        )
        return self.extract_latex(response)

    def evaluate_paper(self, paper_content):
        system_prompt = """
        You are an AI-powered research reviewer assessing a generated paper.
        Provide a score from 0 to 1 based on clarity, scientific merit, and completeness.
        Output only a numeric score.
        """
        response = query_model(
            model_str=self.model,
            system_prompt=system_prompt,
            prompt=f"Paper Content:\n{paper_content}\n\nTopic:\n{self.project_topic}",
            openai_api_key=self.openai_api_key
        )
        try:
            return float(response.strip())
        except ValueError:
            return 0

    def refine_paper(self, paper_content, feedback="Improve clarity and completeness."):
        system_prompt = """
        You are an AI-powered research paper refinement assistant.
        Your goal is to enhance the provided paper based on feedback.
        Ensure readability, coherence, and scientific validity.
        Output the improved version wrapped in ```latex.
        """
        response = query_model(
            model_str=self.model,
            system_prompt=system_prompt,
            prompt=f"Feedback: {feedback}\n\nPaper:\n{paper_content}",
            openai_api_key=self.openai_api_key
        )
        return self.extract_latex(response)

    def optimize_paper(self):
        initial_paper = self.generate_initial_paper()
        if not initial_paper:
            return "Failed to generate initial paper."

        for _ in range(self.max_steps):
            score = self.evaluate_paper(initial_paper)
            if score > self.best_score:
                self.best_paper = initial_paper
                self.best_score = score

            refined_paper = self.refine_paper(initial_paper)
            if refined_paper:
                initial_paper = refined_paper

        return self.best_paper

    @staticmethod
    def extract_latex(response):
        match = re.search(r"```latex(.*?)```", response, re.DOTALL)
        return match.group(1).strip() if match else None
