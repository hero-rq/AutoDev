from utils import *
from tools import *
from inference import *

class DevelopmentReviewAgent:
    def __init__(self, model="gpt-4o-mini", notes=None, openai_api_key=None):
        if notes is None: self.notes = []
        else: self.notes = notes
        self.model = model
        self.openai_api_key = openai_api_key

    def review_code(self, project_plan, codebase):
        reviewer_prompt = """
        You are a software engineering reviewer assessing an automated development pipeline.
        Provide a detailed review of the given project plan and corresponding codebase.
        Format the response as JSON with the following keys:
        - "Summary": Briefly summarize the project plan and codebase.
        - "Strengths": Highlight key strengths of the codebase.
        - "Weaknesses": Identify any potential weaknesses or areas for improvement.
        - "Code Quality": Rate the overall code quality on a scale of 1 to 10.
        - "Maintainability": Rate how maintainable the code is on a scale of 1 to 10.
        - "Security": Rate the security considerations on a scale of 1 to 10.
        - "Scalability": Rate the scalability of the architecture on a scale of 1 to 10.
        - "Decision": Either "Accept" or "Needs Improvement".
        """
        review = query_model(
            model_str=self.model,
            system_prompt=reviewer_prompt,
            openai_api_key=self.openai_api_key,
            prompt=f"Project Plan: {project_plan}\n\nCodebase: {codebase}"
        )
        return extract_json_between_markers(review)

class SoftwareEngineerAgent:
    def __init__(self, model="gpt-4o-mini", notes=None, max_steps=100, openai_api_key=None):
        if notes is None: self.notes = []
        else: self.notes = notes
        self.max_steps = max_steps
        self.model = model
        self.openai_api_key = openai_api_key

    def develop_feature(self, project_requirements, feature_spec):
        development_prompt = """
        You are an AI-powered software engineer responsible for implementing new features.
        Given the project requirements and feature specifications, generate high-quality Python code.
        Ensure the implementation follows best practices in modularity, documentation, and performance.
        """
        code = query_model(
            model_str=self.model,
            system_prompt=development_prompt,
            openai_api_key=self.openai_api_key,
            prompt=f"Project Requirements: {project_requirements}\n\nFeature Specification: {feature_spec}"
        )
        return code.strip()

class DevOpsEngineerAgent:
    def __init__(self, model="gpt-4o-mini", notes=None, max_steps=100, openai_api_key=None):
        if notes is None: self.notes = []
        else: self.notes = notes
        self.max_steps = max_steps
        self.model = model
        self.openai_api_key = openai_api_key

    def deploy_application(self, infrastructure_config, deployment_strategy):
        devops_prompt = """
        You are an AI-powered DevOps engineer responsible for automating deployment pipelines.
        Given the infrastructure configuration and deployment strategy, generate a deployment script.
        Ensure the script follows best practices for CI/CD, security, and scalability.
        """
        deployment_script = query_model(
            model_str=self.model,
            system_prompt=devops_prompt,
            openai_api_key=self.openai_api_key,
            prompt=f"Infrastructure Configuration: {infrastructure_config}\n\nDeployment Strategy: {deployment_strategy}"
        )
        return deployment_script.strip()

class QAEngineerAgent:
    def __init__(self, model="gpt-4o-mini", notes=None, max_steps=100, openai_api_key=None):
        if notes is None: self.notes = []
        else: self.notes = notes
        self.max_steps = max_steps
        self.model = model
        self.openai_api_key = openai_api_key

    def generate_tests(self, feature_code):
        qa_prompt = """
        You are an AI-powered QA engineer responsible for writing unit and integration tests.
        Given the feature implementation, generate a comprehensive set of tests to ensure reliability.
        Ensure tests cover edge cases and follow best testing practices.
        """
        test_cases = query_model(
            model_str=self.model,
            system_prompt=qa_prompt,
            openai_api_key=self.openai_api_key,
            prompt=f"Feature Code:\n{feature_code}"
        )
        return test_cases.strip()
