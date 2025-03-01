from agents import *
from copy import copy
from common_imports import *
from torch.backends.mkl import verbose

import argparse
import pickle
import os
import time

DEFAULT_LLM_BACKBONE = "o1-mini"

class AutomatedDevWorkflow:
    def __init__(self, project_name, openai_api_key, max_steps=100, agent_model_backbone=f"{DEFAULT_LLM_BACKBONE}", notes=list(), human_in_loop_flag=None):
        """
        Initialize the automated development workflow.
        @param project_name: (str) Description of the software project to develop.
        @param max_steps: (int) Maximum number of steps per phase.
        @param agent_model_backbone: (str or dict) Model backbone for agents.
        @param notes: (list) Development notes and guidelines.
        """
        self.notes = notes
        self.max_steps = max_steps
        self.openai_api_key = openai_api_key
        self.project_name = project_name
        self.model_backbone = agent_model_backbone
        self.human_in_loop_flag = human_in_loop_flag
        
        self.print_cost = True
        self.review_override = True
        self.review_ovrd_steps = 0
        self.reference_documents = list()
        
        self.phases = [
            ("requirements analysis", ["gather requirements", "define scope"]),
            ("architecture design", ["design components", "define data structures"]),
            ("implementation", ["develop modules", "write tests"]),
            ("integration and testing", ["integrate components", "run tests"]),
            ("deployment", ["deploy application", "monitor performance"]),
            ("maintenance", ["bug fixes", "feature enhancements"])
        ]
        self.phase_status = {subtask: False for _, subtasks in self.phases for subtask in subtasks}
        
        self.statistics_per_phase = {subtask: {"time": 0.0, "steps": 0.0} for _, subtasks in self.phases for subtask in subtasks}
        
        self.engineer = QAEngineerAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.qa_engineer = QAEngineerAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        self.devops_engineer = DevOpsEngineerAgent(model=self.model_backbone, notes=self.notes, max_steps=self.max_steps, openai_api_key=self.openai_api_key)
        
        os.makedirs("./project_repo", exist_ok=True)
        os.makedirs("./project_repo/src", exist_ok=True)
        os.makedirs("./project_repo/tests", exist_ok=True)
        os.makedirs("./project_repo/docs", exist_ok=True)
    
    def perform_development(self):
        """
        Execute the full development workflow.
        """
        for phase, subtasks in self.phases:
            phase_start_time = time.time()
            print(f"Starting phase: {phase}")
            for subtask in subtasks:
                print(f"  -> Executing subtask: {subtask}")
                self.execute_subtask(subtask)
                self.phase_status[subtask] = True
            phase_end_time = time.time()
            phase_duration = phase_end_time - phase_start_time
            print(f"Completed phase: {phase} in {phase_duration:.2f} seconds\n")
            self.statistics_per_phase[phase] = {"time": phase_duration}
    
    def execute_subtask(self, subtask):
        """
        Execute an individual subtask.
        """
        agent = self.get_agent_for_subtask(subtask)
        if agent:
            result = agent.perform_task(self.project_name, subtask)
            self.save_result(subtask, result)
    
    def get_agent_for_subtask(self, subtask):
        """
        Return the appropriate agent for a given subtask.
        """
        if subtask in ["gather requirements", "define scope"]:
            return self.engineer
        elif subtask in ["design components", "define data structures"]:
            return self.engineer
        elif subtask in ["develop modules", "write tests"]:
            return self.engineer
        elif subtask in ["integrate components", "run tests"]:
            return self.qa_engineer
        elif subtask in ["deploy application", "monitor performance"]:
            return self.devops_engineer
        elif subtask in ["bug fixes", "feature enhancements"]:
            return self.engineer
        return None
    
    def save_result(self, subtask, result):
        """
        Save the results of a subtask.
        """
        with open(f"./project_repo/{subtask.replace(' ', '_')}.txt", "w") as f:
            f.write(result)
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Automated Software Development Workflow")
    parser.add_argument('--project-name', type=str, required=True, help='Specify the software project name.')
    parser.add_argument('--api-key', type=str, required=True, help='Provide the OpenAI API key.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    workflow = AutomatedDevWorkflow(
        project_name=args.project_name,
        openai_api_key=args.api_key
    )
    workflow.perform_development()
