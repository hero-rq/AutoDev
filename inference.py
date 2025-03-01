import time
import os
import json
import tiktoken
from openai import OpenAI
import openai
import anthropic

# Token tracking dictionaries
TOKENS_IN = {}
TOKENS_OUT = {}

# Initialize encoding once for all cost estimations
ENCODING = tiktoken.encoding_for_model("gpt-4o")

def curr_cost_est():
    """Estimate the current cost based on tokens used."""
    costmap_in = {
        "gpt-4o": 2.50 / 1000000,
        "gpt-4o-mini": 0.150 / 1000000,
        "o1-preview": 15.00 / 1000000,
        "o1-mini": 3.00 / 1000000,
        "claude-3-5-sonnet": 3.00 / 1000000,
        "deepseek-chat": 1.00 / 1000000,
        "o1": 15.00 / 1000000,
    }
    costmap_out = {
        "gpt-4o": 10.00 / 1000000,
        "gpt-4o-mini": 0.6 / 1000000,
        "o1-preview": 60.00 / 1000000,
        "o1-mini": 12.00 / 1000000,
        "claude-3-5-sonnet": 12.00 / 1000000,
        "deepseek-chat": 5.00 / 1000000,
        "o1": 60.00 / 1000000,
    }
    total_cost = sum(costmap_in[m] * TOKENS_IN.get(m, 0) for m in costmap_in) + \
                 sum(costmap_out[m] * TOKENS_OUT.get(m, 0) for m in costmap_out)
    return total_cost

def get_api_key(api_key_arg, env_var):
    """
    Returns the API key from the argument or environment variable.
    Raises an Exception if the key is not found.
    """
    key = api_key_arg or os.getenv(env_var)
    if not key:
        raise Exception(f"No API key provided. Please set the {env_var} environment variable or pass it as an argument.")
    return key

def query_openai(model_str, messages, temperature):
    """
    Queries the OpenAI API using the provided model and messages.
    """
    client = OpenAI()
    return client.chat.completions.create(
        model=f"{model_str}-2024-07-18",
        messages=messages,
        temperature=temperature
    )

def query_anthropic(system_prompt, prompt):
    """
    Queries the Anthropic API using the provided system prompt and user prompt.
    """
    anthropic_key = get_api_key(None, "ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=anthropic_key)
    message = client.messages.create(
        model="claude-3-5-sonnet-latest",
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}]
    )
    completion = json.loads(message.to_json())
    # Assuming the answer is stored under content[0]["text"]
    return completion["content"][0]["text"]

def query_model(model_str, prompt, system_prompt,
                openai_api_key=None, anthropic_api_key=None,
                tries=5, timeout=5.0, temp=None, print_cost=True, version="1.5"):
    """
    Queries the chosen model with retries, error handling, and cost estimation.
    Supports both OpenAI and Anthropic APIs.
    """
    # Set API keys
    if openai_api_key or os.getenv("OPENAI_API_KEY"):
        openai.api_key = get_api_key(openai_api_key, "OPENAI_API_KEY")
    if anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = get_api_key(anthropic_api_key, "ANTHROPIC_API_KEY")
    
    # Prepare messages for the model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # Try querying the model up to 'tries' times with exponential backoff
    for attempt in range(tries):
        try:
            if model_str in ["gpt-4o-mini", "gpt4", "gpt-4o"]:
                completion = query_openai(model_str, messages, temp or 0.7)
                answer = completion.choices[0].message.content
            elif model_str == "claude-3.5-sonnet":
                answer = query_anthropic(system_prompt, prompt)
            else:
                raise ValueError(f"Unsupported model: {model_str}")

            # Track tokens used for cost estimation
            TOKENS_IN[model_str] = TOKENS_IN.get(model_str, 0) + len(ENCODING.encode(system_prompt + prompt))
            TOKENS_OUT[model_str] = TOKENS_OUT.get(model_str, 0) + len(ENCODING.encode(answer))
            if print_cost:
                print(f"Current cost estimate: ${curr_cost_est():.6f}")
            return answer
        
        except Exception as e:
            print(f"Model query error on attempt {attempt + 1}: {e}")
            time.sleep(timeout * (attempt + 1))  # Increase sleep time on each retry
    
    raise Exception("Max retries reached")
