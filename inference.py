import time
import os
import json
import re
import tiktoken
from openai import OpenAI
import openai
import anthropic

TOKENS_IN = {}
TOKENS_OUT = {}

encoding = tiktoken.get_encoding("cl100k_base")

def curr_cost_est():
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
    return sum(costmap_in[m] * TOKENS_IN.get(m, 0) for m in costmap_in) + sum(costmap_out[m] * TOKENS_OUT.get(m, 0) for m in costmap_out)

def query_model(model_str, prompt, system_prompt, openai_api_key=None, anthropic_api_key=None, tries=5, timeout=5.0, temp=None, print_cost=True, version="1.5"):
    if openai_api_key is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    if anthropic_api_key is None:
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not openai_api_key and not anthropic_api_key:
        raise Exception("No API key provided in query_model function")
    if openai_api_key:
        openai.api_key = openai_api_key
    if anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    for _ in range(tries):
        try:
            client = OpenAI()
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

            if model_str in ["gpt-4o-mini", "gpt4omini", "gpt-4o"]:
                completion = client.chat.completions.create(
                    model=f"{model_str}-2024-07-18", messages=messages, temperature=temp or 0.7)
            elif model_str == "claude-3.5-sonnet":
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                completion = json.loads(message.to_json())
                answer = completion["content"][0]["text"]
            else:
                raise ValueError(f"Unsupported model: {model_str}")

            answer = completion.choices[0].message.content

            # Token tracking
            try:
                encoding = tiktoken.encoding_for_model("gpt-4o")
                TOKENS_IN[model_str] = TOKENS_IN.get(model_str, 0) + len(encoding.encode(system_prompt + prompt))
                TOKENS_OUT[model_str] = TOKENS_OUT.get(model_str, 0) + len(encoding.encode(answer))
                if print_cost:
                    print(f"Current cost estimate: ${curr_cost_est():.6f}")
            except Exception as e:
                if print_cost:
                    print(f"Cost estimation error: {e}")
            return answer
        except Exception as e:
            print("Model query error:", e)
            time.sleep(timeout)
    raise Exception("Max retries reached")
