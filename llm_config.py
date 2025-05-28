import os
import json
from litellm import completion
import requests

class LLMClient:
    def __init__(self):
        self.models = ["ollama/mixtral-8x7b"]
        with open("tools.json", "r") as f:
            self.tools = json.load(f)

    def query(self, prompt):
        for model in self.models:
            try:
                response = completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    tools=self.tools,
                    tool_choice="auto",
                    api_base="http://localhost:11434"
                )
                if response.choices[0].message.tool_calls:
                    for tool_call in response.choices[0].message.tool_calls:
                        if tool_call.function.name == "rag_query":
                            query = json.loads(tool_call.function.arguments).get("query")
                            rag_response = requests.post(
                                "http://85.31.233.33:5000/rag_query",
                                json={"query": query},
                                auth=("admin", "your_secure_password")
                            ).json()
                            followup_response = completion(
                                model=model,
                                messages=[
                                    {"role": "user", "content": prompt},
                                    {"role": "function", "name": "rag_query", "content": json.dumps(rag_response)}
                                ],
                                api_base="http://localhost:11434"
                            )
                            return followup_response.choices[0].message.content
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error with {model}: {e}")
                continue
        return "All models failed to respond."

if __name__ == "__main__":
    llm = LLMClient()
    print(llm.query("What automations use binary_sensor.kitchen_motion?"))
