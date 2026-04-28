import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

load_dotenv()

class DataAgent:
    def __init__(self, df):
        # 1. Initialize Mistral with a slightly longer timeout
        self.llm = ChatOllama(
            model="mistral", 
            temperature=0,
            base_url="http://localhost:11434"
        )

        # 2. Simplified Agent Creation
        # We use the 'zero-shot-react-description' which is the most stable for local LLMs
        self.agent = create_pandas_dataframe_agent(
            llm=self.llm,
            df=df,
            verbose=True,
            agent_type="zero-shot-react-description",
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            max_iterations=3
        )

    def ask(self, query):
        try:
            # We guide Mistral to provide a final answer clearly
            response = self.agent.invoke({
                "input": f"{query}. Please provide a clear summary and a business insight."
            })
            
            if isinstance(response, dict):
                return response.get("output", "Analysis complete, but summary was empty.")
            return response
        except Exception as e:
            return f"Agent Analysis Error: {str(e)}"