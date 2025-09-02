from state import State
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.utilities.requests import TextRequestsWrapper
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from config import EXECUTOR_MODEL

ALLOW_DANGEROUS_REQUEST = True

class ExecutorAgent:
    """
    ExecutorAgent is responsible for invoking the LLM-based React agent
    using the system message and user query from the current state.
    It executes API calls through the RequestsToolkit and captures
    the resulting response to store it back in the state.
    """
    def __init__(self, llm_model: str = EXECUTOR_MODEL):
        self.llm = ChatOllama(
            model=llm_model,
            temperature=0.0,
            top_p=0.95,
            num_ctx=2048
        )

    def run(self, state: State) -> State:
        print("Running ExecutorAgent...")
        if not state.get("system_message"):
            return state

        try:
            print("Generating the answer...")
            toolkit = RequestsToolkit(
                requests_wrapper=TextRequestsWrapper(headers={}),
                allow_dangerous_requests=ALLOW_DANGEROUS_REQUEST,
            )

            http_tools = toolkit.get_tools()

            agent = create_react_agent(
                self.llm,
                http_tools,
                prompt=state["system_message"]
            )

            result = agent.invoke({"messages": [("user", state["user_query"])]})

        except Exception as e:
            print(f"Executor agent execution failed: {e}")
            return {**state, "last_response": None}
        
        last_response = None
        if isinstance(result, dict) and "messages" in result and result["messages"]:
            last_response = result["messages"][-1].content
        else:
            last_response = str(result)

        print(f"\nResult using: {state.get('current_api_path')} \n{last_response}\n")

        return {**state, "last_response": last_response}
