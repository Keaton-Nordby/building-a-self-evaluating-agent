import os
import logging


from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

load_dotenv()


llm = ChatGroq(model="llama-3.3-70b-versatile")

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


"""
stores logs in log dir if not already created and set level to logging.
made easy to read format. added file mode "w" to give logger write access to file
"""
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "execution.log"),
    level=logging,
    format="%(asctime)s- %(levelname)s = %(message)s",
    filemode="w",
)

logger = logging.getLogger(__name__)

"""
writes text inside folder in log dir  
use this to better organize logging based on what agent is being used
"""
def write_text_file(filename: str, content: str):
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        
        

"""
by design - the state is essentially a dictionary that holds all info 
flowing between agents. These agents will read from the state and send it 
to next agent in workflow
"""        
def planner_agent(state):
    logger.info("Planner agent started..")
    logger.info(f"User query: {state['user_query']}")
    prompt = f"""
    You are a planning agent in a multi agent AI system.
    
    Your job is to read the user's query and create a short plan for how the worker agent should answer.
    
    User query: {state['user_query']}
    """
    response = llm.invoke(prompt)
    plan = response.content if hasattr(response, "content") else str(response)
    """ Here we update the state to get ready to hand it off to next agent """
    state['plan'] = plan 
    write_text_file("planner_output.txt", plan)
    logger.info(f"Planner output:{plan}")
    return state