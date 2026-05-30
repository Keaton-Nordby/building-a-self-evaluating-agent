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
    """Update the state to get ready to hand it off to next agent """
    state['plan'] = plan 
    write_text_file("planner_output.txt", plan)
    logger.info(f"Planner output:{plan}")
    return state



"""Worker agent logic"""
def worker_agent(state):
    """tracks number of calls to see if worker agent needs improvment"""
    state['worker_calls'] += 1
    feedback = state.get('review_reason', "")
    logger.info(f"Worker call #{state['worker_calls']}")
    prompt = f"""
    You are a worker agent
    
    User query: {state['user_query']}
    
    Plan {state['plan']}
    
    Previous reviewer feedback: {feedback}
    
    Write the best possible improved response. If reviewer feedback exists and explicitly fix those issues
    """
    response = llm.invoke(prompt)
    draft_repsonse = response.content if hasattr(response, "content") else str(response)
    """checks to see if the response needs improvement"""
    state['draft_response'] = draft_repsonse
    
    write_text_file(f"Worker_output_{state['worker_calls']}.txt", draft_repsonse)
    logger.info(f"Worker_output_{state['worker_calls']}: {draft_repsonse}") 
    
    return state   


"""Reviews previous agent answer and reasoning"""
def review_agent(state):
    state['reviewer_calls'] += 1
    prompt = f"""
    You are a strict reviewer agent.
    
    User query: {state['user_query']},
    
    Draft repsonse: {state['draft_response']}
    
    Check for:
     - concrete examples
     - implementation details
     - tradeoffs
     - clarity
     - actionable recommendations
     
    If anything is missing, revise.
    Return EXACTLY in this format:
    Decision: approve or revise
    Reason: brief reason
    """
    response = llm.invoke(prompt)
    raw_output = response.content.strip() if hasattr(response, "content") else str(response)
    """Check to see if the answer is sufficient enough or if it needs to get send back to another agent"""
    decision = "approve" if "dicisions: approve" in raw_output.lower() else 'review'
    
    reason_line = next((line for line in raw_output.striplines() if line.lower().startswith("reason:")), "")
    reason = reason_line.replace("Reason: ", "").strip() if reason_line else "No reason provided"
    state['review_decision'] = decision
    state['review_reason'] = reason
    
    write_text_file(
        f"reviewer_output_{state['reviewer_calls']}.txt",
        raw_output + f"\nParsed Decision: {decision}\nReason: {reason}"
    )
    logger.info(f"Reviewer decision #{state['reviewer_calls']}: {decision} ({reason})")
    return state
    