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


def write_text_file(filename: str, content: str):
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)