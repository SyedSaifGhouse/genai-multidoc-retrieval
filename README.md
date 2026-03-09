## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:

Extracting specific, nuanced information from a collection of dense academic papers is a slow and inefficient manual process. Standard search tools rely on exact keywords and fail to understand the conceptual context of a user's question. This program aims to build an AI agent that can intelligently query multiple documents to synthesize precise answers to complex questions.

### DESIGN STEPS:

#### STEP 1:
Load PDF documents and create specialized search and summary tools for each paper.

#### STEP 2:
Initialize an AI agent with an OpenAI model, giving it access to all the created tools.

#### STEP 3:
Query the agent with a specific question about one paper to get a detailed answer from its content.

### PROGRAM:
```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
import nest_asyncio
nest_asyncio.apply()
urls = [
    "https://openreview.net/pdf?id=0wSlFpMsGb",
    "https://openreview.net/pdf?id=MS9nWFY7LG",
    "https://openreview.net/pdf?id=FtL9eEmU6v",
]

papers = [
    "eval.pdf",
    "long.pdf",
    "largest.pdf",
]
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
len(initial_tools)
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)
response = agent.query("Give me a summary of eval,long and largest")
response = agent.query("what is the main idea of each papers eval,long and largest ?")
response = agent.query("What are the key differences between the eval and long papers ?")
print(str(response))
```

### OUTPUT:
<img width="687" height="638" alt="Screenshot 2026-03-09 161921" src="https://github.com/user-attachments/assets/5c70c511-7a2a-4a6c-8026-5a983a0cb6b3" />
<img width="697" height="409" alt="Screenshot 2026-03-09 162016" src="https://github.com/user-attachments/assets/c95e7b85-f652-44c4-b315-d926260c0a91" />
<img width="684" height="577" alt="Screenshot 2026-03-09 162026" src="https://github.com/user-attachments/assets/f93fee16-b34e-48d7-a1ab-3da35c60aa50" />

### RESULT:
The system successfully retrieves and synthesizes relevant information from multiple documents, providing concise and relevant answers to the user's query. Performance is evaluated based on the accuracy, relevance, and coherence of the responses.    
