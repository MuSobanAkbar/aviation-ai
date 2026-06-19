import os, json, glob, requests
from functools import lru_cache
from dotenv import load_dotenv
from groq import Groq
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
groq_client = Groq()
MODEL = "llama-3.3-70b-versatile"
AVIATIONSTACK_KEY = os.getenv("AVIATIONSTACK_KEY")
BASE_URL = "https://api.aviationstack.com/v1/flights"   
collection = None   

def build_index(folder="data"):
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection("policies")
    except Exception:
        pass
    coll = client.get_or_create_collection("policies")

    next_id = 0
    for pdf_path in glob.glob(os.path.join(folder, "*.pdf")):
        pages = PyPDFLoader(pdf_path).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(pages)
        coll.add(
            documents=[c.page_content for c in chunks],
            ids=[f"chunk_{next_id + i}" for i in range(len(chunks))],
            metadatas=[{"source": os.path.basename(pdf_path)} for _ in chunks],
        )
        next_id += len(chunks)
        print(f"Indexed {len(chunks)} chunks from {os.path.basename(pdf_path)}")
    return coll



# rag retrieval 
def search_policies(query):
    """Search airline baggage/fare policy documents for an answer."""
    results = collection.query(query_texts=[query], n_results=3)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    if not docs:
        return {"error": "No relevant policy found in the documents."}
    return {"results": [{"source": m.get("source"), "text": d}
                        for d, m in zip(docs, metas)]}


# flight status retrieval

@lru_cache(maxsize=128)             # caches repeats -> protects your 100/mo quota
def get_flight_status(flight_iata):
    """Get live status, departure and arrival info for a flight by IATA code."""
    try:
        resp = requests.get(BASE_URL, params={
            "access_key": AVIATIONSTACK_KEY, "flight_iata": flight_iata,
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            return {"error": f"No flight found for {flight_iata}"}
        f = data[0]
        return {
            "flight": f["flight"]["iata"],
            "airline": f["airline"]["name"],
            "status": f["flight_status"],
            "from": f["departure"]["airport"],
            "departs_scheduled": f["departure"]["scheduled"],
            "to": f["arrival"]["airport"],
            "arrives_scheduled": f["arrival"]["scheduled"],
        }
    except Exception as e:
        return {"error": str(e)}
    


#defining the tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_flight_status",
            "description": "Get LIVE status, departure/arrival info for a specific flight by its IATA code. Use for real-time questions like delays, status, or arrival time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_iata": {"type": "string", "description": "IATA flight code, e.g. 'EK231', 'AC856'."}
                },
                "required": ["flight_iata"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policies",
            "description": "Search airline baggage POLICIES, fare, and change/cancellation POLICY documents. Use for static questions like baggage allowance, fees, or rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What the user wants to know, e.g. 'Emirates carry-on weight limit'."}
                },
                "required": ["query"],
            },
        },
    },
]

available_tools = {"get_flight_status": get_flight_status, "search_policies": search_policies}



# agent
SYSTEM = (
    "You are a helpful travel assistant. You have access to tools to look up real-time flight data and internal airline policy documents.\n\n"
    "When a user asks about a specific flight status or delay, use your flight lookup tool to find the answers.\n"
    "When a user asks about baggage limits, cancellations, or ticket rules, search the policy documents.\n"
    "Always rely entirely on the tool outputs to answer the user's questions. Do not make up information."
)

def respond(message, history):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": message}]

    first = groq_client.chat.completions.create(
        model=MODEL, messages=messages, tools=tools, tool_choice="auto",
    )
    msg = first.choices[0].message

    if not msg.tool_calls:
        return msg.content

    messages.append(msg)
    for tc in msg.tool_calls:
        fn = available_tools[tc.function.name]
        args = json.loads(tc.function.arguments)
        result = fn(**args)
        messages.append({
            "role": "tool", "tool_call_id": tc.id, "content": json.dumps(result),
        })

    second = groq_client.chat.completions.create(model=MODEL, messages=messages)
    return second.choices[0].message.content


if __name__ == "__main__":
    collection = build_index("data")     # sets the module-global RAG index
    import gradio as gr
    gr.ChatInterface(
        respond,
        title="Flight & Travel Agent",
        description="Ask about live flights (e.g. 'Is EK231 delayed?') or airline policies (e.g. 'Emirates baggage allowance?').",
        examples=["Is EK231 delayed?", "What's Emirates' carry-on limit?"],
    ).launch(server_name="0.0.0.0", server_port=7860)