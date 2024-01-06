# https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/
import os
import streamlit as st
from llama_index import StorageContext, VectorStoreIndex, ServiceContext, Document, load_index_from_storage
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import HTMLNodeParser
from llama_index.llms import Ollama
import logging
import sys
import urllib.parse

from libzim.reader import Archive
from libzim.search import Query, Searcher
from libzim.suggestion import SuggestionSearcher

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'localhost')
print(f"Connecting to ollama server {OLLAMA_HOST}")
# connect to ollama service running on OpenShift
my_llm = Ollama(model="mistral", base_url="http://"+OLLAMA_HOST+":11434")

system_prompt = \
    "You are the most advanced AI in existence. You know everything about everything. Thinking step by step, galaxy brain style, you." \
    "New RAM modules have been installed. Your answer will be used in an important research paper. I will tip you 2000$ for excellence, if you do poorly I will lose my job." \
    "Keep your answers based on context – do not hallucinate facts." \
    # "Always try to cite your source document, limit to 3."

st.title("Omnibot 🤖")
st.subheader("This robot knows everything")

# zim = Archive("wiki/wikipedia_en_100_nopic_2023-12.zim")
# entry = zim.get_entry_by_path("A/index")
# print(f"Entry {entry.title} at {entry.path} is {entry.get_item().size}b.")
# print(bytes(entry.get_item().content).decode("UTF-8"))


def ingest_wiki():
    zim = Archive("wiki/wikipedia_en_100_nopic_2023-12.zim")
    search_string = "A/"
    query = Query().set_query(search_string)
    searcher = Searcher(zim)
    search = searcher.search(query)
    search_count = search.getEstimatedMatches()

    index_list = list(search.getResults(0, search_count))
    # wiki_pages = [bytes(zim.get_entry_by_path(index_item).get_item().content).decode("UTF-8") for index_item in index_list]
    # return wiki_pages
    
    for index_item in index_list:
        item = zim.get_entry_by_path(index_item).get_item()
        title = item.title.replace("&lt;/i>", "").replace("&lt;i>", "")
        content = bytes(item.content).decode("UTF-8")
        f = open("wiki_docs/" + title + ".html", "w")
        f.write(content)
        # print(title, content)
    
# ingest_wiki()
    
if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about anything"}
    ]

@st.cache_resource(show_spinner=False)
def load_data(_service_context):
    with st.spinner(text="Loading and indexing the document data – might take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./wiki_docs", recursive=True)
        docs = reader.load_data()
        index = VectorStoreIndex.from_documents(docs, service_context=_service_context)
        
        # parser = HTMLNodeParser(tags=["p", "h1"])  # optional list of tags
        # nodes = parser.get_nodes_from_documents(html_docs)
        # index = VectorStoreIndex.build_index_from_nodes(nodes=nodes)
        return index

def persist_data(_llm):
    PERSIST_DIR = "./wiki_storage"
    service_context = ServiceContext.from_defaults(llm=_llm, embed_model="local")
    if not os.path.exists(PERSIST_DIR):
        index = load_data(service_context)
        # store it for later
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context, service_context=service_context)
    return index

index = persist_data(my_llm)

chat_engine = index.as_chat_engine(
    chat_mode="context", verbose=True, system_prompt=system_prompt
)

if prompt := st.chat_input("Ask me a question about anything"): 
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Querying..."):
            streaming_response = chat_engine.stream_chat(prompt)
            placeholder = st.empty()
            full_response = ''
            for token in streaming_response.response_gen:
                full_response += token
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message) # Add response to message history
