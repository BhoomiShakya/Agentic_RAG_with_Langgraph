# 🚀 AI Assistant with LangChain and LangGraph

## 📌 Overview
This project implements an **AI Assistant** using **LangChain** and **LangGraph**, integrating **retrieval-augmented generation (RAG)** and **autonomous agent capabilities**. It leverages tools like **HuggingFace embeddings, Chroma vector store, and ChatGroq LLM** to process and respond to user queries efficiently.

---

## ⚙️ **Tech Stack**
- **LangChain** for LLM integration and RAG
- **LangGraph** for workflow management
- **HuggingFace Embeddings** for document vectorization
- **ChromaDB** for efficient retrieval
- **ChatGroq (LLaMA3-70B)** as the large language model
- **Graphviz & MermaidJS** for workflow visualization

---

## 🔧 **Setup & Installation**
To install the necessary dependencies, run:

```bash
pip install langchain-huggingface langchain-community langgraph langchain-groq chromadb graphviz pydot tiktoken
```

---

## 📜 **Project Workflow**
The project follows a structured AI workflow using LangGraph:

1. **AI Assistant Node:** Processes user queries and invokes the LLM.
2. **Retriever Node:** Searches ChromaDB for relevant documents.
3. **Rewriter Node:** Reformulates user queries to improve retrieval accuracy.
4. **Generate Node:** Uses RAG to generate an informed response.
5. **Grader Function:** Determines whether retrieved documents are relevant.

🔹 The workflow is **dynamically managed** using conditional edges to ensure an efficient response pipeline.

---

## 📁 **Core Components**
### 1️⃣ **Embeddings & Vector Store**
```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=doc_split,
    collection_name="rag-chroma",
    embedding=embeddings
)
retriever = vectorstore.as_retriever()
```

### 2️⃣ **AI Assistant Function**
```python
def AI_Assistant(state: AgentState):
    print("-----CALL AGENT-----")
    messages = state['messages']
    llm_with_tool = llm.bind_tools(tools)
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}
```

### 3️⃣ **Retriever Tool**
```python
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng's blog posts on LLM agents, prompt engineering, etc."
)
```

### 4️⃣ **Query Rewriter**
```python
def rewriter(state: AgentState):
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content
    response = llm.invoke([HumanMessage(content=f"Improve this query: {question}")])
    return {"messages": [response]}
```

### 5️⃣ **Document Grading (Relevance Scoring)**
```python
def grade_documents(state: AgentState):
    llm_with_structure_op = llm.with_structured_output(grade)
    prompt = PromptTemplate(
        template="""Decide if the document is relevant to the question.
                    Document: {context}
                    Question: {question}
                    Answer 'yes' or 'no'.""",
        input_variables=["context", "question"]
    )
    chain = prompt | llm_with_structure_op
    score = chain.invoke({"question": question, "context": docs}).binary_score
    return "generate" if score == "yes" else "rewriter"
```

---

## 🔄 **Graph Workflow Construction**
```python
workflow = StateGraph(AgentState)
workflow.add_node("AI_ASSISTANT", AI_Assistant)
workflow.add_node("retriever", retriever)
workflow.add_node("rewriter", rewriter)
workflow.add_node("generate", generate)
workflow.add_edge(START, "AI_ASSISTANT")
workflow.add_conditional_edges("AI_ASSISTANT", tools_condition, {"tools": "retriever", END: END})
workflow.add_conditional_edges("retriever", grade_documents, {"rewriter": "rewriter", "generate": "generate"})
workflow.add_edge("generate", END)
workflow.add_edge("rewriter", "AI_ASSISTANT")
app = workflow.compile()
```

---

## 📊 **Graph Visualization**
To visualize the workflow, run:
```python
from IPython.display import display, Image

graph_image = app.get_graph(xray=True).draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_image)
display(Image("graph.png"))
```

---

## 🚀 **Run the Assistant**
To interact with the assistant, invoke the app:
```python
app.invoke({"messages": ["What is an Autonomous Agent?"]})
```

---

## 🎯 **Key Features**
- **Retrieval-Augmented Generation (RAG)** for contextual AI responses
- **Conditional Graph Workflow** for optimized query handling
- **LangGraph Integration** for modular AI processing
- **Automatic Query Rewriting** for better search accuracy
- **Document Relevance Scoring** to refine results

---

## 📌 **Future Enhancements**
- 🌟 Support for more LLM models like GPT-4, Claude, and Mistral
- 📚 Expand retriever to multiple data sources
- 🛠️ Improve grading with more complex relevance scoring

---

## 💡 **Contributing**
Feel free to fork, modify, and submit PRs! 😊

📩 Contact: *[Your Email/GitHub Profile]*

