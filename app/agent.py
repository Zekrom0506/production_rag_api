from typing import Optional, Literal
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langsmith import traceable

from app.config import get_settings
from app.vectorstore import get_vectorstore


class AgentState(TypedDict):
    """State for the production agentic RAG agent."""
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    rewritten_query: str
    documents: list[Document]
    relevance_score: float
    retry_count: int
    max_retries: int
    error: Optional[str]
    model_used: str


class ProductionAgent:
    """
    Production agentic RAG agent with:
    - Vector retrieval (PGVector on Supabase)
    - Self-correcting retrieval (grade -> rewrite -> retry loop)
    - Graceful fallback when retrieval fails
    - Groq LLM provider with primary/fallback models
    - LangSmith tracing
    """

    def __init__(self):
        settings = get_settings()

        self.primary_llm = ChatGroq(
            model=settings.primary_model,
            api_key=settings.groq_api_key,
            temperature=0,
            timeout=30,
            max_retries=0,
        )
        self.fallback_llm = ChatGroq(
            model=settings.fallback_model,
            api_key=settings.groq_api_key,
            temperature=0,
            timeout=30,
            max_retries=0,
        )
        self.grader_llm = ChatGroq(
            model=settings.fallback_model,
            api_key=settings.groq_api_key,
            temperature=0,
            timeout=30,
            max_retries=0,
        )

        self.vectorstore = get_vectorstore()
        self.max_retries = settings.max_retries
        self.graph = self._build_graph()

    def _build_graph(self):

        def retrieve(state: AgentState) -> dict:
            """Retrieve documents from PGVector based on the query."""
            query = state.get("rewritten_query") or state["query"]
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(query)
            return {"documents": docs}

        def grade(state: AgentState) -> dict:
            """Use an LLM to grade retrieved documents for relevance."""
            query = state["query"]
            docs = state["documents"]

            if not docs:
                return {"documents": [], "relevance_score": 0.0}

            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a relevance grader. Output ONLY a number between 0 and 1 "
                 "indicating how relevant the document is to the query. "
                 "1.0 = directly answers it, 0.0 = not relevant at all."),
                ("human", "Query: {query}\n\nDocument: {document}\n\nScore:"),
            ])

            scores = []
            relevant_docs = []

            for doc in docs:
                chain = prompt | self.grader_llm
                result = chain.invoke({"query": query, "document": doc.page_content})
                try:
                    score = float(result.content.strip())
                except ValueError:
                    score = 0.5
                scores.append(score)
                if score >= 0.5:
                    relevant_docs.append(doc)

            avg_score = sum(scores) / len(scores) if scores else 0.0
            return {"documents": relevant_docs, "relevance_score": avg_score}

        def rewrite(state: AgentState) -> dict:
            """Rewrite the query to improve retrieval on the next attempt."""
            query = state["query"]
            retry_count = state.get("retry_count", 0)

            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a query rewriter for a RAG system. The original query "
                 "did not retrieve relevant documents. Rewrite it to be more "
                 "specific and likely to match relevant content. "
                 "Output ONLY the rewritten query, nothing else."),
                ("human", "Original query: {query}\n\nRewritten query:"),
            ])

            chain = prompt | self.fallback_llm
            result = chain.invoke({"query": query})

            return {
                "rewritten_query": result.content.strip(),
                "retry_count": retry_count + 1,
            }

        def generate(state: AgentState) -> dict:
            """Generate the final answer using retrieved context."""
            query = state["query"]
            docs = state["documents"]

            context = "\n\n".join(doc.page_content for doc in docs)

            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a helpful assistant. Answer the question using ONLY "
                 "the information in the context below. If the context doesn't "
                 "contain enough information, say so clearly. Do not make things up."),
                ("human", "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"),
            ])

            try:
                chain = prompt | self.primary_llm
                result = chain.invoke({"context": context, "query": query})
                return {"messages": [result], "error": None, "model_used": "primary_rag"}
            except Exception:
                try:
                    chain = prompt | self.fallback_llm
                    result = chain.invoke({"context": context, "query": query})
                    return {"messages": [result], "error": None, "model_used": "fallback_rag"}
                except Exception as e:
                    return {
                        "messages": [AIMessage(content="I'm having trouble generating a response right now.")],
                        "error": str(e),
                        "model_used": "error_handler",
                    }

        def fallback(state: AgentState) -> dict:
            """Graceful failure when no relevant documents were found at all."""
            query = state["query"]
            return {
                "messages": [AIMessage(
                    content=(
                        f"I couldn't find relevant information to answer: \"{query}\". "
                        "Try rephrasing your question, or it may not be covered in "
                        "the available documents."
                    )
                )],
                "model_used": "no_context_fallback",
            }

        def route_after_grade(state: AgentState) -> Literal["rewrite", "generate", "fallback"]:
            """Decide whether to retry retrieval, generate, or give up gracefully."""
            score = state.get("relevance_score", 0.0)
            retry_count = state.get("retry_count", 0)
            max_retries = state.get("max_retries", self.max_retries)
            docs = state.get("documents", [])

            if score >= 0.5 and len(docs) > 0:
                return "generate"
            if retry_count < max_retries:
                return "rewrite"
            if len(docs) > 0:
                return "generate"
            return "fallback"

        graph = StateGraph(AgentState)

        graph.add_node("retrieve", retrieve)
        graph.add_node("grade", grade)
        graph.add_node("rewrite", rewrite)
        graph.add_node("generate", generate)
        graph.add_node("fallback", fallback)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "grade")

        graph.add_conditional_edges(
            "grade",
            route_after_grade,
            {"rewrite": "rewrite", "generate": "generate", "fallback": "fallback"},
        )

        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("generate", END)
        graph.add_edge("fallback", END)

        return graph.compile()

    @traceable(name="production_agent_invoke")
    def invoke(self, message: str) -> dict:
        """
        Invoke the agentic RAG pipeline with a user message.
        Returns: {"response": str, "model_used": str, "error": str | None}
        """
        result = self.graph.invoke({
            "messages": [HumanMessage(content=message)],
            "query": message,
            "rewritten_query": "",
            "documents": [],
            "relevance_score": 0.0,
            "retry_count": 0,
            "max_retries": self.max_retries,
            "error": None,
            "model_used": "",
        })

        return {
            "response": result["messages"][-1].content,
            "model_used": result.get("model_used", "unknown"),
            "error": result.get("error"),
        }