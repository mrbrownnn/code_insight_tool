from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from typing import List, Dict, Any


# ============================================================================
# CODE QUESTION & ANSWER TEMPLATE
# ============================================================================
CODE_QA_EXAMPLES = [
    {
        "code_context": """
def calculate_fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
""",
        "question": "What does this function do and what's wrong with it?",
        "answer": "This function calculates the nth Fibonacci number recursively. However, it has exponential time complexity O(2^n) due to redundant calculations. For n=40, it will be extremely slow. A better approach would be to use memoization or dynamic programming to achieve O(n) complexity."
    },
    {
        "code_context": """
async def fetch_user_data(user_id: str) -> Dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/users/{user_id}") as resp:
            return await resp.json()
""",
        "question": "How does this function handle errors?",
        "answer": "This function lacks error handling. If the API is down, the user_id is invalid, or the response is not valid JSON, it will raise an unhandled exception. Best practices would include try-except blocks, proper status code checking, and timeout management. Also consider adding retry logic for transient failures."
    },
    {
        "code_context": """
class DataProcessor:
    def __init__(self):
        self.data = []
    
    def add_item(self, item):
        self.data.append(item)
    
    def get_all(self):
        return self.data
""",
        "question": "What design patterns could improve this class?",
        "answer": "This class could benefit from several improvements: (1) Use type hints for clarity, (2) Add data validation in add_item(), (3) Consider using __slots__ for memory efficiency, (4) Add methods for filtering/searching, (5) Implement __iter__ for iteration support, (6) Consider immutability patterns if data shouldn't be modified externally."
    }
]

CODE_QA_PROMPT_TEMPLATE = FewShotPromptTemplate(
    examples=CODE_QA_EXAMPLES,
    example_prompt=PromptTemplate(
        input_variables=["code_context", "question", "answer"],
        template="Code Context:\n{code_context}\n\nQuestion: {question}\nAnswer: {answer}"
    ),
    suffix="Code Context:\n{code_context}\n\nQuestion: {question}\nAnswer:",
    input_variables=["code_context", "question"],
    example_separator="\n---\n"
)


# ============================================================================
# CODE EXPLANATION TEMPLATE
# ============================================================================
CODE_EXPLANATION_EXAMPLES = [
    {
        "code": """
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""",
        "explanation": "This function computes Fibonacci numbers using recursion with memoization. The @lru_cache decorator stores previous results to avoid recalculating them. It has O(n) time complexity and can efficiently handle larger values of n compared to naive recursion."
    },
    {
        "code": """
def merge_sorted_lists(list1: List[int], list2: List[int]) -> List[int]:
    result = []
    i = j = 0
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    result.extend(list1[i:])
    result.extend(list2[j:])
    return result
""",
        "explanation": "This function merges two sorted lists into one sorted list using the two-pointer technique. It iterates through both lists simultaneously, always picking the smaller element. Time complexity is O(n+m) where n and m are list lengths. This is the merge step used in merge sort algorithm."
    }
]

CODE_EXPLANATION_PROMPT_TEMPLATE = FewShotPromptTemplate(
    examples=CODE_EXPLANATION_EXAMPLES,
    example_prompt=PromptTemplate(
        input_variables=["code", "explanation"],
        template="Code:\n{code}\n\nExplanation: {explanation}"
    ),
)
# ============================================================================
# CODE SUMMARIZATION TEMPLATE
# ============================================================================
CODE_SUMMARIZATION_EXAMPLES = [
    {
        "code": """
class FileProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
    
    def read(self):
        with open(self.file_path, 'r') as f:
            self.data = f.read()
        return self.data
    
    def process(self):
        lines = self.data.split('\\n')
        return [line.strip() for line in lines if line.strip()]
    
    def write_output(self, output_path: str):
        with open(output_path, 'w') as f:
            f.write('\\n'.join(self.process()))
""",
        "summary": "FileProcessor reads a text file, removes empty lines and whitespace, and writes the cleaned data to an output file. It uses context managers for safe file handling and provides a simple pipeline: read → process → write."
    },
    {
        "code": """
def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_password(password: str) -> bool:
    return len(password) >= 8 and any(c.isupper() for c in password)

def validate_user_input(email: str, password: str) -> Dict[str, bool]:
    return {
        'email_valid': validate_email(email),
        'password_valid': validate_password(password)
    }
""",
        "summary": "This module provides user input validation with three functions: validate_email() checks email format using regex, validate_password() ensures minimum 8 characters and at least one uppercase letter, and validate_user_input() combines both checks returning a dictionary of results."
    }
]

CODE_SUMMARIZATION_PROMPT_TEMPLATE = FewShotPromptTemplate(
example_prompt=PromptTemplate(
    input_variables=["code", "summary"],
    template="Code:\n{code}\n\nSummary: {summary}"),
    examples=CODE_SUMMARIZATION_EXAMPLES
)


# ============================================================================
# CONTEXT-AWARE RAG TEMPLATE
# ============================================================================
# This template is used in the RAG chain to provide both code context and conversation history

RAG_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are an expert code analyst. Use the provided code context to answer the user's question.

Code Context:
{context}

Previous Conversation:
{chat_history}

User Question: {question}

Provide a detailed, well-structured answer based on the code context. Include:
1. Direct answer to the question
2. Relevant code snippets from context (if applicable)
3. Best practices or potential improvements
4. Related files or functions to check (if mentioned in context)

Answer:"""
)


# ============================================================================
# 5. RETRIEVAL QUERY GENERATION TEMPLATE
# ============================================================================
# Used to transform user questions into retrieval queries

QUERY_GENERATION_TEMPLATE = PromptTemplate(
    input_variables=["question"],
    template="""Given the user question about code, generate 3 diverse retrieval search queries:
1. A direct search query focusing on keywords
2. A semantic search query rephrasing the question
3. A related functionality search query that might help understand context

User Question: {question}

Queries (JSON format):
{{
    "keyword_query": "...",
    "semantic_query": "...",
    "related_query": "..."
}}"""
)


# ============================================================================
# CODE REFERENCE EXTRACTION TEMPLATE
# ============================================================================
# Used to extract file:line references from LLM responses

CODE_REFERENCE_TEMPLATE = PromptTemplate(
    input_variables=["response"],
    template="""Extract all file:line references from this response about code. Format: filename.py:line_number

Response: {response}

References (one per line, or "None" if no references):"""
)


# ============================================================================
# Helper Functions
# ============================================================================

def get_qa_prompt(code_context: str, question: str) -> str:
    """Generate a formatted Q&A prompt with few-shot examples."""
    return CODE_QA_PROMPT_TEMPLATE.format(
        code_context=code_context,
        question=question
    )


def get_explanation_prompt(code: str) -> str:
    """Generate a formatted explanation prompt with few-shot examples."""
    return CODE_EXPLANATION_PROMPT_TEMPLATE.format(code=code)


def get_summarization_prompt(code: str) -> str:
    """Generate a formatted summarization prompt with few-shot examples."""
    return CODE_SUMMARIZATION_PROMPT_TEMPLATE.format(code=code)


def get_rag_prompt(context: str, question: str, chat_history: str = "") -> str:
    """Generate a formatted RAG prompt with context and conversation history."""
    return RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
        chat_history=chat_history if chat_history else "No previous conversation"
    )


def get_query_generation_prompt(question: str) -> str:
    """Generate a prompt for creating diverse retrieval queries."""
    return QUERY_GENERATION_TEMPLATE.format(question=question)


def get_code_reference_prompt(response: str) -> str:
    """Generate a prompt for extracting code references from a response."""
    return CODE_REFERENCE_TEMPLATE.format(response=response)


# ============================================================================
# Template Registry (for dynamic template selection)
# ============================================================================

TEMPLATE_REGISTRY: Dict[str, Any] = {
    "qa": {
        "template": CODE_QA_PROMPT_TEMPLATE,
        "formatter": get_qa_prompt,
        "description": "Question & Answer about code with few-shot examples"
    },
    "explanation": {
        "template": CODE_EXPLANATION_PROMPT_TEMPLATE,
        "formatter": get_explanation_prompt,
        "description": "Code explanation template"
    },
    "summarization": {
        "template": CODE_SUMMARIZATION_PROMPT_TEMPLATE,
        "formatter": get_summarization_prompt,
        "description": "Code summarization template"
    },
    "rag": {
        "template": RAG_PROMPT_TEMPLATE,
        "formatter": get_rag_prompt,
        "description": "RAG prompt with context and chat history"
    },
    "query_generation": {
        "template": QUERY_GENERATION_TEMPLATE,
        "formatter": get_query_generation_prompt,
        "description": "Generate diverse retrieval queries"
    },
    "code_reference": {
        "template": CODE_REFERENCE_TEMPLATE,
        "formatter": get_code_reference_prompt,
        "description": "Extract code references from LLM responses"
    }
}


def get_template(template_type: str) -> PromptTemplate:
    """Retrieve a template by type from the registry."""
    if template_type not in TEMPLATE_REGISTRY:
        raise ValueError(f"Unknown template type: {template_type}. Available: {list(TEMPLATE_REGISTRY.keys())}")
    return TEMPLATE_REGISTRY[template_type]["template"]


def get_formatter(template_type: str):
    """Retrieve a formatter function by template type."""
    if template_type not in TEMPLATE_REGISTRY:
        raise ValueError(f"Unknown template type: {template_type}")
    return TEMPLATE_REGISTRY[template_type]["formatter"]
