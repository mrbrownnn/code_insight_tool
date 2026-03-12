"""
Comprehensive Chunker Test Suite — ~50 test cases.
Tests the Chunker module using AST-parsed data from ASTParser.
Each test prints detailed chunk output for RAG pipeline validation.

Covers:
  1. CodeChunk dataclass (creation, to_embedding_text, to_dict)
  2. _classify_node_type (class, function, method, interface, etc.)
  3. chunk_from_ast with real AST data (Python, JS, Java)
  4. _chunk_node — small nodes within token limit
  5. _split_large_node — oversized nodes, sliding window
  6. _chunk_uncovered_lines — imports, top-level code
  7. chunk_fallback — fallback when no AST available
  8. End-to-end: ASTParser → Chunker integration

Run:
  python -m pytest tests/test_chunker.py -v
  Or: python tests/test_chunker.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ingestion.ast_parser import ASTParser, ASTNode
from core.ingestion.chunker import Chunker, CodeChunk
from utils.hash_utils import hash_content
from utils.token_counter import estimate_tokens

# ============================================================
#  Shared instances
# ============================================================
_parser = ASTParser()
# Small max_tokens to easily test split logic
_chunker = Chunker(max_tokens=2048, fallback_window=10, overlap_lines=2)


def _parse(language: str, source: str):
    """Helper — parse source and return list of ASTNode."""
    return _parser.parse_file(
        file_path=Path("test_file"),
        language=language,
        source_code=source,
    )


def _skip_if_unsupported(language: str):
    if not _parser.is_supported(language):
        print(f"SKIP: {language} parser not installed")
        return True
    return False


def _make_node(
    node_type="function_definition",
    name="test_func",
    start_line=1,
    end_line=3,
    source_code="def test_func():\n    pass\n",
    language="python",
    children=None,
    parent_name=None,
):
    """Helper — create an ASTNode for testing."""
    return ASTNode(
        node_type=node_type,
        name=name,
        start_line=start_line,
        end_line=end_line,
        source_code=source_code,
        language=language,
        children=children or [],
        parent_name=parent_name,
    )


def _print_chunk(chunk, index=None):
    """Print detailed chunk info for RAG validation."""
    prefix = f"    [Chunk {index}]" if index is not None else "    [Chunk]"
    print(prefix)
    print(f"      chunk_id:          {chunk.chunk_id[:12]}...")
    print(f"      conversation_id:   {chunk.conversation_id[:12]}...")
    print(f"      conversation_name: {chunk.conversation_name}")
    print(f"      file_path:         {chunk.file_path}")
    print(f"      language:          {chunk.language}")
    print(f"      chunk_type:        {chunk.chunk_type}")
    print(f"      parent_scope:      {chunk.parent_scope}")
    print(f"      lines:             {chunk.line_start}–{chunk.line_end}")
    print(f"      tokens (est.):     {estimate_tokens(chunk.source_code)}")
    print(f"      content_hash:      {chunk.content_hash[:25]}...")
    # Show first 120 chars of source_code
    preview = chunk.source_code[:120].replace("\n", "\\n")
    print(f"      source_preview:    {preview}")
    print(f"      embedding_text:    {chunk.to_embedding_text()[:100]}...")
    print()


def _print_chunks(chunks, label=""):
    """Print all chunks with summary."""
    if label:
        print(f"    --- {label} ---")
    print(f"    Total chunks: {len(chunks)}")
    type_counts = {}
    for c in chunks:
        type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1
    print(f"    By type: {type_counts}")
    print()
    for i, chunk in enumerate(chunks):
        _print_chunk(chunk, i)


def _print_ast_nodes(nodes, label=""):
    """Print AST nodes used as input."""
    if label:
        print(f"    --- {label} ---")
    print(f"    Total AST nodes: {len(nodes)}")
    for i, node in enumerate(nodes):
        print(f"    [Node {i}] {node.node_type} '{node.name}' "
              f"L{node.start_line}–{node.end_line} "
              f"children={len(node.children)} "
              f"parent={node.parent_name}")
    print()


# ============================================================
#  1. CodeChunk DATACLASS TESTS (test_01 — test_05)
# ============================================================

def test_01_codechunk_creation():
    """CodeChunk can be created with all required fields."""
    chunk = CodeChunk(
        conversation_id="conv-1",
        chunk_id="chunk-1",
        file_path="test.py",
        language="python",
        chunk_type="function",
        parent_scope=None,
        line_start=1,
        line_end=3,
        source_code="def foo():\n    pass",
        content_hash="sha256:abc123",
        conversation_name="foo",
    )
    _print_chunk(chunk)
    assert chunk.conversation_id == "conv-1"
    assert chunk.chunk_id == "chunk-1"
    assert chunk.file_path == "test.py"
    assert chunk.language == "python"
    assert chunk.chunk_type == "function"
    assert chunk.conversation_name == "foo"
    assert chunk.line_start == 1
    assert chunk.line_end == 3


def test_02_codechunk_to_embedding_text():
    """to_embedding_text() produces correct format for RAG embedding."""
    chunk = CodeChunk(
        conversation_id="conv-1",
        chunk_id="chunk-1",
        file_path="utils/helper.py",
        language="python",
        chunk_type="function",
        parent_scope=None,
        line_start=1,
        line_end=2,
        source_code="def helper():\n    return True",
        content_hash="sha256:xyz",
        conversation_name="helper",
    )
    text = chunk.to_embedding_text()
    print(f"    Embedding text: '{text}'")
    print(f"    Token estimate: {estimate_tokens(text)}")
    assert "python" in text
    assert "function" in text
    assert "utils/helper.py" in text
    assert "def helper():" in text


def test_03_codechunk_to_dict():
    """to_dict() contains all required keys for vector store."""
    chunk = CodeChunk(
        conversation_id="conv-1",
        chunk_id="chunk-1",
        file_path="test.py",
        language="python",
        chunk_type="function",
        parent_scope="MyClass",
        line_start=10,
        line_end=20,
        source_code="def foo(): pass",
        content_hash="sha256:aaa",
        conversation_name="foo",
    )
    d = chunk.to_dict()
    print(f"    to_dict() keys: {list(d.keys())}")
    for k, v in d.items():
        val_str = str(v)[:80]
        print(f"      {k}: {val_str}")
    assert d["conversation_id"] == "conv-1"
    assert d["chunk_id"] == "chunk-1"
    assert d["file_path"] == "test.py"
    assert d["language"] == "python"
    assert d["chunk_type"] == "function"
    assert d["conversation_name"] == "foo"
    assert d["parent_scope"] == "MyClass"
    assert d["line_start"] == 10
    assert d["line_end"] == 20
    assert d["content_hash"] == "sha256:aaa"
    assert d["source_code"] == "def foo(): pass"


def test_04_codechunk_metadata_default():
    """metadata defaults to empty dict."""
    chunk = CodeChunk(
        conversation_id="c", chunk_id="c", file_path="f",
        language="python", chunk_type="block", parent_scope=None,
        line_start=1, line_end=1, source_code="x",
        content_hash="h", conversation_name="n",
    )
    print(f"    metadata: {chunk.metadata}")
    assert chunk.metadata == {}


def test_05_codechunk_metadata_in_dict():
    """metadata keys are merged into to_dict() output."""
    chunk = CodeChunk(
        conversation_id="c", chunk_id="c", file_path="f",
        language="python", chunk_type="block", parent_scope=None,
        line_start=1, line_end=1, source_code="x",
        content_hash="h", conversation_name="n",
        metadata={"custom_key": "custom_value"},
    )
    d = chunk.to_dict()
    print(f"    to_dict() with metadata: { {k: v for k, v in d.items() if k == 'custom_key'} }")
    assert d["custom_key"] == "custom_value"


# ============================================================
#  2. _classify_node_type TESTS (test_06 — test_14)
# ============================================================

def test_06_classify_class_definition():
    """class_definition → 'class'."""
    result = Chunker._classify_node_type("class_definition")
    print(f"    'class_definition' → '{result}'")
    assert result == "class"


def test_07_classify_class_declaration():
    """class_declaration → 'class'."""
    result = Chunker._classify_node_type("class_declaration")
    print(f"    'class_declaration' → '{result}'")
    assert result == "class"


def test_08_classify_interface_declaration():
    """interface_declaration → 'class'."""
    result = Chunker._classify_node_type("interface_declaration")
    print(f"    'interface_declaration' → '{result}'")
    assert result == "class"


def test_09_classify_function_definition():
    """function_definition → 'function'."""
    result = Chunker._classify_node_type("function_definition")
    print(f"    'function_definition' → '{result}'")
    assert result == "function"


def test_10_classify_function_declaration():
    """function_declaration → 'function'."""
    result = Chunker._classify_node_type("function_declaration")
    print(f"    'function_declaration' → '{result}'")
    assert result == "function"


def test_11_classify_method_definition():
    """method_definition → 'method'."""
    result = Chunker._classify_node_type("method_definition")
    print(f"    'method_definition' → '{result}'")
    assert result == "method"


def test_12_classify_constructor_declaration():
    """constructor_declaration → 'method'."""
    result = Chunker._classify_node_type("constructor_declaration")
    print(f"    'constructor_declaration' → '{result}'")
    assert result == "method"


def test_13_classify_arrow_function():
    """arrow_function → 'function'."""
    result = Chunker._classify_node_type("arrow_function")
    print(f"    'arrow_function' → '{result}'")
    assert result == "function"


def test_14_classify_unknown_type():
    """Unknown type → 'block'."""
    for t in ["variable_declaration", "import_statement", "expression_statement"]:
        result = Chunker._classify_node_type(t)
        print(f"    '{t}' → '{result}'")
        assert result == "block"


# ============================================================
#  3. _chunk_node — SMALL NODES (test_15 — test_20)
# ============================================================

def test_15_chunk_small_function_node():
    """Small function node → single CodeChunk."""
    node = _make_node()
    print(f"    Input node: {node.node_type} '{node.name}' L{node.start_line}–{node.end_line}")
    print(f"    Source tokens: {estimate_tokens(node.source_code)}, max_tokens: {_chunker.max_tokens}")
    chunks = _chunker._chunk_node(node, "test.py", "python")
    _print_chunks(chunks, "Output chunks")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "function"
    assert chunks[0].source_code == node.source_code
    assert chunks[0].line_start == 1
    assert chunks[0].line_end == 3


def test_16_chunk_node_preserves_file_path():
    """Chunk preserves the file_path."""
    node = _make_node()
    chunks = _chunker._chunk_node(node, "src/core/utils.py", "python")
    print(f"    file_path: {chunks[0].file_path}")
    assert chunks[0].file_path == "src/core/utils.py"


def test_17_chunk_node_preserves_language():
    """Chunk preserves the language field."""
    node = _make_node(language="javascript", node_type="function_declaration")
    chunks = _chunker._chunk_node(node, "app.js", "javascript")
    print(f"    language: {chunks[0].language}")
    assert chunks[0].language == "javascript"


def test_18_chunk_node_class_type():
    """Class node → chunk_type = 'class'."""
    node = _make_node(
        node_type="class_definition",
        name="MyClass",
        source_code="class MyClass:\n    pass\n",
    )
    chunks = _chunker._chunk_node(node, "test.py", "python")
    print(f"    node_type: {node.node_type} → chunk_type: {chunks[0].chunk_type}")
    assert chunks[0].chunk_type == "class"


def test_19_chunk_node_has_content_hash():
    """Chunk has a valid content_hash matching hash_content()."""
    node = _make_node()
    chunks = _chunker._chunk_node(node, "test.py", "python")
    expected = hash_content(node.source_code)
    print(f"    content_hash:  {chunks[0].content_hash}")
    print(f"    expected_hash: {expected}")
    assert chunks[0].content_hash.startswith("sha256:")
    assert chunks[0].content_hash == expected


def test_20_chunk_node_parent_scope():
    """Method with parent_name → parent_scope is set."""
    node = _make_node(
        node_type="function_definition",
        name="save",
        parent_name="Repository",
        source_code="def save(self):\n    pass\n",
    )
    chunks = _chunker._chunk_node(node, "test.py", "python")
    print(f"    parent_name (input):   {node.parent_name}")
    print(f"    parent_scope (output): {chunks[0].parent_scope}")
    assert chunks[0].parent_scope == "Repository"


# ============================================================
#  4. _split_large_node TESTS (test_21 — test_25)
# ============================================================

def test_21_split_large_node_produces_multiple_chunks():
    """Node exceeding max_tokens is split into multiple chunks."""
    small_chunker = Chunker(max_tokens=10, fallback_window=5, overlap_lines=1)
    body = "\n".join([f"    x_{i} = {i}" for i in range(20)])
    source = f"def big():\n{body}\n"
    node = _make_node(name="big", source_code=source, start_line=1, end_line=22)
    print(f"    Source tokens: {estimate_tokens(source)}, max_tokens: {small_chunker.max_tokens}")
    chunks = small_chunker._split_large_node(node, "test.py", "python", "function")
    _print_chunks(chunks, "Split chunks")
    assert len(chunks) > 1


def test_22_split_large_node_chunk_type():
    """Split chunks preserve the chunk_type."""
    small_chunker = Chunker(max_tokens=10, fallback_window=5, overlap_lines=1)
    body = "\n".join([f"    line_{i} = {i}" for i in range(20)])
    source = f"def big():\n{body}\n"
    node = _make_node(name="big", source_code=source, start_line=1, end_line=22)
    chunks = small_chunker._split_large_node(node, "test.py", "python", "function")
    types = [c.chunk_type for c in chunks]
    print(f"    All chunk_types: {types}")
    for chunk in chunks:
        assert chunk.chunk_type == "function"


def test_23_split_large_node_naming():
    """Split chunks have _part1, _part2, etc. in conversation_name."""
    small_chunker = Chunker(max_tokens=10, fallback_window=5, overlap_lines=1)
    body = "\n".join([f"    line_{i} = {i}" for i in range(20)])
    source = f"def big():\n{body}\n"
    node = _make_node(name="big", source_code=source, start_line=1, end_line=22)
    chunks = small_chunker._split_large_node(node, "test.py", "python", "function")
    names = [c.conversation_name for c in chunks]
    print(f"    Part names: {names}")
    for i, chunk in enumerate(chunks, 1):
        assert f"big_part{i}" in chunk.conversation_name


def test_24_split_large_node_line_numbers():
    """Split chunks have correct line_start/line_end relative to node."""
    small_chunker = Chunker(max_tokens=10, fallback_window=5, overlap_lines=1)
    body = "\n".join([f"    line_{i} = {i}" for i in range(15)])
    source = f"def func():\n{body}\n"
    node = _make_node(name="func", source_code=source, start_line=5, end_line=21)
    chunks = small_chunker._split_large_node(node, "test.py", "python", "function")
    print(f"    Node start_line: {node.start_line}")
    for i, c in enumerate(chunks):
        print(f"    Chunk {i}: lines {c.line_start}–{c.line_end}")
    assert chunks[0].line_start == 5
    for chunk in chunks:
        assert chunk.line_start <= chunk.line_end


def test_25_split_large_node_all_have_hash():
    """All split chunks have valid content hashes."""
    small_chunker = Chunker(max_tokens=10, fallback_window=5, overlap_lines=1)
    body = "\n".join([f"    x = {i}" for i in range(20)])
    source = f"def func():\n{body}\n"
    node = _make_node(name="func", source_code=source, start_line=1, end_line=22)
    chunks = small_chunker._split_large_node(node, "test.py", "python", "function")
    for i, c in enumerate(chunks):
        print(f"    Chunk {i} hash: {c.content_hash[:30]}...")
        assert c.content_hash.startswith("sha256:")


# ============================================================
#  5. _chunk_uncovered_lines TESTS (test_26 — test_31)
# ============================================================

def test_26_uncovered_imports_block():
    """Import lines not covered by AST nodes are chunked."""
    source = (
        "import os\n"
        "import sys\n"
        "import json\n"
        "import logging\n"
        "\n"
        "def foo():\n"
        "    pass\n"
    )
    covered = {6, 7}
    print(f"    Covered lines: {sorted(covered)}")
    chunks = _chunker._chunk_uncovered_lines(source, "test.py", "python", covered)
    _print_chunks(chunks, "Uncovered line chunks")
    assert len(chunks) >= 1
    assert chunks[0].chunk_type == "block"
    assert "import os" in chunks[0].source_code


def test_27_uncovered_lines_below_threshold():
    """Uncovered blocks with < 3 lines are NOT chunked (threshold=3)."""
    source = "import os\nimport sys\n\ndef foo():\n    pass\n"
    covered = {4, 5}
    print(f"    Uncovered non-empty lines: 2 (< threshold 3)")
    chunks = _chunker._chunk_uncovered_lines(source, "test.py", "python", covered)
    print(f"    Chunks created: {len(chunks)}")
    assert len(chunks) == 0


def test_28_uncovered_lines_all_covered():
    """When all lines are covered, no uncovered chunks are generated."""
    source = "def foo():\n    pass\n"
    covered = {1, 2}
    print(f"    All lines covered: {sorted(covered)}")
    chunks = _chunker._chunk_uncovered_lines(source, "test.py", "python", covered)
    print(f"    Chunks: {len(chunks)}")
    assert len(chunks) == 0


def test_29_uncovered_lines_empty_lines_ignored():
    """Empty lines are skipped, not added to uncovered blocks."""
    source = "\n\nimport os\nimport sys\nimport json\n\n\n"
    covered = set()
    chunks = _chunker._chunk_uncovered_lines(source, "test.py", "python", covered)
    _print_chunks(chunks, "Chunks (empty lines should be ignored)")
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.source_code.strip() != ""


def test_30_uncovered_lines_multiple_blocks():
    """Multiple uncovered blocks in different regions."""
    source = (
        "import os\nimport sys\nimport json\nimport re\n"
        "\n"
        "class Foo:\n    pass\n"
        "\n"
        "CONST_A = 1\nCONST_B = 2\nCONST_C = 3\nCONST_D = 4\n"
    )
    covered = {6, 7}
    chunks = _chunker._chunk_uncovered_lines(source, "test.py", "python", covered)
    _print_chunks(chunks, "Multiple uncovered blocks")
    assert len(chunks) == 2


def test_31_uncovered_lines_naming():
    """Uncovered chunks have conversation_name starting with 'imports_block_'."""
    source = "import a\nimport b\nimport c\nimport d\n\ndef foo():\n    pass\n"
    covered = {6, 7}
    chunks = _chunker._chunk_uncovered_lines(source, "test.py", "python", covered)
    names = [c.conversation_name for c in chunks]
    print(f"    conversation_names: {names}")
    for chunk in chunks:
        assert chunk.conversation_name.startswith("imports_block_")


# ============================================================
#  6. chunk_fallback TESTS (test_32 — test_36)
# ============================================================

def test_32_fallback_basic():
    """Fallback chunking produces chunks from plain text."""
    source = "\n".join([f"line_{i}" for i in range(20)])
    chunks = _chunker.chunk_fallback(source, "readme.txt", "text")
    _print_chunks(chunks, "Fallback chunks")
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.chunk_type == "block"
        assert chunk.file_path == "readme.txt"
        assert chunk.language == "text"


def test_33_fallback_empty_source():
    """Fallback with empty/whitespace source → no chunks."""
    chunks1 = _chunker.chunk_fallback("", "test.py", "python")
    chunks2 = _chunker.chunk_fallback("   \n  \n   ", "test.py", "python")
    print(f"    Empty string → {len(chunks1)} chunks")
    print(f"    Whitespace only → {len(chunks2)} chunks")
    assert len(chunks1) == 0
    assert len(chunks2) == 0


def test_34_fallback_window_size():
    """Fallback chunks respect the fallback_window setting."""
    source = "\n".join([f"line_{i}" for i in range(30)])
    chunks = _chunker.chunk_fallback(source, "test.py", "python")
    print(f"    fallback_window: {_chunker.fallback_window}")
    for i, chunk in enumerate(chunks):
        line_count = chunk.source_code.count("\n") + 1
        print(f"    Chunk {i}: {line_count} lines (L{chunk.line_start}–{chunk.line_end})")
        assert line_count <= _chunker.fallback_window


def test_35_fallback_line_numbers():
    """Fallback chunks have correct 1-indexed line numbers."""
    source = "\n".join([f"line_{i}" for i in range(15)])
    chunks = _chunker.chunk_fallback(source, "test.py", "python")
    print(f"    First chunk starts at line: {chunks[0].line_start}")
    assert chunks[0].line_start == 1
    for chunk in chunks:
        assert chunk.line_start >= 1
        assert chunk.line_end >= chunk.line_start


def test_36_fallback_has_hashes():
    """All fallback chunks have content hashes."""
    source = "\n".join([f"code_{i}" for i in range(20)])
    chunks = _chunker.chunk_fallback(source, "test.py", "python")
    for i, c in enumerate(chunks):
        print(f"    Chunk {i} hash: {c.content_hash[:30]}...")
        assert c.content_hash.startswith("sha256:")


# ============================================================
#  7. chunk_from_ast — REAL AST DATA (test_37 — test_45)
# ============================================================

def test_37_chunk_from_ast_python_function():
    """chunk_from_ast: Python single function → at least 1 chunk."""
    if _skip_if_unsupported("python"): return
    source = "def greet(name):\n    return f'Hello {name}'\n"
    nodes = _parse("python", source)
    _print_ast_nodes(nodes, "AST input")
    chunks = _chunker.chunk_from_ast(nodes, "greet.py", "python", source)
    _print_chunks(chunks, "chunk_from_ast output")
    assert len(chunks) >= 1
    func_chunks = [c for c in chunks if c.chunk_type == "function"]
    assert len(func_chunks) == 1
    assert "def greet" in func_chunks[0].source_code


def test_38_chunk_from_ast_python_class_with_methods():
    """chunk_from_ast: Python class → chunks for class + methods."""
    if _skip_if_unsupported("python"): return
    source = (
        "class Calculator:\n"
        "    def add(self, a, b):\n"
        "        return a + b\n"
        "    def sub(self, a, b):\n"
        "        return a - b\n"
    )
    nodes = _parse("python", source)
    _print_ast_nodes(nodes, "AST input")
    chunks = _chunker.chunk_from_ast(nodes, "calc.py", "python", source)
    _print_chunks(chunks, "chunk_from_ast output")
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    method_chunks = [c for c in chunks if c.chunk_type == "function"]
    assert len(class_chunks) == 1
    assert len(method_chunks) >= 2


def test_39_chunk_from_ast_python_with_imports():
    """chunk_from_ast: Python file with imports → uncovered import chunk."""
    if _skip_if_unsupported("python"): return
    source = (
        "import os\n"
        "import sys\n"
        "import json\n"
        "import logging\n"
        "\n"
        "def process():\n"
        "    pass\n"
    )
    nodes = _parse("python", source)
    _print_ast_nodes(nodes, "AST input")
    chunks = _chunker.chunk_from_ast(nodes, "proc.py", "python", source)
    _print_chunks(chunks, "chunk_from_ast output (with imports)")
    block_chunks = [c for c in chunks if c.chunk_type == "block"]
    func_chunks = [c for c in chunks if c.chunk_type == "function"]
    assert len(func_chunks) >= 1
    assert len(block_chunks) >= 1


def test_40_chunk_from_ast_js_function():
    """chunk_from_ast: JS function declaration."""
    if _skip_if_unsupported("javascript"): return
    source = "function hello() {\n  console.log('hi');\n}\n"
    nodes = _parse("javascript", source)
    _print_ast_nodes(nodes, "AST input (JS)")
    chunks = _chunker.chunk_from_ast(nodes, "app.js", "javascript", source)
    _print_chunks(chunks, "chunk_from_ast output (JS)")
    func_chunks = [c for c in chunks if c.chunk_type == "function"]
    assert len(func_chunks) >= 1


def test_41_chunk_from_ast_js_class():
    """chunk_from_ast: JS class with methods."""
    if _skip_if_unsupported("javascript"): return
    source = (
        "class Animal {\n"
        "  constructor(name) {\n"
        "    this.name = name;\n"
        "  }\n"
        "  speak() {\n"
        "    return this.name;\n"
        "  }\n"
        "}\n"
    )
    nodes = _parse("javascript", source)
    _print_ast_nodes(nodes, "AST input (JS class)")
    chunks = _chunker.chunk_from_ast(nodes, "animal.js", "javascript", source)
    _print_chunks(chunks, "chunk_from_ast output (JS class)")
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert len(class_chunks) >= 1


def test_42_chunk_from_ast_java_class():
    """chunk_from_ast: Java class with method."""
    if _skip_if_unsupported("java"): return
    source = (
        "public class Greeter {\n"
        "    public void greet() {\n"
        "        System.out.println(\"Hello\");\n"
        "    }\n"
        "}\n"
    )
    nodes = _parse("java", source)
    _print_ast_nodes(nodes, "AST input (Java)")
    chunks = _chunker.chunk_from_ast(nodes, "Greeter.java", "java", source)
    _print_chunks(chunks, "chunk_from_ast output (Java)")
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert len(class_chunks) >= 1


def test_43_chunk_from_ast_empty_nodes():
    """chunk_from_ast: empty AST nodes list → only uncovered chunks."""
    source = "import os\nimport sys\nimport json\nimport re\n"
    print(f"    Input: empty AST nodes, source has {len(source.splitlines())} lines")
    chunks = _chunker.chunk_from_ast([], "test.py", "python", source)
    _print_chunks(chunks, "Chunks from empty AST")
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.chunk_type == "block"


def test_44_chunk_from_ast_unique_ids():
    """All chunks from chunk_from_ast have unique chunk_ids."""
    if _skip_if_unsupported("python"): return
    source = (
        "import os\nimport sys\nimport json\nimport re\n\n"
        "def foo():\n    pass\n\n"
        "def bar():\n    pass\n"
    )
    nodes = _parse("python", source)
    chunks = _chunker.chunk_from_ast(nodes, "test.py", "python", source)
    ids = [c.chunk_id for c in chunks]
    print(f"    Total chunks: {len(ids)}")
    print(f"    Unique IDs:   {len(set(ids))}")
    assert len(ids) == len(set(ids)), "chunk_ids must be unique"


def test_45_chunk_from_ast_all_have_conversation_id():
    """All chunks have conversation_id set (required for RAG)."""
    if _skip_if_unsupported("python"): return
    source = "def foo():\n    pass\n"
    nodes = _parse("python", source)
    chunks = _chunker.chunk_from_ast(nodes, "test.py", "python", source)
    for i, chunk in enumerate(chunks):
        print(f"    Chunk {i} conversation_id: {chunk.conversation_id[:12]}...")
        assert chunk.conversation_id is not None
        assert len(chunk.conversation_id) > 0


# ============================================================
#  8. END-TO-END INTEGRATION (test_46 — test_51)
# ============================================================

def test_46_e2e_python_full_file():
    """E2E: Parse a realistic Python file → chunk → verify all fields for RAG."""
    if _skip_if_unsupported("python"): return
    source = (
        "import os\n"
        "import sys\n"
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "class FileProcessor:\n"
        "    def __init__(self, path):\n"
        "        self.path = path\n"
        "\n"
        "    def read(self):\n"
        "        with open(self.path) as f:\n"
        "            return f.read()\n"
        "\n"
        "    def write(self, data):\n"
        "        with open(self.path, 'w') as f:\n"
        "            f.write(data)\n"
        "\n"
        "def main():\n"
        "    proc = FileProcessor('test.txt')\n"
        "    print(proc.read())\n"
    )
    print("    === INPUT SOURCE ===")
    for i, line in enumerate(source.split("\n"), 1):
        print(f"    {i:3d} | {line}")
    print()

    nodes = _parse("python", source)
    _print_ast_nodes(nodes, "AST nodes parsed")

    chunks = _chunker.chunk_from_ast(nodes, "processor.py", "python", source)
    _print_chunks(chunks, "Final chunks for RAG")

    # Verify all chunks have RAG-required fields
    for chunk in chunks:
        assert chunk.conversation_id
        assert chunk.chunk_id
        assert chunk.file_path == "processor.py"
        assert chunk.language == "python"
        assert chunk.chunk_type in ("class", "function", "method", "block")
        assert chunk.line_start >= 1
        assert chunk.line_end >= chunk.line_start
        assert chunk.source_code
        assert chunk.content_hash.startswith("sha256:")


def test_47_e2e_python_class_children_chunked():
    """E2E: Class children (methods) are also chunked separately for granular retrieval."""
    if _skip_if_unsupported("python"): return
    source = (
        "class Service:\n"
        "    def start(self):\n"
        "        pass\n"
        "    def stop(self):\n"
        "        pass\n"
    )
    nodes = _parse("python", source)
    _print_ast_nodes(nodes, "AST nodes")
    chunks = _chunker.chunk_from_ast(nodes, "svc.py", "python", source)
    _print_chunks(chunks, "Chunked output")

    assert len(chunks) >= 3
    types = [c.chunk_type for c in chunks]
    print(f"    Chunk types: {types}")
    assert "class" in types
    assert types.count("function") >= 2


def test_48_e2e_js_full_file():
    """E2E: Parse realistic JS file → chunk → verify for RAG."""
    if _skip_if_unsupported("javascript"): return
    source = (
        "class EventEmitter {\n"
        "  constructor() {\n"
        "    this.events = {};\n"
        "  }\n"
        "\n"
        "  on(event, callback) {\n"
        "    if (!this.events[event]) this.events[event] = [];\n"
        "    this.events[event].push(callback);\n"
        "  }\n"
        "\n"
        "  emit(event, ...args) {\n"
        "    if (this.events[event]) {\n"
        "      this.events[event].forEach(cb => cb(...args));\n"
        "    }\n"
        "  }\n"
        "}\n"
    )
    nodes = _parse("javascript", source)
    _print_ast_nodes(nodes, "JS AST nodes")
    chunks = _chunker.chunk_from_ast(nodes, "emitter.js", "javascript", source)
    _print_chunks(chunks, "JS chunks for RAG")
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.language == "javascript"
        assert chunk.content_hash.startswith("sha256:")


def test_49_e2e_java_interface_and_class():
    """E2E: Java interface + class → chunks for both."""
    if _skip_if_unsupported("java"): return
    source = (
        "public interface Drawable {\n"
        "    void draw();\n"
        "}\n"
        "\n"
        "public class Circle implements Drawable {\n"
        "    public void draw() {\n"
        "        System.out.println(\"Drawing circle\");\n"
        "    }\n"
        "}\n"
    )
    nodes = _parse("java", source)
    _print_ast_nodes(nodes, "Java AST nodes")
    chunks = _chunker.chunk_from_ast(nodes, "Shape.java", "java", source)
    _print_chunks(chunks, "Java chunks for RAG")
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert len(class_chunks) >= 2


def test_50_e2e_chunker_custom_config():
    """E2E: Chunker with custom config works correctly."""
    if _skip_if_unsupported("python"): return
    custom_chunker = Chunker(max_tokens=4096, fallback_window=50, overlap_lines=5)
    print(f"    Custom config: max_tokens={custom_chunker.max_tokens}, "
          f"window={custom_chunker.fallback_window}, overlap={custom_chunker.overlap_lines}")
    source = "def simple():\n    return 42\n"
    nodes = _parse("python", source)
    chunks = custom_chunker.chunk_from_ast(nodes, "simple.py", "python", source)
    _print_chunks(chunks, "Custom config output")
    assert len(chunks) >= 1


def test_51_e2e_fallback_vs_ast_coverage():
    """E2E: Compare AST chunking vs fallback — AST has semantic types for better RAG retrieval."""
    if _skip_if_unsupported("python"): return
    source = (
        "import os\nimport sys\nimport json\nimport re\n\n"
        "class Foo:\n"
        "    def bar(self):\n"
        "        return 42\n"
        "\n"
        "def baz():\n"
        "    pass\n"
    )
    nodes = _parse("python", source)
    ast_chunks = _chunker.chunk_from_ast(nodes, "test.py", "python", source)
    fallback_chunks = _chunker.chunk_fallback(source, "test.py", "python")

    print("    === AST CHUNKING ===")
    ast_types = {c.chunk_type for c in ast_chunks}
    print(f"    Chunk types: {ast_types}")
    print(f"    Total chunks: {len(ast_chunks)}")
    for c in ast_chunks:
        print(f"      [{c.chunk_type:8s}] {c.conversation_name} (L{c.line_start}–{c.line_end})")

    print("\n    === FALLBACK CHUNKING ===")
    fb_types = {c.chunk_type for c in fallback_chunks}
    print(f"    Chunk types: {fb_types}")
    print(f"    Total chunks: {len(fallback_chunks)}")
    for c in fallback_chunks:
        print(f"      [{c.chunk_type:8s}] {c.conversation_name} (L{c.line_start}–{c.line_end})")

    print("\n    → AST chunking provides semantic labels for better RAG retrieval")
    assert "function" in ast_types or "class" in ast_types
    for c in fallback_chunks:
        assert c.chunk_type == "block"


# ============================================================
#  RUNNER
# ============================================================

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0

    for test_fn in tests:
        name = test_fn.__name__
        doc = test_fn.__doc__ or ""
        print(f"\n{'─'*60}")
        print(f"  ▶ {name}: {doc.strip()}")
        print(f"{'─'*60}")
        try:
            test_fn()
            passed += 1
            print(f"  ✅ PASS")
        except AssertionError as e:
            failed += 1
            print(f"  ❌ FAIL => {e}")
        except Exception as e:
            failed += 1
            print(f"  ❌ ERROR => {type(e).__name__}: {e}")

    total = passed + failed
    print(f"\n{'═'*60}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("  🎉 ALL TESTS PASSED — Chunker ready for RAG pipeline!")
    else:
        print(f"  ⚠️ FAILURES: {failed}")
    print(f"{'═'*60}")
