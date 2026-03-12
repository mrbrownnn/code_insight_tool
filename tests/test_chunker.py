"""
Comprehensive Chunker Test Suite — ~50 test cases.
Tests the Chunker module using AST-parsed data from ASTParser.

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
    assert chunk.conversation_id == "conv-1"
    assert chunk.chunk_id == "chunk-1"
    assert chunk.file_path == "test.py"
    assert chunk.language == "python"
    assert chunk.chunk_type == "function"
    assert chunk.conversation_name == "foo"
    assert chunk.line_start == 1
    assert chunk.line_end == 3


def test_02_codechunk_to_embedding_text():
    """to_embedding_text() produces correct format."""
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
    assert "python" in text
    assert "function" in text
    assert "utils/helper.py" in text
    assert "def helper():" in text


def test_03_codechunk_to_dict():
    """to_dict() contains all required keys."""
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
    assert d["custom_key"] == "custom_value"


# ============================================================
#  2. _classify_node_type TESTS (test_06 — test_13)
# ============================================================

def test_06_classify_class_definition():
    """class_definition → 'class'."""
    assert Chunker._classify_node_type("class_definition") == "class"


def test_07_classify_class_declaration():
    """class_declaration → 'class'."""
    assert Chunker._classify_node_type("class_declaration") == "class"


def test_08_classify_interface_declaration():
    """interface_declaration → 'class'."""
    assert Chunker._classify_node_type("interface_declaration") == "class"


def test_09_classify_function_definition():
    """function_definition → 'function'."""
    assert Chunker._classify_node_type("function_definition") == "function"


def test_10_classify_function_declaration():
    """function_declaration → 'function'."""
    assert Chunker._classify_node_type("function_declaration") == "function"


def test_11_classify_method_definition():
    """method_definition → 'method'."""
    assert Chunker._classify_node_type("method_definition") == "method"


def test_12_classify_constructor_declaration():
    """constructor_declaration → 'method'."""
    assert Chunker._classify_node_type("constructor_declaration") == "method"


def test_13_classify_arrow_function():
    """arrow_function → 'function'."""
    assert Chunker._classify_node_type("arrow_function") == "function"


def test_14_classify_unknown_type():
    """Unknown type → 'block'."""
    assert Chunker._classify_node_type("variable_declaration") == "block"
    assert Chunker._classify_node_type("import_statement") == "block"
    assert Chunker._classify_node_type("expression_statement") == "block"


# ============================================================
#  3. _chunk_node — SMALL NODES (test_15 — test_19)
# ============================================================

def test_15_chunk_small_function_node():
    """Small function node → single CodeChunk."""
    node = _make_node()
    chunks = _chunker._chunk_node(node, "test.py", "python")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "function"
    assert chunks[0].source_code == node.source_code
    assert chunks[0].line_start == 1
    assert chunks[0].line_end == 3


def test_16_chunk_node_preserves_file_path():
    """Chunk preserves the file_path."""
    node = _make_node()
    chunks = _chunker._chunk_node(node, "src/core/utils.py", "python")
    assert chunks[0].file_path == "src/core/utils.py"


def test_17_chunk_node_preserves_language():
    """Chunk preserves the language field."""
    node = _make_node(language="javascript", node_type="function_declaration")
    chunks = _chunker._chunk_node(node, "app.js", "javascript")
    assert chunks[0].language == "javascript"


def test_18_chunk_node_class_type():
    """Class node → chunk_type = 'class'."""
    node = _make_node(
        node_type="class_definition",
        name="MyClass",
        source_code="class MyClass:\n    pass\n",
    )
    chunks = _chunker._chunk_node(node, "test.py", "python")
    assert chunks[0].chunk_type == "class"


def test_19_chunk_node_has_content_hash():
    """Chunk has a valid content_hash starting with 'sha256:'."""
    node = _make_node()
    chunks = _chunker._chunk_node(node, "test.py", "python")
    assert chunks[0].content_hash.startswith("sha256:")
    assert chunks[0].content_hash == hash_content(node.source_code)


def test_20_chunk_node_parent_scope():
    """Method with parent_name → parent_scope is set."""
    node = _make_node(
        node_type="function_definition",
        name="save",
        parent_name="Repository",
        source_code="def save(self):\n    pass\n",
    )
    chunks = _chunker._chunk_node(node, "test.py", "python")
    assert chunks[0].parent_scope == "Repository"


# ============================================================
#  4. _split_large_node TESTS (test_21 — test_25)
# ============================================================

def test_21_split_large_node_produces_multiple_chunks():
    """Node exceeding max_tokens is split into multiple chunks."""
    # Create a chunker with very small token limit
    small_chunker = Chunker(max_tokens=10, fallback_window=5, overlap_lines=1)
    body = "\n".join([f"    x_{i} = {i}" for i in range(20)])
    source = f"def big():\n{body}\n"
    node = _make_node(
        name="big",
        source_code=source,
        start_line=1,
        end_line=22,
    )
    chunks = small_chunker._split_large_node(node, "test.py", "python", "function")
    assert len(chunks) > 1


def test_22_split_large_node_chunk_type():
    """Split chunks preserve the chunk_type."""
    small_chunker = Chunker(max_tokens=10, fallback_window=5, overlap_lines=1)
    body = "\n".join([f"    line_{i} = {i}" for i in range(20)])
    source = f"def big():\n{body}\n"
    node = _make_node(name="big", source_code=source, start_line=1, end_line=22)
    chunks = small_chunker._split_large_node(node, "test.py", "python", "function")
    for chunk in chunks:
        assert chunk.chunk_type == "function"


def test_23_split_large_node_naming():
    """Split chunks have _part1, _part2, etc. in conversation_name."""
    small_chunker = Chunker(max_tokens=10, fallback_window=5, overlap_lines=1)
    body = "\n".join([f"    line_{i} = {i}" for i in range(20)])
    source = f"def big():\n{body}\n"
    node = _make_node(name="big", source_code=source, start_line=1, end_line=22)
    chunks = small_chunker._split_large_node(node, "test.py", "python", "function")
    for i, chunk in enumerate(chunks, 1):
        assert f"big_part{i}" in chunk.conversation_name


def test_24_split_large_node_line_numbers():
    """Split chunks have correct line_start/line_end relative to node."""
    small_chunker = Chunker(max_tokens=10, fallback_window=5, overlap_lines=1)
    body = "\n".join([f"    line_{i} = {i}" for i in range(15)])
    source = f"def func():\n{body}\n"
    node = _make_node(name="func", source_code=source, start_line=5, end_line=21)
    chunks = small_chunker._split_large_node(node, "test.py", "python", "function")
    # First chunk should start at node's start_line
    assert chunks[0].line_start == 5
    # Each chunk should have valid line ranges
    for chunk in chunks:
        assert chunk.line_start <= chunk.line_end


def test_25_split_large_node_all_have_hash():
    """All split chunks have valid content hashes."""
    small_chunker = Chunker(max_tokens=10, fallback_window=5, overlap_lines=1)
    body = "\n".join([f"    x = {i}" for i in range(20)])
    source = f"def func():\n{body}\n"
    node = _make_node(name="func", source_code=source, start_line=1, end_line=22)
    chunks = small_chunker._split_large_node(node, "test.py", "python", "function")
    for chunk in chunks:
        assert chunk.content_hash.startswith("sha256:")


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
    # Lines 6-7 are covered by function node
    covered = {6, 7}
    chunks = _chunker._chunk_uncovered_lines(source, "test.py", "python", covered)
    # 4 import lines >= 3 threshold → should create a chunk
    assert len(chunks) >= 1
    assert chunks[0].chunk_type == "block"
    assert "import os" in chunks[0].source_code


def test_27_uncovered_lines_below_threshold():
    """Uncovered blocks with < 3 lines are NOT chunked."""
    source = "import os\nimport sys\n\ndef foo():\n    pass\n"
    covered = {4, 5}
    chunks = _chunker._chunk_uncovered_lines(source, "test.py", "python", covered)
    # 2 import lines < 3 threshold → no chunk
    assert len(chunks) == 0


def test_28_uncovered_lines_all_covered():
    """When all lines are covered, no uncovered chunks are generated."""
    source = "def foo():\n    pass\n"
    covered = {1, 2}
    chunks = _chunker._chunk_uncovered_lines(source, "test.py", "python", covered)
    assert len(chunks) == 0


def test_29_uncovered_lines_empty_lines_ignored():
    """Empty lines are skipped, not added to uncovered blocks."""
    source = "\n\nimport os\nimport sys\nimport json\n\n\n"
    covered = set()
    chunks = _chunker._chunk_uncovered_lines(source, "test.py", "python", covered)
    # 3 import lines >= threshold, empty lines are ignored
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
    # Class covers lines 6-7
    covered = {6, 7}
    chunks = _chunker._chunk_uncovered_lines(source, "test.py", "python", covered)
    # Should have 2 blocks: imports (lines 1-4) and constants (lines 9-12)
    assert len(chunks) == 2


def test_31_uncovered_lines_naming():
    """Uncovered chunks have conversation_name starting with 'imports_block_'."""
    source = "import a\nimport b\nimport c\nimport d\n\ndef foo():\n    pass\n"
    covered = {6, 7}
    chunks = _chunker._chunk_uncovered_lines(source, "test.py", "python", covered)
    for chunk in chunks:
        assert chunk.conversation_name.startswith("imports_block_")


# ============================================================
#  6. chunk_fallback TESTS (test_32 — test_36)
# ============================================================

def test_32_fallback_basic():
    """Fallback chunking produces chunks from plain text."""
    source = "\n".join([f"line_{i}" for i in range(20)])
    chunks = _chunker.chunk_fallback(source, "readme.txt", "text")
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.chunk_type == "block"
        assert chunk.file_path == "readme.txt"
        assert chunk.language == "text"


def test_33_fallback_empty_source():
    """Fallback with empty/whitespace source → no chunks."""
    chunks = _chunker.chunk_fallback("", "test.py", "python")
    assert len(chunks) == 0
    chunks2 = _chunker.chunk_fallback("   \n  \n   ", "test.py", "python")
    assert len(chunks2) == 0


def test_34_fallback_window_size():
    """Fallback chunks respect the fallback_window setting."""
    # Window = 10, so each chunk covers at most 10 lines
    source = "\n".join([f"line_{i}" for i in range(30)])
    chunks = _chunker.chunk_fallback(source, "test.py", "python")
    for chunk in chunks:
        line_count = chunk.source_code.count("\n") + 1
        assert line_count <= _chunker.fallback_window


def test_35_fallback_line_numbers():
    """Fallback chunks have correct 1-indexed line numbers."""
    source = "\n".join([f"line_{i}" for i in range(15)])
    chunks = _chunker.chunk_fallback(source, "test.py", "python")
    assert chunks[0].line_start == 1
    for chunk in chunks:
        assert chunk.line_start >= 1
        assert chunk.line_end >= chunk.line_start


def test_36_fallback_has_hashes():
    """All fallback chunks have content hashes."""
    source = "\n".join([f"code_{i}" for i in range(20)])
    chunks = _chunker.chunk_fallback(source, "test.py", "python")
    for chunk in chunks:
        assert chunk.content_hash.startswith("sha256:")


# ============================================================
#  7. chunk_from_ast — REAL AST DATA (test_37 — test_45)
# ============================================================

def test_37_chunk_from_ast_python_function():
    """chunk_from_ast: Python single function → at least 1 chunk."""
    if _skip_if_unsupported("python"): return
    source = "def greet(name):\n    return f'Hello {name}'\n"
    nodes = _parse("python", source)
    chunks = _chunker.chunk_from_ast(nodes, "greet.py", "python", source)
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
    chunks = _chunker.chunk_from_ast(nodes, "calc.py", "python", source)
    # Should have: class chunk + method chunks (add, sub)
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
    chunks = _chunker.chunk_from_ast(nodes, "proc.py", "python", source)
    # Should include: function chunk + uncovered import block
    block_chunks = [c for c in chunks if c.chunk_type == "block"]
    func_chunks = [c for c in chunks if c.chunk_type == "function"]
    assert len(func_chunks) >= 1
    assert len(block_chunks) >= 1


def test_40_chunk_from_ast_js_function():
    """chunk_from_ast: JS function declaration."""
    if _skip_if_unsupported("javascript"): return
    source = "function hello() {\n  console.log('hi');\n}\n"
    nodes = _parse("javascript", source)
    chunks = _chunker.chunk_from_ast(nodes, "app.js", "javascript", source)
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
    chunks = _chunker.chunk_from_ast(nodes, "animal.js", "javascript", source)
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
    chunks = _chunker.chunk_from_ast(nodes, "Greeter.java", "java", source)
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert len(class_chunks) >= 1


def test_43_chunk_from_ast_empty_nodes():
    """chunk_from_ast: empty AST nodes list → only uncovered chunks."""
    source = "import os\nimport sys\nimport json\nimport re\n"
    chunks = _chunker.chunk_from_ast([], "test.py", "python", source)
    # All lines are uncovered, >= 3 lines → should create a block
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
    assert len(ids) == len(set(ids)), "chunk_ids must be unique"


def test_45_chunk_from_ast_all_have_conversation_id():
    """All chunks have conversation_id set."""
    if _skip_if_unsupported("python"): return
    source = "def foo():\n    pass\n"
    nodes = _parse("python", source)
    chunks = _chunker.chunk_from_ast(nodes, "test.py", "python", source)
    for chunk in chunks:
        assert chunk.conversation_id is not None
        assert len(chunk.conversation_id) > 0


# ============================================================
#  8. END-TO-END INTEGRATION (test_46 — test_51)
# ============================================================

def test_46_e2e_python_full_file():
    """E2E: Parse a realistic Python file → chunk → verify all chunks valid."""
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
    nodes = _parse("python", source)
    chunks = _chunker.chunk_from_ast(nodes, "processor.py", "python", source)

    # Verify all chunks have required fields
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
    """E2E: Class children (methods) are also chunked separately."""
    if _skip_if_unsupported("python"): return
    source = (
        "class Service:\n"
        "    def start(self):\n"
        "        pass\n"
        "    def stop(self):\n"
        "        pass\n"
    )
    nodes = _parse("python", source)
    chunks = _chunker.chunk_from_ast(nodes, "svc.py", "python", source)
    # Should have class + 2 methods = at least 3 chunks
    assert len(chunks) >= 3
    types = [c.chunk_type for c in chunks]
    assert "class" in types
    assert types.count("function") >= 2


def test_48_e2e_js_full_file():
    """E2E: Parse realistic JS file → chunk → verify."""
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
    chunks = _chunker.chunk_from_ast(nodes, "emitter.js", "javascript", source)
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
    chunks = _chunker.chunk_from_ast(nodes, "Shape.java", "java", source)
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert len(class_chunks) >= 2  # interface + class both map to 'class'


def test_50_e2e_chunker_custom_config():
    """E2E: Chunker with custom config works correctly."""
    if _skip_if_unsupported("python"): return
    custom_chunker = Chunker(max_tokens=4096, fallback_window=50, overlap_lines=5)
    source = "def simple():\n    return 42\n"
    nodes = _parse("python", source)
    chunks = custom_chunker.chunk_from_ast(nodes, "simple.py", "python", source)
    assert len(chunks) >= 1


def test_51_e2e_fallback_vs_ast_coverage():
    """E2E: AST chunking covers more semantics than fallback."""
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

    # AST chunks should have semantic types
    ast_types = {c.chunk_type for c in ast_chunks}
    assert "function" in ast_types or "class" in ast_types

    # Fallback chunks should all be 'block'
    for c in fallback_chunks:
        assert c.chunk_type == "block"


# ============================================================
#  RUNNER
# ============================================================

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    skipped = 0

    for test_fn in tests:
        name = test_fn.__name__
        doc = test_fn.__doc__ or ""
        try:
            test_fn()
            passed += 1
            print(f"  PASS  {name}: {doc.strip()}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {doc.strip()} => {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {doc.strip()} => {type(e).__name__}: {e}")

    total = passed + failed
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED!")
    else:
        print(f"FAILURES: {failed}")
