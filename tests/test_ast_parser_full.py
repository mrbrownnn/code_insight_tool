"""
Comprehensive AST Parser Test Suite — 100 test cases.
Covers Python, JavaScript, Java parsing + edge cases.

Run: python -m pytest tests/test_ast_parser_full.py -v
  Or: python tests/test_ast_parser_full.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ingestion.ast_parser import ASTParser, ASTNode

# ============================================================
#  Shared parser instance (initialized once)
# ============================================================
_parser = ASTParser()


def _parse(language: str, source: str):
    """Helper — parse source code and return list of ASTNode."""
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


# ============================================================
#  PYTHON TESTS (1–35)
# ============================================================

def test_01_simple_function():
    """Python: single simple function."""
    if _skip_if_unsupported("python"): return
    nodes = _parse("python", "def greet():\n    pass\n")
    assert len(nodes) == 1
    assert nodes[0].name == "greet"
    assert nodes[0].node_type == "function_definition"


def test_02_function_with_args():
    """Python: function with multiple arguments."""
    if _skip_if_unsupported("python"): return
    src = "def add(a, b, c=0):\n    return a + b + c\n"
    nodes = _parse("python", src)
    assert len(nodes) == 1
    assert nodes[0].name == "add"


def test_03_function_with_type_hints():
    """Python: function with type annotations."""
    if _skip_if_unsupported("python"): return
    src = "def process(data: list[str]) -> bool:\n    return True\n"
    nodes = _parse("python", src)
    assert len(nodes) == 1
    assert nodes[0].name == "process"


def test_04_async_function():
    """Python: async function definition."""
    if _skip_if_unsupported("python"): return
    src = "async def fetch_data(url: str):\n    return await get(url)\n"
    nodes = _parse("python", src)
    assert len(nodes) == 1
    assert nodes[0].name == "fetch_data"


def test_05_simple_class():
    """Python: simple class with no methods."""
    if _skip_if_unsupported("python"): return
    src = "class Empty:\n    pass\n"
    nodes = _parse("python", src)
    assert len(nodes) == 1
    assert nodes[0].name == "Empty"
    assert nodes[0].node_type == "class_definition"


def test_06_class_with_init():
    """Python: class with __init__."""
    if _skip_if_unsupported("python"): return
    src = "class User:\n    def __init__(self, name):\n        self.name = name\n"
    nodes = _parse("python", src)
    assert len(nodes) == 1
    assert len(nodes[0].children) == 1
    assert nodes[0].children[0].name == "__init__"


def test_07_class_multiple_methods():
    """Python: class with multiple methods."""
    if _skip_if_unsupported("python"): return
    src = (
        "class Calculator:\n"
        "    def add(self, a, b):\n        return a + b\n"
        "    def sub(self, a, b):\n        return a - b\n"
        "    def mul(self, a, b):\n        return a * b\n"
        "    def div(self, a, b):\n        return a / b\n"
    )
    nodes = _parse("python", src)
    assert len(nodes) == 1
    assert len(nodes[0].children) == 4
    names = [c.name for c in nodes[0].children]
    assert names == ["add", "sub", "mul", "div"]


def test_08_class_parent_name_propagation():
    """Python: children of class should have parent_name set."""
    if _skip_if_unsupported("python"): return
    src = "class Repo:\n    def save(self):\n        pass\n    def load(self):\n        pass\n"
    nodes = _parse("python", src)
    for child in nodes[0].children:
        assert child.parent_name == "Repo"


def test_09_class_inheritance():
    """Python: class with inheritance."""
    if _skip_if_unsupported("python"): return
    src = "class Admin(User):\n    def promote(self):\n        pass\n"
    nodes = _parse("python", src)
    assert nodes[0].name == "Admin"
    assert nodes[0].node_type == "class_definition"


def test_10_multiple_classes():
    """Python: multiple classes in one file."""
    if _skip_if_unsupported("python"): return
    src = (
        "class Dog:\n    def bark(self):\n        pass\n\n"
        "class Cat:\n    def meow(self):\n        pass\n"
    )
    nodes = _parse("python", src)
    assert len(nodes) == 2
    assert nodes[0].name == "Dog"
    assert nodes[1].name == "Cat"


def test_11_function_and_class_mixed():
    """Python: functions and classes mixed in one file."""
    if _skip_if_unsupported("python"): return
    src = (
        "def helper():\n    pass\n\n"
        "class Service:\n    def run(self):\n        pass\n\n"
        "def another_helper():\n    pass\n"
    )
    nodes = _parse("python", src)
    assert len(nodes) == 3
    types = [n.node_type for n in nodes]
    assert types == ["function_definition", "class_definition", "function_definition"]


def test_12_decorated_function():
    """Python: function with decorator."""
    if _skip_if_unsupported("python"): return
    src = "@app.route('/api')\ndef api_handler():\n    return 'ok'\n"
    nodes = _parse("python", src)
    assert len(nodes) == 1
    assert nodes[0].name == "api_handler"


def test_13_decorated_class():
    """Python: class with decorator."""
    if _skip_if_unsupported("python"): return
    src = "@dataclass\nclass Point:\n    x: float\n    y: float\n"
    nodes = _parse("python", src)
    assert len(nodes) == 1
    assert nodes[0].name == "Point"


def test_14_staticmethod():
    """Python: class with staticmethod."""
    if _skip_if_unsupported("python"): return
    src = (
        "class Utils:\n"
        "    @staticmethod\n"
        "    def format_date(d):\n"
        "        return str(d)\n"
    )
    nodes = _parse("python", src)
    assert len(nodes[0].children) == 1
    assert nodes[0].children[0].name == "format_date"


def test_15_classmethod():
    """Python: class with classmethod."""
    if _skip_if_unsupported("python"): return
    src = (
        "class Factory:\n"
        "    @classmethod\n"
        "    def create(cls):\n"
        "        return cls()\n"
    )
    nodes = _parse("python", src)
    assert nodes[0].children[0].name == "create"


def test_16_property_accessor():
    """Python: class with @property."""
    if _skip_if_unsupported("python"): return
    src = (
        "class Circle:\n"
        "    @property\n"
        "    def area(self):\n"
        "        return 3.14 * self.r ** 2\n"
    )
    nodes = _parse("python", src)
    assert nodes[0].children[0].name == "area"


def test_17_nested_function():
    """Python: nested function inside another function."""
    if _skip_if_unsupported("python"): return
    src = "def outer():\n    def inner():\n        pass\n    inner()\n"
    nodes = _parse("python", src)
    # outer is extracted; inner is inside outer's body
    assert len(nodes) == 1
    assert nodes[0].name == "outer"


def test_18_lambda_not_extracted():
    """Python: lambda should NOT be extracted as a node."""
    if _skip_if_unsupported("python"): return
    src = "square = lambda x: x ** 2\n"
    nodes = _parse("python", src)
    assert len(nodes) == 0  # lambda is not in _EXTRACT_TYPES


def test_19_multiline_function():
    """Python: function with many lines."""
    if _skip_if_unsupported("python"): return
    body = "\n".join([f"    x = {i}" for i in range(50)])
    src = f"def big_func():\n{body}\n    return x\n"
    nodes = _parse("python", src)
    assert len(nodes) == 1
    assert nodes[0].start_line == 1
    assert nodes[0].end_line == 52  # 1 def + 50 assignments + 1 return


def test_20_correct_line_numbers():
    """Python: line numbers should be 1-indexed and accurate."""
    if _skip_if_unsupported("python"): return
    src = "# comment\n\ndef foo():\n    pass\n\ndef bar():\n    pass\n"
    nodes = _parse("python", src)
    assert len(nodes) == 2
    assert nodes[0].name == "foo"
    assert nodes[0].start_line == 3
    assert nodes[0].end_line == 4
    assert nodes[1].name == "bar"
    assert nodes[1].start_line == 6
    assert nodes[1].end_line == 7


def test_21_source_code_content():
    """Python: extracted source_code should match the actual code."""
    if _skip_if_unsupported("python"): return
    src = "def add(a, b):\n    return a + b\n"
    nodes = _parse("python", src)
    assert "def add(a, b):" in nodes[0].source_code
    assert "return a + b" in nodes[0].source_code


def test_22_class_source_includes_body():
    """Python: class source_code should include the entire class body."""
    if _skip_if_unsupported("python"): return
    src = "class A:\n    x = 1\n    def m(self):\n        pass\n"
    nodes = _parse("python", src)
    assert "class A:" in nodes[0].source_code
    assert "def m(self):" in nodes[0].source_code


def test_23_empty_class():
    """Python: class with only pass."""
    if _skip_if_unsupported("python"): return
    src = "class Nothing:\n    pass\n"
    nodes = _parse("python", src)
    assert len(nodes) == 1
    assert nodes[0].children == []


def test_24_dunder_methods():
    """Python: class with multiple dunder methods."""
    if _skip_if_unsupported("python"): return
    src = (
        "class MyList:\n"
        "    def __init__(self):\n        self.items = []\n"
        "    def __len__(self):\n        return len(self.items)\n"
        "    def __getitem__(self, i):\n        return self.items[i]\n"
        "    def __repr__(self):\n        return str(self.items)\n"
    )
    nodes = _parse("python", src)
    names = [c.name for c in nodes[0].children]
    assert "__init__" in names
    assert "__len__" in names
    assert "__getitem__" in names
    assert "__repr__" in names


def test_25_function_with_docstring():
    """Python: function with a docstring."""
    if _skip_if_unsupported("python"): return
    src = 'def documented():\n    """This is a docstring."""\n    pass\n'
    nodes = _parse("python", src)
    assert len(nodes) == 1
    assert '"""' in nodes[0].source_code


def test_26_function_with_default_args():
    """Python: function with default values of different types."""
    if _skip_if_unsupported("python"): return
    src = "def config(host='localhost', port=8080, debug=False):\n    pass\n"
    nodes = _parse("python", src)
    assert nodes[0].name == "config"


def test_27_function_with_star_args():
    """Python: function with *args and **kwargs."""
    if _skip_if_unsupported("python"): return
    src = "def flexible(*args, **kwargs):\n    pass\n"
    nodes = _parse("python", src)
    assert nodes[0].name == "flexible"


def test_28_async_method_in_class():
    """Python: async method inside a class."""
    if _skip_if_unsupported("python"): return
    src = (
        "class AsyncService:\n"
        "    async def fetch(self, url):\n"
        "        return await self.client.get(url)\n"
    )
    nodes = _parse("python", src)
    assert len(nodes[0].children) == 1
    assert nodes[0].children[0].name == "fetch"


def test_29_multiple_decorators():
    """Python: function with multiple stacked decorators."""
    if _skip_if_unsupported("python"): return
    src = (
        "@login_required\n"
        "@permission('admin')\n"
        "@log_call\n"
        "def admin_action():\n"
        "    pass\n"
    )
    nodes = _parse("python", src)
    assert len(nodes) == 1
    assert nodes[0].name == "admin_action"


def test_30_class_with_class_variable():
    """Python: class with class-level variables and methods."""
    if _skip_if_unsupported("python"): return
    src = (
        "class Config:\n"
        "    DEBUG = True\n"
        "    VERSION = '1.0'\n"
        "    def get_version(self):\n"
        "        return self.VERSION\n"
    )
    nodes = _parse("python", src)
    assert len(nodes[0].children) == 1
    assert nodes[0].children[0].name == "get_version"


def test_31_abstract_class():
    """Python: abstract class pattern."""
    if _skip_if_unsupported("python"): return
    src = (
        "from abc import ABC, abstractmethod\n\n"
        "class Shape(ABC):\n"
        "    @abstractmethod\n"
        "    def area(self):\n"
        "        pass\n"
        "    @abstractmethod\n"
        "    def perimeter(self):\n"
        "        pass\n"
    )
    nodes = _parse("python", src)
    class_node = [n for n in nodes if n.node_type == "class_definition"][0]
    assert class_node.name == "Shape"
    assert len(class_node.children) == 2


def test_32_generator_function():
    """Python: generator function with yield."""
    if _skip_if_unsupported("python"): return
    src = "def count_up(n):\n    for i in range(n):\n        yield i\n"
    nodes = _parse("python", src)
    assert nodes[0].name == "count_up"


def test_33_context_manager_function():
    """Python: function using context manager pattern."""
    if _skip_if_unsupported("python"): return
    src = (
        "def open_db():\n"
        "    conn = connect()\n"
        "    try:\n"
        "        yield conn\n"
        "    finally:\n"
        "        conn.close()\n"
    )
    nodes = _parse("python", src)
    assert nodes[0].name == "open_db"


def test_34_language_field_correct():
    """Python: ASTNode.language should be 'python'."""
    if _skip_if_unsupported("python"): return
    nodes = _parse("python", "def x():\n    pass\n")
    assert nodes[0].language == "python"


def test_35_top_level_function_no_parent():
    """Python: top-level function should have parent_name=None."""
    if _skip_if_unsupported("python"): return
    nodes = _parse("python", "def standalone():\n    pass\n")
    assert nodes[0].parent_name is None


# ============================================================
#  JAVASCRIPT TESTS (36–60)
# ============================================================

def test_36_js_function_declaration():
    """JS: function declaration."""
    if _skip_if_unsupported("javascript"): return
    src = "function sayHello() {\n  console.log('hi');\n}\n"
    nodes = _parse("javascript", src)
    assert len(nodes) == 1
    assert nodes[0].name == "sayHello"
    assert nodes[0].node_type == "function_declaration"


def test_37_js_function_with_params():
    """JS: function with parameters."""
    if _skip_if_unsupported("javascript"): return
    src = "function add(a, b) {\n  return a + b;\n}\n"
    nodes = _parse("javascript", src)
    assert nodes[0].name == "add"


def test_38_js_class_declaration():
    """JS: class declaration."""
    if _skip_if_unsupported("javascript"): return
    src = "class Animal {\n  constructor(name) {\n    this.name = name;\n  }\n}\n"
    nodes = _parse("javascript", src)
    assert len(nodes) == 1
    assert nodes[0].name == "Animal"
    assert nodes[0].node_type == "class_declaration"


def test_39_js_class_with_methods():
    """JS: class with multiple methods."""
    if _skip_if_unsupported("javascript"): return
    src = (
        "class Dog {\n"
        "  bark() {\n    return 'woof';\n  }\n"
        "  sit() {\n    return 'sitting';\n  }\n"
        "}\n"
    )
    nodes = _parse("javascript", src)
    assert nodes[0].name == "Dog"
    child_names = [c.name for c in nodes[0].children]
    assert "bark" in child_names
    assert "sit" in child_names


def test_40_js_class_parent_name():
    """JS: methods should have parent_name set to class name."""
    if _skip_if_unsupported("javascript"): return
    src = "class Box {\n  open() {}\n  close() {}\n}\n"
    nodes = _parse("javascript", src)
    for child in nodes[0].children:
        assert child.parent_name == "Box"


def test_41_js_class_extends():
    """JS: class with extends."""
    if _skip_if_unsupported("javascript"): return
    src = "class Admin extends User {\n  promote() {}\n}\n"
    nodes = _parse("javascript", src)
    assert nodes[0].name == "Admin"


def test_42_js_multiple_functions():
    """JS: multiple function declarations."""
    if _skip_if_unsupported("javascript"): return
    src = (
        "function a() {}\n"
        "function b() {}\n"
        "function c() {}\n"
    )
    nodes = _parse("javascript", src)
    names = [n.name for n in nodes]
    assert "a" in names and "b" in names and "c" in names


def test_43_js_export_function():
    """JS: exported function."""
    if _skip_if_unsupported("javascript"): return
    src = "export function createApp() {\n  return {};\n}\n"
    nodes = _parse("javascript", src)
    assert len(nodes) >= 1


def test_44_js_export_class():
    """JS: exported class."""
    if _skip_if_unsupported("javascript"): return
    src = "export class Router {\n  navigate(path) {}\n}\n"
    nodes = _parse("javascript", src)
    assert len(nodes) >= 1


def test_45_js_export_default():
    """JS: export default function."""
    if _skip_if_unsupported("javascript"): return
    src = "export default function main() {\n  return 42;\n}\n"
    nodes = _parse("javascript", src)
    assert len(nodes) >= 1


def test_46_js_arrow_const():
    """JS: arrow function assigned to const."""
    if _skip_if_unsupported("javascript"): return
    src = "const double = (x) => x * 2;\n"
    nodes = _parse("javascript", src)
    # Arrow function should be detected
    assert len(nodes) >= 1


def test_47_js_arrow_multiline():
    """JS: multi-line arrow function."""
    if _skip_if_unsupported("javascript"): return
    src = (
        "const process = (data) => {\n"
        "  const result = transform(data);\n"
        "  return result;\n"
        "};\n"
    )
    nodes = _parse("javascript", src)
    assert len(nodes) >= 1


def test_48_js_function_and_class_mixed():
    """JS: mixed functions and classes."""
    if _skip_if_unsupported("javascript"): return
    src = (
        "function helper() {}\n\n"
        "class Service {\n  run() {}\n}\n\n"
        "function another() {}\n"
    )
    nodes = _parse("javascript", src)
    assert len(nodes) >= 3


def test_49_js_class_static_method():
    """JS: class with static method."""
    if _skip_if_unsupported("javascript"): return
    src = "class Math2 {\n  static square(x) {\n    return x * x;\n  }\n}\n"
    nodes = _parse("javascript", src)
    assert nodes[0].name == "Math2"
    assert len(nodes[0].children) >= 1


def test_50_js_class_getter_setter():
    """JS: class with getter and setter."""
    if _skip_if_unsupported("javascript"): return
    src = (
        "class Person {\n"
        "  get fullName() {\n    return this.first + ' ' + this.last;\n  }\n"
        "  set fullName(val) {\n    this.first = val;\n  }\n"
        "}\n"
    )
    nodes = _parse("javascript", src)
    assert nodes[0].name == "Person"
    assert len(nodes[0].children) >= 2


def test_51_js_async_function():
    """JS: async function declaration."""
    if _skip_if_unsupported("javascript"): return
    src = "async function loadData() {\n  const res = await fetch('/api');\n  return res;\n}\n"
    nodes = _parse("javascript", src)
    assert len(nodes) >= 1


def test_52_js_class_async_method():
    """JS: async method inside class."""
    if _skip_if_unsupported("javascript"): return
    src = (
        "class API {\n"
        "  async get(url) {\n    return await fetch(url);\n  }\n"
        "}\n"
    )
    nodes = _parse("javascript", src)
    assert len(nodes[0].children) >= 1


def test_53_js_class_constructor():
    """JS: class constructor method."""
    if _skip_if_unsupported("javascript"): return
    src = "class Foo {\n  constructor(x) {\n    this.x = x;\n  }\n}\n"
    nodes = _parse("javascript", src)
    child_names = [c.name for c in nodes[0].children]
    assert "constructor" in child_names


def test_54_js_empty_function():
    """JS: empty function body."""
    if _skip_if_unsupported("javascript"): return
    src = "function noop() {}\n"
    nodes = _parse("javascript", src)
    assert len(nodes) == 1
    assert nodes[0].name == "noop"


def test_55_js_function_line_numbers():
    """JS: correct line numbers for JS function."""
    if _skip_if_unsupported("javascript"): return
    src = "// comment\n\nfunction test() {\n  return 1;\n}\n"
    nodes = _parse("javascript", src)
    fn = [n for n in nodes if n.node_type == "function_declaration"][0]
    assert fn.start_line == 3
    assert fn.end_line == 5


def test_56_js_language_field():
    """JS: ASTNode.language should be 'javascript'."""
    if _skip_if_unsupported("javascript"): return
    nodes = _parse("javascript", "function x() {}\n")
    assert nodes[0].language == "javascript"


def test_57_js_source_code_content():
    """JS: source_code should contain the function text."""
    if _skip_if_unsupported("javascript"): return
    src = "function greet(name) {\n  return 'Hello ' + name;\n}\n"
    nodes = _parse("javascript", src)
    assert "function greet" in nodes[0].source_code
    assert "return" in nodes[0].source_code


def test_58_js_multiple_classes():
    """JS: multiple classes in one file."""
    if _skip_if_unsupported("javascript"): return
    src = (
        "class A {\n  m() {}\n}\n\n"
        "class B {\n  n() {}\n}\n\n"
        "class C {\n  o() {}\n}\n"
    )
    nodes = _parse("javascript", src)
    class_nodes = [n for n in nodes if n.node_type == "class_declaration"]
    assert len(class_nodes) == 3


def test_59_js_class_no_methods():
    """JS: empty class body."""
    if _skip_if_unsupported("javascript"): return
    src = "class Empty {}\n"
    nodes = _parse("javascript", src)
    assert nodes[0].name == "Empty"
    assert nodes[0].children == []


def test_60_js_top_level_no_parent():
    """JS: top-level declarations should have parent_name=None."""
    if _skip_if_unsupported("javascript"): return
    nodes = _parse("javascript", "function standalone() {}\n")
    assert nodes[0].parent_name is None


# ============================================================
#  JAVA TESTS (61–80)
# ============================================================

def test_61_java_class():
    """Java: simple class declaration."""
    if _skip_if_unsupported("java"): return
    src = "public class HelloWorld {\n}\n"
    nodes = _parse("java", src)
    assert len(nodes) == 1
    assert nodes[0].name == "HelloWorld"
    assert "class" in nodes[0].node_type


def test_62_java_class_with_method():
    """Java: class with one method."""
    if _skip_if_unsupported("java"): return
    src = (
        "public class Greeter {\n"
        "    public void greet() {\n"
        "        System.out.println(\"Hello\");\n"
        "    }\n"
        "}\n"
    )
    nodes = _parse("java", src)
    assert nodes[0].name == "Greeter"
    child_names = [c.name for c in nodes[0].children]
    assert "greet" in child_names


def test_63_java_multiple_methods():
    """Java: class with multiple methods."""
    if _skip_if_unsupported("java"): return
    src = (
        "public class Calc {\n"
        "    public int add(int a, int b) { return a+b; }\n"
        "    public int sub(int a, int b) { return a-b; }\n"
        "    public int mul(int a, int b) { return a*b; }\n"
        "}\n"
    )
    nodes = _parse("java", src)
    assert len(nodes[0].children) == 3


def test_64_java_constructor():
    """Java: class with constructor."""
    if _skip_if_unsupported("java"): return
    src = (
        "public class User {\n"
        "    private String name;\n"
        "    public User(String name) {\n"
        "        this.name = name;\n"
        "    }\n"
        "}\n"
    )
    nodes = _parse("java", src)
    child_types = [c.node_type for c in nodes[0].children]
    assert "constructor_declaration" in child_types


def test_65_java_interface():
    """Java: interface declaration."""
    if _skip_if_unsupported("java"): return
    src = (
        "public interface Drawable {\n"
        "    void draw();\n"
        "    void resize(int w, int h);\n"
        "}\n"
    )
    nodes = _parse("java", src)
    assert len(nodes) == 1
    assert "interface" in nodes[0].node_type


def test_66_java_static_method():
    """Java: static method."""
    if _skip_if_unsupported("java"): return
    src = (
        "public class Utils {\n"
        "    public static String format(String s) {\n"
        "        return s.trim();\n"
        "    }\n"
        "}\n"
    )
    nodes = _parse("java", src)
    assert len(nodes[0].children) >= 1


def test_67_java_abstract_class():
    """Java: abstract class with abstract method."""
    if _skip_if_unsupported("java"): return
    src = (
        "public abstract class Shape {\n"
        "    public abstract double area();\n"
        "    public void describe() {\n"
        "        System.out.println(\"shape\");\n"
        "    }\n"
        "}\n"
    )
    nodes = _parse("java", src)
    assert nodes[0].name == "Shape"


def test_68_java_multiple_classes():
    """Java: multiple classes in one file."""
    if _skip_if_unsupported("java"): return
    src = (
        "class A {\n    void m() {}\n}\n\n"
        "class B {\n    void n() {}\n}\n"
    )
    nodes = _parse("java", src)
    class_nodes = [n for n in nodes if "class" in n.node_type]
    assert len(class_nodes) == 2


def test_69_java_method_parent_name():
    """Java: methods should have parent_name set."""
    if _skip_if_unsupported("java"): return
    src = (
        "public class Engine {\n"
        "    public void start() {}\n"
        "    public void stop() {}\n"
        "}\n"
    )
    nodes = _parse("java", src)
    for child in nodes[0].children:
        assert child.parent_name == "Engine"


def test_70_java_class_extends():
    """Java: class with extends."""
    if _skip_if_unsupported("java"): return
    src = (
        "public class Car extends Vehicle {\n"
        "    public void drive() {}\n"
        "}\n"
    )
    nodes = _parse("java", src)
    assert nodes[0].name == "Car"


def test_71_java_class_implements():
    """Java: class implementing interface."""
    if _skip_if_unsupported("java"): return
    src = (
        "public class Circle implements Drawable {\n"
        "    public void draw() {}\n"
        "}\n"
    )
    nodes = _parse("java", src)
    assert nodes[0].name == "Circle"


def test_72_java_void_method():
    """Java: void method."""
    if _skip_if_unsupported("java"): return
    src = (
        "public class Logger {\n"
        "    public void log(String msg) {\n"
        "        System.out.println(msg);\n"
        "    }\n"
        "}\n"
    )
    nodes = _parse("java", src)
    assert nodes[0].children[0].name == "log"


def test_73_java_return_type_method():
    """Java: method with return type."""
    if _skip_if_unsupported("java"): return
    src = (
        "public class Converter {\n"
        "    public int toInt(String s) {\n"
        "        return Integer.parseInt(s);\n"
        "    }\n"
        "}\n"
    )
    nodes = _parse("java", src)
    assert nodes[0].children[0].name == "toInt"


def test_74_java_empty_class():
    """Java: empty class."""
    if _skip_if_unsupported("java"): return
    src = "public class Marker {}\n"
    nodes = _parse("java", src)
    assert nodes[0].name == "Marker"
    assert nodes[0].children == []


def test_75_java_language_field():
    """Java: ASTNode.language should be 'java'."""
    if _skip_if_unsupported("java"): return
    src = "public class X {}\n"
    nodes = _parse("java", src)
    assert nodes[0].language == "java"


def test_76_java_method_line_numbers():
    """Java: correct line numbers."""
    if _skip_if_unsupported("java"): return
    src = (
        "// comment\n"
        "public class T {\n"
        "    public void m() {\n"
        "        int x = 1;\n"
        "    }\n"
        "}\n"
    )
    nodes = _parse("java", src)
    assert nodes[0].start_line == 2


def test_77_java_source_code_content():
    """Java: source_code should contain the actual code."""
    if _skip_if_unsupported("java"): return
    src = "public class Demo {\n    public void run() { return; }\n}\n"
    nodes = _parse("java", src)
    assert "Demo" in nodes[0].source_code
    assert "run" in nodes[0].source_code


def test_78_java_constructor_and_methods():
    """Java: mix of constructor and methods."""
    if _skip_if_unsupported("java"): return
    src = (
        "public class Product {\n"
        "    public Product(String name) { this.name = name; }\n"
        "    public String getName() { return this.name; }\n"
        "    public void setName(String n) { this.name = n; }\n"
        "}\n"
    )
    nodes = _parse("java", src)
    child_names = [c.name for c in nodes[0].children]
    assert "Product" in child_names  # constructor
    assert "getName" in child_names
    assert "setName" in child_names


def test_79_java_overloaded_methods():
    """Java: overloaded methods (same name, different params)."""
    if _skip_if_unsupported("java"): return
    src = (
        "public class Printer {\n"
        "    public void print(String s) {}\n"
        "    public void print(int n) {}\n"
        "    public void print(double d) {}\n"
        "}\n"
    )
    nodes = _parse("java", src)
    names = [c.name for c in nodes[0].children]
    assert names.count("print") == 3


def test_80_java_top_level_no_parent():
    """Java: top-level class should have parent_name=None."""
    if _skip_if_unsupported("java"): return
    src = "public class Root {}\n"
    nodes = _parse("java", src)
    assert nodes[0].parent_name is None


# ============================================================
#  EDGE CASES & ERROR HANDLING TESTS (81–100)
# ============================================================

def test_81_empty_source():
    """Edge: empty source code should return no nodes."""
    if _skip_if_unsupported("python"): return
    nodes = _parse("python", "")
    assert nodes == []


def test_82_whitespace_only():
    """Edge: whitespace-only source should return no nodes."""
    if _skip_if_unsupported("python"): return
    nodes = _parse("python", "   \n\n   \n")
    assert nodes == []


def test_83_comments_only():
    """Edge: file with only comments should return no nodes."""
    if _skip_if_unsupported("python"): return
    src = "# This is a comment\n# Another comment\n"
    nodes = _parse("python", src)
    assert nodes == []


def test_84_imports_only():
    """Edge: file with only imports should return no nodes."""
    if _skip_if_unsupported("python"): return
    src = "import os\nimport sys\nfrom pathlib import Path\n"
    nodes = _parse("python", src)
    assert nodes == []


def test_85_single_line_function():
    """Edge: single-line function."""
    if _skip_if_unsupported("python"): return
    src = "def one_liner(): return 42\n"
    nodes = _parse("python", src)
    assert len(nodes) == 1
    assert nodes[0].start_line == nodes[0].end_line


def test_86_unicode_in_source():
    """Edge: source code with unicode characters."""
    if _skip_if_unsupported("python"): return
    src = "def greet():\n    return '你好世界 🌍'\n"
    nodes = _parse("python", src)
    assert len(nodes) == 1
    assert nodes[0].name == "greet"


def test_87_unicode_function_name():
    """Edge: Python allows unicode identifiers."""
    if _skip_if_unsupported("python"): return
    src = "def xử_lý():\n    pass\n"
    nodes = _parse("python", src)
    assert len(nodes) == 1
    assert nodes[0].name == "xử_lý"


def test_88_very_long_function_name():
    """Edge: extremely long function name."""
    if _skip_if_unsupported("python"): return
    name = "a" * 200
    src = f"def {name}():\n    pass\n"
    nodes = _parse("python", src)
    assert nodes[0].name == name


def test_89_many_functions():
    """Edge: file with 50 functions."""
    if _skip_if_unsupported("python"): return
    src = "\n".join([f"def func_{i}():\n    pass\n" for i in range(50)])
    nodes = _parse("python", src)
    assert len(nodes) == 50


def test_90_deeply_nested_classes():
    """Edge: class inside class (Python allows this)."""
    if _skip_if_unsupported("python"): return
    src = (
        "class Outer:\n"
        "    class Inner:\n"
        "        def method(self):\n"
        "            pass\n"
    )
    nodes = _parse("python", src)
    assert nodes[0].name == "Outer"
    # Inner should be a child of Outer
    inner = [c for c in nodes[0].children if c.name == "Inner"]
    assert len(inner) == 1


def test_91_is_supported_python():
    """Parser: is_supported returns True for installed languages."""
    if _skip_if_unsupported("python"): return
    assert _parser.is_supported("python") == True


def test_92_is_supported_unsupported():
    """Parser: is_supported returns False for unknown languages."""
    assert _parser.is_supported("ruby") == False
    assert _parser.is_supported("rust") == False
    assert _parser.is_supported("go") == False


def test_93_is_supported_empty_string():
    """Parser: is_supported returns False for empty string."""
    assert _parser.is_supported("") == False


def test_94_parser_reuse():
    """Parser: same parser instance can parse multiple files."""
    if _skip_if_unsupported("python"): return
    nodes1 = _parse("python", "def a():\n    pass\n")
    nodes2 = _parse("python", "def b():\n    pass\n")
    assert nodes1[0].name == "a"
    assert nodes2[0].name == "b"


def test_95_parser_cross_language():
    """Parser: same parser can switch between Python and JS."""
    py_supported = _parser.is_supported("python")
    js_supported = _parser.is_supported("javascript")
    if not (py_supported and js_supported):
        print("⚠️  SKIP: need both python and javascript parsers")
        return
    py_nodes = _parse("python", "def py_func():\n    pass\n")
    js_nodes = _parse("javascript", "function jsFunc() {}\n")
    assert py_nodes[0].name == "py_func"
    assert js_nodes[0].name == "jsFunc"


def test_96_node_children_default_empty():
    """ASTNode: children should default to empty list."""
    node = ASTNode(
        node_type="function_definition",
        name="test",
        start_line=1,
        end_line=2,
        source_code="def test():\n    pass",
        language="python",
    )
    assert node.children == []
    assert node.parent_name is None


def test_97_node_with_children():
    """ASTNode: manually created node with children."""
    child = ASTNode(
        node_type="function_definition",
        name="method",
        start_line=2,
        end_line=3,
        source_code="def method(): pass",
        language="python",
        parent_name="MyClass",
    )
    parent = ASTNode(
        node_type="class_definition",
        name="MyClass",
        start_line=1,
        end_line=4,
        source_code="class MyClass:\n  def method(): pass",
        language="python",
        children=[child],
    )
    assert len(parent.children) == 1
    assert parent.children[0].parent_name == "MyClass"


def test_98_multiline_string_in_function():
    """Edge: function containing multiline string."""
    if _skip_if_unsupported("python"): return
    src = (
        "def get_sql():\n"
        '    return """\n'
        "    SELECT *\n"
        "    FROM users\n"
        "    WHERE active = 1\n"
        '    """\n'
    )
    nodes = _parse("python", src)
    assert len(nodes) == 1
    assert nodes[0].name == "get_sql"


def test_99_function_with_try_except():
    """Edge: function with complex control flow."""
    if _skip_if_unsupported("python"): return
    src = (
        "def safe_divide(a, b):\n"
        "    try:\n"
        "        return a / b\n"
        "    except ZeroDivisionError:\n"
        "        return None\n"
        "    finally:\n"
        "        print('done')\n"
    )
    nodes = _parse("python", src)
    assert len(nodes) == 1
    assert nodes[0].name == "safe_divide"
    assert nodes[0].end_line == 7


def test_100_get_node_name_anonymous():
    """Edge: _get_node_name fallback to <anonymous>."""
    if _skip_if_unsupported("python"): return
    # Parse a source with a function — then test the internal method
    src = "x = 1\n"
    nodes = _parse("python", src)
    # No named nodes, so we just verify no crash
    assert isinstance(nodes, list)


# ============================================================
#  RUNNER
# ============================================================

if __name__ == "__main__":
    import inspect
    import re

    print("=" * 60, flush=True)
    print("  Running Comprehensive AST Parser Tests (100 tests)", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    # Collect all test functions sorted by numeric suffix
    def _sort_key(name):
        m = re.search(r"(\d+)", name)
        return int(m.group(1)) if m else 0

    test_functions = [
        obj for name, obj in globals().items()
        if name.startswith("test_") and inspect.isfunction(obj)
    ]
    test_functions.sort(key=lambda fn: _sort_key(fn.__name__))

    passed = 0
    failed = 0
    skipped = 0
    total = len(test_functions)

    for i, test_fn in enumerate(test_functions, 1):
        desc = (test_fn.__doc__ or "").strip()
        try:
            test_fn()
            print(f"  [PASS] [{i:3d}/{total}] {test_fn.__name__}: {desc}", flush=True)
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] [{i:3d}/{total}] {test_fn.__name__}: {desc}", flush=True)
            print(f"           AssertionError: {e}", flush=True)
            failed += 1
        except Exception as e:
            print(f"  [FAIL] [{i:3d}/{total}] {test_fn.__name__}: {desc}", flush=True)
            print(f"           {type(e).__name__}: {e}", flush=True)
            failed += 1

    print(flush=True)
    print("=" * 60, flush=True)
    print(f"  Results: {passed} passed, {failed} failed, {total} total", flush=True)
    print("=" * 60, flush=True)

    if failed > 0:
        sys.exit(1)

