"""
AST Parser — parse source code using tree-sitter.
Extracts functions, classes, and methods as structured nodes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Tree-sitter language grammars
_LANGUAGE_MODULES = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "java": "tree_sitter_java",
    "typescript": "tree_sitter_typescript",
}

# AST node types to extract per language
# extract including: component structure of python, javascript, java
_EXTRACT_TYPES = {
    "python": [
        "function_definition",
        "class_definition",
    ],
    "javascript": [
        "function_declaration",
        "class_declaration",
        "arrow_function",
        "method_definition",
        "export_statement",
    ],
    "java": [
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "constructor_declaration",
    ],
    "typescript":[
        "function_declaration",
        "class_declaration",
        "arrow_function",
        "method_definition",
        "export_statement",
    ]
}


@dataclass
class ASTNode:
    """Represents an extracted code structure node."""

    node_type: str          # e.g., "function_definition", "class_definition"
    name: str               # e.g., "authenticate_user"
    start_line: int         # 1-indexed
    end_line: int           # 1-indexed
    source_code: str        # Raw source text of this node
    language: str
    children: List["ASTNode"] = field(default_factory=list)
    parent_name: Optional[str] = None


class ASTParser:
    def __init__(self):
        self._parsers = {}
        self._init_parsers()

    def _init_parsers(self) -> None:
        try:
            import tree_sitter as ts
        except ImportError:
            logger.error("tree-sitter not installed")
            return

        for language, module_name in _LANGUAGE_MODULES.items():
            try:
                lang_module = __import__(module_name)
                lang = ts.Language(lang_module.language())
                parser = ts.Parser(lang)
                self._parsers[language] = {
                    "parser": parser,
                    "language": lang,
                }
            except Exception as e:
                logger.warning(f"Failed to init parser for {language}: {e}")

    def parse_file(
        self,
        file_path: Path,
        language: str,
        source_code: str = None,
    ) -> List[ASTNode]:
        parser = self._parsers.get(language)
        tree = parser["parser"].parse(source_code.encode("utf-8"))
        root = tree.root_node
        node = self._extract_nodes(root, source_code, language, extract_types=
        _EXTRACT_TYPES.get(language,[]))

        return node

    def _extract_nodes(
        self,
        node,
        source_code: str,
        language: str,
        extract_types: List[str],
        parent_name: Optional[str] = None,
    ) -> List[ASTNode]:
        results = []
        source_lines = source_code.split("\n")

        for child in node.children:
            if child.type in extract_types:
                name = self._get_node_name(child, language)
                start_line = child.start_point[0] + 1  
                end_line = child.end_point[0] + 1
                chunk_source = "\n".join(
                    source_lines[start_line - 1 : end_line]
                )

                ast_node = ASTNode(
                    node_type=child.type,
                    name=name,
                    start_line=start_line,
                    end_line=end_line,
                    source_code=chunk_source,
                    language=language,
                    parent_name=parent_name,
                )

                # Recurse into class body for methods
                if "class" in child.type:
                    ast_node.children = self._extract_nodes(
                        child, source_code, language, extract_types,
                        parent_name=name,
                    )

                results.append(ast_node)
            else:
                results.extend(
                    self._extract_nodes(
                        child, source_code, language, extract_types,
                        parent_name=parent_name,
                    )
                )

        return results

    def _get_node_name(self, node, language: str) -> str:
        for child in node.children:
            if child.type in ("identifier", "name", "property_identifier"):
                return child.text.decode("utf-8")
        if node.type == "arrow_function" and node.parent:
            for sibling in node.parent.children:
                if sibling.type == "identifier":
                    return sibling.text.decode("utf-8")
        return "<anonymous>"

    def is_supported(self, language: str) -> bool:
        return language in self._parsers
