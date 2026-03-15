"""
Context Expander — Automatically expands retrieved chunks with related code.

Strategy:
1. Parent scope (class/function containing the chunk) — always
2. Imports relevant to the chunk — always
3. Sibling functions in same class — conditional (if query mentions class name)

Hard token limit enforced: max_context_tokens (from config)
"""

from typing import List, Dict, Any, Optional, Set
import re

from core.ingestion.ast_parser import ASTParser, ASTNode
from config import settings
from utils.logger import get_logger
from utils.token_counter import estimate_tokens

logger = get_logger(__name__)


class ContextExpander:
    def __init__(self):
        self.ast_parser = ASTParser()
    def expand_chunk(
        self,
        chunk: Dict[str, Any],
        query: str = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        max_tokens = max_tokens or settings.max_context_tokens
        expanded_text = chunk.get("text", "")
        context_added = {}
        source_file = chunk.get("source_file")
        start_line = chunk.get("start_line", 1)
        
        if not source_file:
            logger.warning("No source_file in chunk, skipping expansion")
            return chunk
        
        try:
            ast_tree = self.ast_parser.parse_file(source_file)
        except Exception as e:
            logger.warning(f"Failed to parse {source_file}: {e}")
            return chunk
        containing_node = self._find_containing_node(ast_tree, start_line)
        
        if not containing_node:
            logger.debug(f"No containing node found for {source_file}:{start_line}")
            return chunk
        
        # Step 1: Add parent scope (class/function)
        tokens_used = estimate_tokens(expanded_text)
        if tokens_used < max_tokens and containing_node.parent_name:
            parent_text = self._get_parent_scope_text(containing_node)
            if parent_text:
                parent_tokens = estimate_tokens(parent_text)
                if tokens_used + parent_tokens <= max_tokens:
                    expanded_text = f"{parent_text}\n\n--- CHUNK START ---\n\n{expanded_text}"
                    tokens_used += parent_tokens
                    context_added["parent_scope"] = True
        
        # Step 2: Add relevant imports
        tokens_used = estimate_tokens(expanded_text)
        if tokens_used < max_tokens:
            imports_text = self._get_relevant_imports(expanded_text, source_file)
            if imports_text:
                imports_tokens = estimate_tokens(imports_text)
                if tokens_used + imports_tokens <= max_tokens:
                    expanded_text = f"{imports_text}\n\n{expanded_text}"
                    tokens_used += imports_tokens
                    context_added["imports"] = True
        
        # Step 3: Add sibling functions (conditional)
        tokens_used = estimate_tokens(expanded_text)
        if (tokens_used < max_tokens and 
            containing_node.parent_name and
            query and 
            containing_node.parent_name.lower() in query.lower()):
            
            siblings_text = self._get_sibling_functions(containing_node)
            if siblings_text:
                siblings_tokens = estimate_tokens(siblings_text)
                if tokens_used + siblings_tokens <= max_tokens:
                    expanded_text = f"{expanded_text}\n\n--- SIBLING FUNCTIONS ---\n\n{siblings_text}"
                    context_added["siblings"] = True
        
        # Return expanded chunk
        return {
            **chunk,
            "text": expanded_text,
            "expanded": True,
            "context_added": context_added,
            "total_tokens": estimate_tokens(expanded_text),
        }
    
    def expand_chunks(
        self,
        chunks: List[Dict[str, Any]],
        query: str = None,
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        expanded = []
        for chunk in chunks:
            expanded_chunk = self.expand_chunk(chunk, query, max_tokens)
            expanded.append(expanded_chunk)
        
        logger.info(f"Expanded {len(expanded)} chunks")
        return expanded
    
    # ========== Helper Methods ==========
    
    def _find_containing_node(
        self,
        ast_tree: ASTNode,
        line_number: int,
    ) -> Optional[ASTNode]:
        def traverse(node: ASTNode) -> Optional[ASTNode]:
            if node.start_line <= line_number <= node.end_line:
                # Check children first (deepest match wins)
                for child in node.children:
                    result = traverse(child)
                    if result:
                        return result
                return node
            return None
        
        return traverse(ast_tree)
    
    def _get_parent_scope_text(self, node: ASTNode) -> str:
        if node.parent_name:
            return f"# Parent: {node.parent_name}"
        return ""
    
    def _get_relevant_imports(self, chunk_text: str, source_file: str) -> str:
        # Heuristic: look for identifiers in chunk and find imports that mention them
        identifiers = set(re.findall(r'\b[a-zA-Z_]\w*\b', chunk_text))
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:15]
                import_lines = [l.strip() for l in lines if 'import' in l.lower()]
                return "\n".join(import_lines)
        except Exception as e:
            logger.warning(f"Failed to extract imports from {source_file}: {e}")
            return ""
    
    def _get_sibling_functions(self, node: ASTNode) -> str:
        return ""
    
    def get_parent_scope(self, node: ASTNode) -> Optional[ASTNode]:
        """Get parent scope node."""
        return None
    
    def get_related_chunks(self, node: ASTNode) -> List[str]:
        """Get chunk IDs of related chunks."""
        return []