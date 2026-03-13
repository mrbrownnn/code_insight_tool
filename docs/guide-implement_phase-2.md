# Hướng Dẫn Triển Khai Phase 2: RAG & Chat

## Tổng Quan

Phase 2 triển khai hệ thống **Retrieval-Augmented Generation (RAG)** để cho phép người dùng đặt câu hỏi tự nhiên về codebase đã được index, nhận câu trả lời có ngữ cảnh với tham chiếu code chính xác.

## Chiến Lược Triển Khai

### Phương Án Sử Dụng LangChain

**Quyết định**: Sử dụng LangChain làm framework chính cho Phase 2.

#### Lý Do
- **Dễ triển khai**: Giảm 60-80% development time với components có sẵn
- **Dễ mở rộng**: Framework mature cho future features (Phase 3 intelligence, multi-modal)
- **Ecosystem**: Rich integrations và community support
- **Production-ready**: Built-in error handling, streaming, memory management

#### Trade-offs
- Dependency overhead (~200MB additional packages)
- Learning curve cho LangChain APIs
- Potential performance impact (monitoring required)

## Cấu Trúc File & Components

### Files Cần Thêm

```
core/
├── retrieval/
│   ├── __init__.py
│   ├── hybrid_retriever.py      # BM25 + Vector hybrid search
│   └── context_expander.py      # AST-based context expansion
├── generation/
│   ├── __init__.py
│   ├── llm_factory.py           # LangChain LLM provider factory
│   ├── prompt_templates.py      # Code-aware prompt templates
│   ├── rag_chain.py             # ConversationalRetrievalChain wrapper
│   └── code_mapper.py           # Response → code reference mapping
└── intelligence/                 # Phase 3 placeholder
    └── __init__.py

ui/
├── pages/
│   ├── __init__.py
│   └── chat.py                  # Chat interface với streaming
└── components/
    ├── __init__.py
    ├── streaming_text.py        # Real-time text streaming
    ├── code_reference.py        # Clickable code references
    └── conversation_history.py  # Chat history management
```

### Files Cần Sửa Đổi

| File | Changes |
|------|---------|
| `config.py` | Thêm LangChain settings |
| `storage/metadata_store.py` | Thêm conversation tables |
| `storage/vector_store.py` | LangChain compatibility layer |
| `app.py` | Thêm Chat page routing |
| `requirements.txt` | Đã cập nhật với LangChain dependencies |

## Features Cần Thêm

### 1. Hybrid Search & Retrieval
- **BM25 Keyword Search**: Text-based retrieval cho exact matches
- **Vector Similarity**: Semantic search với UniXcoder embeddings
- **Score Combination**: Weighted fusion (vector 70% + BM25 30%)
- **Re-ranking**: Recency và relevance-based sorting

### 2. Context Expansion
- **AST-based Expansion**: Thêm parent classes, sibling functions
- **Scope Analysis**: Include related code context
- **Deduplication**: Remove redundant chunks
- **Window Management**: Configurable context size

### 3. LLM Provider Abstraction
- **Multi-Provider Support**: Gemini, Groq, Ollama fallback
- **Streaming**: Real-time response generation
- **Rate Limiting**: API call management
- **Error Handling**: Graceful provider switching

### 4. Conversation Memory
- **SQLite Backend**: Persistent conversation storage
- **5-turn Window**: Sliding window memory management
- **Context Preservation**: Maintain conversation state
- **Auto-cleanup**: Remove old conversations

### 5. Code Reference Mapping
- **Response Parsing**: Extract code mentions from LLM output
- **File:Line Mapping**: Convert to clickable references
- **Snippet Extraction**: Show relevant code sections
- **IDE Integration**: Prepare for VS Code deeplinks (Phase 4)

### 6. Chat UI
- **Streaming Interface**: Real-time message updates
- **Code Highlighting**: Syntax highlighting cho references
- **Conversation Management**: New chat, load history
- **Progress Indicators**: Retrieval/generation status

## Tính Năng Cần Viết

### Core Implementation Priority

#### 1. LLM Factory (Day 1-2)
- Provider initialization
- API key management
- Streaming setup

#### 2. Prompt Templates (Day 2-3)
- Code Q&A templates
- Explanation templates
- Summarization templates
- Few-shot examples

#### 3. Hybrid Retriever (Day 3-5)
- BM25 implementation
- Vector search integration
- Score fusion logic
- LangChain BaseRetriever extension

#### 4. Context Expander (Day 5-6)
- AST metadata processing
- Parent/sibling retrieval
- Context window management

#### 5. RAG Chain (Day 6-8)
- ConversationalRetrievalChain setup
- Memory integration
- Streaming callbacks
- Code reference extraction

#### 6. Database Schema (Day 1-2)
- Conversation tables
- Message storage
- Reference tracking

#### 7. Chat UI (Day 8-10)
- Streamlit chat interface
- Component integration
- Real-time updates

## Phân Tích Rủi Ro

### Technical Risks

#### 1. Performance Issues
**Risk**: High latency với large codebases
**Impact**: Poor user experience
**Mitigation**:
- Implement caching layers
- Optimize vector search queries
- Add pagination cho large results
- Monitor response times (< 5s target)

#### 2. API Rate Limits & Costs
**Risk**: LLM API costs exceed budget
**Impact**: Service disruption
**Mitigation**:
- Implement token counting
- Add rate limiting (requests/minute)
- Cost monitoring dashboard
- Fallback to local Ollama

#### 3. Memory Management
**Risk**: Conversation memory grows unbounded
**Impact**: Database bloat, slow queries
**Mitigation**:
- Enforce 5-turn limit
- Auto-cleanup old conversations
- Compress message content
- Database indexing

#### 4. Code Reference Accuracy
**Risk**: Incorrect file:line mappings
**Impact**: User confusion, broken links
**Mitigation**:
- Validate mappings against AST metadata
- Unit tests cho mapping logic
- Fallback to search-based location
- User feedback mechanism

### Integration Risks

#### 5. LangChain Version Conflicts
**Risk**: Breaking changes trong future versions
**Impact**: Maintenance overhead
**Mitigation**:
- Pin versions trong requirements.txt
- Abstract LangChain behind custom interfaces
- Regular dependency updates
- Fallback implementations

#### 6. Vector Store Compatibility
**Risk**: Qdrant version conflicts với LangChain
**Impact**: Retrieval failures
**Mitigation**:
- Test compatibility matrix
- Maintain custom wrapper layer
- Monitor Qdrant performance
- Backup retrieval strategies

### Operational Risks

#### 7. Data Privacy
**Risk**: Code snippets leaked to LLM providers
**Impact**: Security violations
**Mitigation**:
- Local LLM fallback (Ollama)
- Code anonymization
- User consent mechanisms
- Audit logging

#### 8. Scalability Limits
**Risk**: System fails với very large codebases
**Impact**: Service unavailability
**Mitigation**:
- Implement chunking limits
- Add horizontal scaling
- Database optimization
- Load testing

## Timeline & Milestones

### Week 1: Foundation (Days 1-5)
- ✅ Dependencies setup
- 🔄 LLM factory + prompts
- 🔄 Database schema
- 🔄 Hybrid retriever core

**Milestone**: Basic RAG pipeline working locally

### Week 2: Core Features (Days 6-10)
- 🔄 Context expansion
- 🔄 RAG chain implementation
- 🔄 Memory management
- 🔄 Code reference mapping

**Milestone**: End-to-end chat with code references

### Week 3: UI & Polish (Days 11-14)
- 🔄 Chat UI implementation
- 🔄 Streaming integration
- 🔄 Error handling
- 🔄 Performance optimization

**Milestone**: Production-ready chat interface

### Week 4: Testing & Deployment (Days 15-16)
- 🔄 Integration testing
- 🔄 Performance benchmarking
- 🔄 Documentation
- 🔄 Deployment preparation

**Milestone**: Phase 2 complete, ready for Phase 3

## Success Criteria

| Aspect | Target |
|--------|--------|
| **Functional** | Users can ask code questions and get accurate answers |
| **Performance** | < 3s response time, < 5s with streaming |
| **Accuracy** | > 80% code reference accuracy |
| **Usability** | Intuitive chat interface với code highlighting |
| **Reliability** | 99% uptime, graceful error handling |
| **Maintainability** | Clean code, comprehensive tests |

## Conclusion

Việc sử dụng LangChain cho Phase 2 cân bằng giữa tốc độ triển khai và khả năng mở rộng. Chiến lược này cho phép deliver features nhanh chóng trong khi xây dựng foundation vững chắc cho các phase sau.

### Recommended Next Steps
1. Begin với LLM factory implementation
2. Setup database schema changes
3. Implement hybrid retriever
4. Build RAG chain iteratively

### Risk Monitoring
Track performance metrics và user feedback để validate architectural decisions.