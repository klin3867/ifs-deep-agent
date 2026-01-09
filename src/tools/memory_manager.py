"""
Memory Manager for DeepAgent's Brain-Inspired Memory Architecture.

This module implements a three-component memory system:
1. Episodic Memory: High-level log of key events, decisions, and sub-task completions
2. Working Memory: Current sub-goal and near-term plans (session-only)
3. Tool Memory: Consolidated tool-related interactions for learning and strategy refinement

Memory is persisted to disk in JSON format, following the existing cache pattern in ToolManager.
"""

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import hashlib
import yaml


# =============================================================================
# SCHEMA EXTRACTION HELPER
# =============================================================================

def extract_param_rules_from_schema(schema: Dict[str, Any]) -> str:
    """
    Extract parameter documentation from MCP tool schema.

    This enables runtime merge of MCP schema info with YAML domain knowledge,
    ensuring parameter names are always correct (from authoritative source)
    and preventing drift between YAML and MCP server.

    Args:
        schema: Tool schema from MCPToolCaller.get_tool_schema()
                Format: {name, description, parameters: {properties, required}}

    Returns:
        Formatted string of parameter rules, e.g.:
        "Parameters: from_warehouse (string, required), to_warehouse (string, required)"
    """
    params = schema.get('parameters', {})
    properties = params.get('properties', {})
    required = set(params.get('required', []))

    if not properties:
        return ""

    param_descriptions = []
    for param_name, param_info in properties.items():
        param_type = param_info.get('type', 'any')
        is_required = param_name in required
        default = param_info.get('default')

        # Build description
        parts = [f"{param_name} ({param_type}"]
        if is_required:
            parts.append(", required")
        elif default is not None:
            parts.append(f", default: '{default}'")
        else:
            parts.append(", optional")
        parts.append(")")

        param_descriptions.append("".join(parts))

    return f"Parameters: {', '.join(param_descriptions)}"


# =============================================================================
# SEMANTIC KNOWLEDGE RETRIEVER
# =============================================================================

class KnowledgeRetriever:
    """
    Semantic retriever for knowledge base (procedural rules + semantic facts).
    Uses SentenceTransformer for embedding-based similarity search.
    """

    def __init__(self, knowledge_base: Dict, model_path: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer, util

        self.embedder = SentenceTransformer(model_path)
        self._util = util
        self.corpus = []
        self.corpus_to_item = {}

        # Index procedural rules
        for tool_name, tool_data in knowledge_base.get('procedural', {}).items():
            if not isinstance(tool_data, dict):
                continue
            for rule in tool_data.get('rules', []):
                # Create searchable text with tool context
                text = f"{tool_name}: {rule}"
                self.corpus.append(text)
                self.corpus_to_item[text] = ('rule', tool_name, rule)

        # Index semantic facts
        for category, cat_data in knowledge_base.get('semantic', {}).items():
            if not isinstance(cat_data, dict):
                continue
            for fact in cat_data.get('facts', []):
                text = f"{category}: {fact}"
                self.corpus.append(text)
                self.corpus_to_item[text] = ('fact', category, fact)

        # Index error corrections
        for error in knowledge_base.get('common_errors', []):
            if not isinstance(error, dict):
                continue
            correction = error.get('correction', '')
            pattern = error.get('pattern', '')
            if correction:
                text = f"error pattern '{pattern}': {correction}"
                self.corpus.append(text)
                self.corpus_to_item[text] = ('error', pattern, correction)

        # Pre-compute embeddings
        if self.corpus:
            self.corpus_embeddings = self.embedder.encode(
                self.corpus,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            print(f"KnowledgeRetriever: Indexed {len(self.corpus)} knowledge items")
        else:
            self.corpus_embeddings = None
            print("KnowledgeRetriever: No knowledge items to index")

    def retrieve(self, query: str, top_k: int = 10, min_score: float = 0.3) -> Dict[str, List[str]]:
        """
        Retrieve relevant knowledge items using semantic similarity.

        Args:
            query: Natural language query
            top_k: Maximum items to return
            min_score: Minimum similarity score threshold (0-1)

        Returns:
            Dict with 'procedural_rules', 'semantic_facts', 'error_corrections' keys
        """
        if not self.corpus or self.corpus_embeddings is None:
            return {'procedural_rules': [], 'semantic_facts': [], 'error_corrections': []}

        query_embedding = self.embedder.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        hits = self._util.semantic_search(
            query_embedding,
            self.corpus_embeddings,
            top_k=top_k,
            score_function=self._util.cos_sim
        )

        rules, facts, errors = [], [], []
        for hit in hits[0]:
            if hit['score'] < min_score:
                continue

            corpus_text = self.corpus[hit['corpus_id']]
            item_type, source, content = self.corpus_to_item[corpus_text]

            if item_type == 'rule':
                rules.append(f"[{source}] {content}")
            elif item_type == 'fact':
                facts.append(content)
            elif item_type == 'error':
                errors.append(content)

        return {
            'procedural_rules': rules[:5],
            'semantic_facts': facts[:5],
            'error_corrections': errors[:3],
        }


class MemoryManager:
    """
    Central manager for brain-inspired memory operations.

    Handles:
    - Memory storage (JSON file-based)
    - Memory retrieval with keyword matching
    - Memory consolidation across tasks
    - Memory pruning based on limits
    """

    def __init__(
        self,
        memory_dir: str = "./cache/memory",
        max_episodic_memories: int = 100,
        max_tool_memories_per_tool: int = 20,
        memory_retrieval_top_k: int = 5,
    ):
        self.memory_dir = memory_dir
        self.max_episodic_memories = max_episodic_memories
        self.max_tool_memories_per_tool = max_tool_memories_per_tool
        self.memory_retrieval_top_k = memory_retrieval_top_k

        # Memory stores
        self.episodic_memories: List[Dict] = []  # List of episode memories from past tasks
        self.tool_memories: Dict[str, List[Dict]] = {}  # Dict of tool experiences keyed by tool name
        self.working_memory: Optional[Dict] = None  # Current session working memory (not persisted)
        self.knowledge_base: Dict[str, Dict] = {}  # Procedural + semantic knowledge from YAML
        self.knowledge_retriever: Optional[KnowledgeRetriever] = None  # Semantic retriever for knowledge

        # Ensure memory directory exists
        os.makedirs(self.memory_dir, exist_ok=True)

        # Load existing memories from disk
        self.load_memories()

    def load_memories(self) -> None:
        """Load episodic and tool memories from disk if available."""
        # Load episodic memories
        episodic_path = os.path.join(self.memory_dir, 'episodic_memories.json')
        if os.path.exists(episodic_path):
            try:
                with open(episodic_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        self.episodic_memories = loaded
            except Exception:
                pass

        # Load tool memories
        tool_path = os.path.join(self.memory_dir, 'tool_memories.json')
        if os.path.exists(tool_path):
            try:
                with open(tool_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        self.tool_memories = loaded
            except Exception:
                pass

    def load_knowledge_base(self, config_path: str = "./config/ifs_knowledge.yaml", use_semantic: bool = True) -> int:
        """
        Load procedural rules and semantic facts from YAML config.

        This provides domain knowledge without requiring actual task execution.
        When use_semantic=True, builds a semantic retriever for embedding-based search.

        Args:
            config_path: Path to the knowledge YAML file
            use_semantic: If True, build KnowledgeRetriever for semantic search

        Returns:
            Number of knowledge entries loaded (procedural rules + semantic facts + error corrections)
        """
        if not os.path.exists(config_path):
            self.knowledge_base = {}
            self.knowledge_retriever = None
            return 0

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.knowledge_base = yaml.safe_load(f) or {}
        except Exception:
            self.knowledge_base = {}
            self.knowledge_retriever = None
            return 0

        # Count entries loaded
        count = 0

        # Count procedural rules
        procedural = self.knowledge_base.get('procedural', {})
        for tool_data in procedural.values():
            if isinstance(tool_data, dict):
                count += len(tool_data.get('rules', []))

        # Count semantic facts
        semantic = self.knowledge_base.get('semantic', {})
        for category_data in semantic.values():
            if isinstance(category_data, dict):
                count += len(category_data.get('facts', []))

        # Count error corrections
        count += len(self.knowledge_base.get('common_errors', []))

        # Build semantic retriever if enabled and we have content
        if use_semantic and count > 0:
            try:
                self.knowledge_retriever = KnowledgeRetriever(self.knowledge_base)
            except Exception as e:
                print(f"  ⚠️ Failed to build KnowledgeRetriever: {e}")
                self.knowledge_retriever = None
        else:
            self.knowledge_retriever = None

        return count

    def retrieve_relevant_knowledge(
        self,
        query: str,
        tool_names: Optional[List[str]] = None,
        mcp_caller: Optional[Any] = None,
        max_rules: int = 10,
        max_facts: int = 5,
    ) -> Dict[str, List[str]]:
        """
        Retrieve knowledge using HYBRID approach:
        - Procedural rules: DIRECT LOOKUP by tool name (guaranteed recall)
        - Parameter rules: AUTO-EXTRACTED from MCP schemas (prevents drift)
        - Semantic facts: Semantic search (cross-category relevance)
        - Error corrections: Semantic search (pattern matching)

        This ensures domain-specific rules (like warehouse codes) are always
        included when the corresponding tool is found by MCPToolRetriever.

        Args:
            query: The user query/task description
            tool_names: Optional list of tool names to fetch ALL procedural rules for
            mcp_caller: Optional MCPToolCaller instance for schema extraction
            max_rules: Maximum procedural rules to return
            max_facts: Maximum semantic facts to return

        Returns:
            Dict with 'procedural_rules', 'semantic_facts', 'error_corrections' keys
        """
        if not self.knowledge_base:
            return {'procedural_rules': [], 'semantic_facts': [], 'error_corrections': []}

        procedural_rules: List[str] = []
        semantic_facts: List[str] = []
        error_corrections: List[str] = []

        # =====================================================================
        # 1. DIRECT LOOKUP: Get ALL procedural rules for matched tools
        # =====================================================================
        # This guarantees 100% recall for rules tied to discovered tools,
        # regardless of semantic similarity between query and rule text.

        # Workflow expansion: if any tool from a workflow is found, include ALL
        # tools from that workflow to ensure complete knowledge is available.
        WORKFLOW_GROUPS = {
            'shipment': [
                'create_shipment_order',
                'add_shipment_order_line',
                'release_shipment_order',
                'reserve_shipment_order',
                'reserve_shipment_line_partial',
                'reserve_shipment_line_handling_unit',
            ],
        }

        expanded_tool_names = set(tool_names or [])

        # Check if any retrieved tool belongs to a workflow group
        for group_name, group_tools in WORKFLOW_GROUPS.items():
            if any(t in expanded_tool_names for t in group_tools):
                # Include all tools from this workflow group
                expanded_tool_names.update(group_tools)

        if expanded_tool_names:
            procedural = self.knowledge_base.get('procedural', {})
            for tool_name in expanded_tool_names:
                if tool_name in procedural:
                    tool_data = procedural[tool_name]
                    if isinstance(tool_data, dict):
                        for rule in tool_data.get('rules', []):
                            procedural_rules.append(f"[{tool_name}] {rule}")

        # =====================================================================
        # 1b. AUTO-EXTRACT: Parameter rules from MCP schemas
        # =====================================================================
        # This guarantees parameter names are always correct (from authoritative source)
        # and prevents drift between YAML and MCP server.
        if expanded_tool_names and mcp_caller is not None:
            for tool_name in expanded_tool_names:
                try:
                    schema = mcp_caller.get_tool_schema(tool_name)
                    if schema and "error" not in schema:
                        param_rule = extract_param_rules_from_schema(schema)
                        if param_rule:
                            procedural_rules.append(f"[{tool_name}] {param_rule}")
                except Exception:
                    # Skip if schema extraction fails - YAML rules still available
                    pass

        # =====================================================================
        # 2. SEMANTIC SEARCH: Get relevant facts and error corrections
        # =====================================================================
        # Facts and errors benefit from semantic search because:
        # - Facts often span multiple tools/categories
        # - Error patterns may match query context even without exact tool match
        if self.knowledge_retriever is not None:
            semantic_result = self.knowledge_retriever.retrieve(
                query,
                top_k=max_facts + 5  # Extra for filtering
            )
            semantic_facts = semantic_result.get('semantic_facts', [])[:max_facts]
            error_corrections = semantic_result.get('error_corrections', [])[:3]
        else:
            # Fallback to keyword matching for facts/errors if no retriever
            query_words = set(query.lower().split())

            semantic = self.knowledge_base.get('semantic', {})
            for category, category_data in semantic.items():
                if not isinstance(category_data, dict):
                    continue
                keywords = set(word.lower() for word in category_data.get('keywords', []))
                if query_words & keywords:
                    semantic_facts.extend(category_data.get('facts', []))

            common_errors = self.knowledge_base.get('common_errors', [])
            for error in common_errors:
                if not isinstance(error, dict):
                    continue
                keywords = set(word.lower() for word in error.get('keywords', []))
                if query_words & keywords:
                    correction = error.get('correction', '')
                    if correction:
                        error_corrections.append(correction)

        return {
            'procedural_rules': procedural_rules[:max_rules],
            'semantic_facts': semantic_facts[:max_facts],
            'error_corrections': error_corrections[:3],
        }

    def save_memories(self) -> None:
        """Persist episodic and tool memories to disk."""
        # Save episodic memories
        episodic_path = os.path.join(self.memory_dir, 'episodic_memories.json')
        try:
            with open(episodic_path, 'w', encoding='utf-8') as f:
                json.dump(self.episodic_memories, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # Save tool memories
        tool_path = os.path.join(self.memory_dir, 'tool_memories.json')
        try:
            with open(tool_path, 'w', encoding='utf-8') as f:
                json.dump(self.tool_memories, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _generate_task_id(self, task_description: str) -> str:
        """Generate a unique task ID based on description and timestamp."""
        timestamp = datetime.now().isoformat()
        content = f"{task_description}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for matching."""
        if not text:
            return []
        # Lowercase and extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        # Remove common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
            'can', 'has', 'have', 'been', 'will', 'would', 'could',
            'should', 'this', 'that', 'with', 'from', 'they', 'what',
            'when', 'where', 'which', 'how', 'why', 'your', 'into',
            'about', 'some', 'than', 'then', 'them', 'these', 'those',
            'only', 'also', 'just', 'more', 'most', 'other', 'such',
            'each', 'very', 'much', 'many', 'well', 'even', 'back',
            'being', 'there', 'after', 'before', 'through', 'between',
        }
        keywords = [w for w in words if w not in stop_words]
        return list(set(keywords))

    def _calculate_relevance_score(self, query_keywords: List[str], memory_keywords: List[str]) -> float:
        """Calculate relevance score based on keyword overlap."""
        if not query_keywords or not memory_keywords:
            return 0.0
        query_set = set(query_keywords)
        memory_set = set(memory_keywords)
        intersection = query_set.intersection(memory_set)
        # Jaccard similarity
        union = query_set.union(memory_set)
        if not union:
            return 0.0
        return len(intersection) / len(union)

    def store_episodic_memory(
        self,
        episode_memory: Dict,
        task_description: str,
        task_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> str:
        """
        Store an episodic memory from a completed or folded task.

        Args:
            episode_memory: The structured episode memory dict
            task_description: Description of the task
            task_id: Optional task ID (generated if not provided)
            dataset_name: Optional dataset name for categorization

        Returns:
            The task ID used for storage
        """
        if task_id is None:
            task_id = self._generate_task_id(task_description)

        # Extract keywords for later retrieval
        keywords = self._extract_keywords(task_description)
        if isinstance(episode_memory, dict):
            # Also extract from task_description in memory if present
            if 'task_description' in episode_memory:
                keywords.extend(self._extract_keywords(episode_memory['task_description']))

        memory_entry = {
            'task_id': task_id,
            'task_description': task_description,
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'keywords': list(set(keywords)),
            'episode_memory': episode_memory,
        }

        self.episodic_memories.append(memory_entry)

        # Prune if exceeds limit (keep most recent)
        if len(self.episodic_memories) > self.max_episodic_memories:
            self.episodic_memories = self.episodic_memories[-self.max_episodic_memories:]

        return task_id

    def store_tool_memory(
        self,
        tool_memory: Dict,
        task_description: str,
        task_id: Optional[str] = None,
    ) -> None:
        """
        Store tool memory and aggregate with existing tool experiences.

        Args:
            tool_memory: The structured tool memory dict
            task_description: Description of the task
            task_id: Optional task ID for reference
        """
        if not isinstance(tool_memory, dict):
            return

        tools_used = tool_memory.get('tools_used', [])
        derived_rules = tool_memory.get('derived_rules', [])

        for tool_info in tools_used:
            if not isinstance(tool_info, dict):
                continue

            tool_name = tool_info.get('tool_name', 'unknown')

            if tool_name not in self.tool_memories:
                self.tool_memories[tool_name] = []

            # Create memory entry with metadata
            memory_entry = {
                'task_id': task_id or self._generate_task_id(task_description),
                'task_description': task_description,
                'timestamp': datetime.now().isoformat(),
                'tool_info': tool_info,
                'derived_rules': [r for r in derived_rules if tool_name.lower() in r.lower()],
            }

            self.tool_memories[tool_name].append(memory_entry)

            # Prune if exceeds per-tool limit (keep most recent)
            if len(self.tool_memories[tool_name]) > self.max_tool_memories_per_tool:
                self.tool_memories[tool_name] = self.tool_memories[tool_name][-self.max_tool_memories_per_tool:]

    def update_working_memory(self, working_memory: Dict) -> None:
        """
        Update the current session's working memory.
        Working memory is not persisted across sessions.

        Args:
            working_memory: The structured working memory dict
        """
        self.working_memory = working_memory

    def retrieve_relevant_episodic_memories(
        self,
        query: str,
        top_k: Optional[int] = None,
        dataset_name: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve episodic memories relevant to the query.

        Args:
            query: The query text (usually the task description)
            top_k: Number of memories to retrieve (defaults to memory_retrieval_top_k)
            dataset_name: Optional filter by dataset name

        Returns:
            List of relevant episodic memory entries, sorted by relevance
        """
        if top_k is None:
            top_k = self.memory_retrieval_top_k

        if not self.episodic_memories:
            return []

        query_keywords = self._extract_keywords(query)

        # Score each memory
        scored_memories = []
        for memory in self.episodic_memories:
            # Filter by dataset if specified
            if dataset_name and memory.get('dataset_name') != dataset_name:
                continue

            memory_keywords = memory.get('keywords', [])
            score = self._calculate_relevance_score(query_keywords, memory_keywords)

            if score > 0:
                scored_memories.append((score, memory))

        # Sort by score (descending) and return top_k
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored_memories[:top_k]]

    def retrieve_relevant_tool_memories(
        self,
        tool_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """
        Retrieve aggregated tool memories for specified tools.

        Args:
            tool_names: List of tool names to retrieve memories for.
                       If None, returns all tool memories.

        Returns:
            Dict mapping tool names to aggregated memory info
        """
        if tool_names is None:
            target_tools = list(self.tool_memories.keys())
        else:
            target_tools = [t for t in tool_names if t in self.tool_memories]

        aggregated = {}
        for tool_name in target_tools:
            memories = self.tool_memories.get(tool_name, [])
            if not memories:
                continue

            # Aggregate tool info across memories
            total_uses = len(memories)
            success_count = 0
            all_effective_params = []
            all_errors = []
            all_rules = []
            all_experiences = []

            for mem in memories:
                tool_info = mem.get('tool_info', {})
                success_rate = tool_info.get('success_rate', 0.5)
                success_count += success_rate

                if 'effective_parameters' in tool_info:
                    all_effective_params.extend(tool_info['effective_parameters'])
                if 'common_errors' in tool_info:
                    all_errors.extend(tool_info['common_errors'])
                if 'experience' in tool_info:
                    all_experiences.append(tool_info['experience'])
                if 'derived_rules' in mem:
                    all_rules.extend(mem['derived_rules'])

            # Compute aggregated stats
            avg_success_rate = success_count / total_uses if total_uses > 0 else 0.0

            # Count parameter frequency
            param_counts = {}
            for p in all_effective_params:
                param_counts[p] = param_counts.get(p, 0) + 1
            top_params = sorted(param_counts.keys(), key=lambda x: param_counts[x], reverse=True)[:5]

            # Count error frequency
            error_counts = {}
            for e in all_errors:
                error_counts[e] = error_counts.get(e, 0) + 1
            top_errors = sorted(error_counts.keys(), key=lambda x: error_counts[x], reverse=True)[:3]

            aggregated[tool_name] = {
                'total_uses': total_uses,
                'avg_success_rate': round(avg_success_rate, 2),
                'effective_parameters': top_params,
                'common_errors': top_errors,
                'derived_rules': list(set(all_rules))[:5],
                'recent_experiences': all_experiences[-3:],  # Keep last 3 experiences
            }

        return aggregated

    def format_memories_for_prompt(
        self,
        query: str,
        available_tool_names: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
    ) -> str:
        """
        Format relevant memories as a string for injection into the prompt.

        Args:
            query: The task description/query
            available_tool_names: List of tool names available for the task
            dataset_name: Optional dataset name filter

        Returns:
            Formatted string of relevant memories
        """
        parts = []

        # 1. Retrieve and format domain knowledge (procedural rules + semantic facts)
        knowledge = self.retrieve_relevant_knowledge(
            query=query,
            tool_names=available_tool_names,
        )
        if any(knowledge.values()):
            parts.append("## Domain Knowledge\n")

            if knowledge['semantic_facts']:
                parts.append("### Facts")
                for fact in knowledge['semantic_facts']:
                    parts.append(f"- {fact}")
                parts.append("")

            if knowledge['procedural_rules']:
                parts.append("### Tool Rules")
                for rule in knowledge['procedural_rules']:
                    parts.append(f"- {rule}")
                parts.append("")

            if knowledge['error_corrections']:
                parts.append("### Avoid These Errors")
                for correction in knowledge['error_corrections']:
                    parts.append(f"- {correction}")
                parts.append("")

        # 2. Retrieve relevant episodic memories
        episodic_memories = self.retrieve_relevant_episodic_memories(
            query=query,
            dataset_name=dataset_name,
        )

        if episodic_memories:
            parts.append("## Relevant Past Experiences\n")
            for i, mem in enumerate(episodic_memories, 1):
                task_desc = mem.get('task_description', 'Unknown task')
                episode = mem.get('episode_memory', {})
                progress = episode.get('current_progress', 'N/A')
                key_events = episode.get('key_events', [])

                parts.append(f"### Experience {i}: {task_desc[:100]}...")
                parts.append(f"Progress: {progress}")
                if key_events:
                    parts.append("Key lessons:")
                    for event in key_events[:3]:  # Limit to 3 events
                        if isinstance(event, dict):
                            outcome = event.get('outcome', '')
                            if outcome:
                                parts.append(f"  - {outcome[:200]}")
                parts.append("")

        # Retrieve relevant tool memories
        if available_tool_names:
            tool_memories = self.retrieve_relevant_tool_memories(tool_names=available_tool_names)

            if tool_memories:
                parts.append("## Tool Experience Summary\n")
                for tool_name, info in tool_memories.items():
                    parts.append(f"### {tool_name}")
                    parts.append(f"  Uses: {info['total_uses']}, Success rate: {info['avg_success_rate']}")
                    if info['effective_parameters']:
                        parts.append(f"  Effective params: {', '.join(info['effective_parameters'][:3])}")
                    if info['common_errors']:
                        parts.append(f"  Watch out for: {', '.join(info['common_errors'][:2])}")
                    if info['derived_rules']:
                        parts.append(f"  Tips: {info['derived_rules'][0]}")
                    parts.append("")

        if not parts:
            return ""

        return "\n".join(parts)

    def store_complete_memory(
        self,
        episode_memory: Dict,
        working_memory: Dict,
        tool_memory: Dict,
        task_description: str,
        task_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> str:
        """
        Store all three types of memory from a fold or task completion.

        Args:
            episode_memory: The episodic memory dict
            working_memory: The working memory dict
            tool_memory: The tool memory dict
            task_description: Description of the task
            task_id: Optional task ID
            dataset_name: Optional dataset name

        Returns:
            The task ID used for storage
        """
        # Generate task ID if not provided
        if task_id is None:
            task_id = self._generate_task_id(task_description)

        # Store episodic memory
        self.store_episodic_memory(
            episode_memory=episode_memory,
            task_description=task_description,
            task_id=task_id,
            dataset_name=dataset_name,
        )

        # Store tool memory
        self.store_tool_memory(
            tool_memory=tool_memory,
            task_description=task_description,
            task_id=task_id,
        )

        # Update working memory (session only)
        self.update_working_memory(working_memory)

        # Persist to disk
        self.save_memories()

        return task_id

    def get_memory_stats(self) -> Dict:
        """Get statistics about stored memories."""
        return {
            'episodic_memory_count': len(self.episodic_memories),
            'tool_memory_count': sum(len(v) for v in self.tool_memories.values()),
            'tools_with_memories': list(self.tool_memories.keys()),
            'has_working_memory': self.working_memory is not None,
        }

    def clear_all_memories(self) -> None:
        """Clear all stored memories (both in memory and on disk)."""
        self.episodic_memories = []
        self.tool_memories = {}
        self.working_memory = None
        self.save_memories()

    def load_seed_memories(self, seed_path: Optional[str] = None) -> int:
        """
        DEPRECATED: Use load_knowledge_base() instead.

        This method now calls load_knowledge_base() for backward compatibility.
        If the old seed_memories.json exists, it logs a migration warning.

        Args:
            seed_path: Path to seed JSON file (ignored, kept for API compatibility)

        Returns:
            Number of knowledge entries loaded from YAML
        """
        # Check if old seed file exists and warn about migration
        if seed_path is None:
            seed_path = os.path.join(self.memory_dir, 'seed_memories.json')

        if os.path.exists(seed_path):
            print(f"  ⚠️  WARNING: Found deprecated {seed_path}")
            print(f"             Please migrate to config/ifs_knowledge.yaml")

        # Load from new knowledge base instead
        return self.load_knowledge_base()
