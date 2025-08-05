"""
Memory retriever for contextual graph querying and intelligent information retrieval.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from .graph_manager import GraphManager

logger = logging.getLogger(__name__)

class MemoryRetriever:
    """Intelligent memory retrieval from the knowledge graph."""
    
    def __init__(self, graph_manager: GraphManager):
        """
        Initialize memory retriever.
        
        Args:
            graph_manager: GraphManager instance for database operations
        """
        self.graph_manager = graph_manager
        self.embedding_model = None
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model for semantic search."""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Initialized sentence transformer model")
        except Exception as e:
            logger.warning(f"Could not initialize embedding model: {e}")
            self.embedding_model = None
    
    def retrieve_relevant_context(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Retrieve relevant context from the knowledge graph based on query.
        
        Args:
            query: User query or conversation
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary with relevant context information
        """
        try:
            context = {
                "entities": [],
                "relationships": [],
                "semantic_matches": [],
                "statistics": {},
                "query": query
            }
            
            # Get keyword-based matches
            keyword_matches = self._get_keyword_matches(query, max_results)
            context["entities"].extend(keyword_matches.get("entities", []))
            context["relationships"].extend(keyword_matches.get("relationships", []))
            
            # Get semantic matches if embedding model is available
            if self.embedding_model:
                semantic_matches = self._get_semantic_matches(query, max_results)
                context["semantic_matches"] = semantic_matches
            
            # Get relevant statistics
            context["statistics"] = self.graph_manager.get_graph_statistics()
            
            # Get user-related information
            user_context = self._get_user_context()
            if user_context:
                context["user_profile"] = user_context
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return {"error": str(e), "query": query}
    
    def _get_keyword_matches(self, query: str, max_results: int) -> Dict[str, List[Dict[str, Any]]]:
        """Get matches based on keyword search."""
        results = {"entities": [], "relationships": []}
        
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        
        # Search for entities containing keywords
        for keyword in keywords:
            entities = self.graph_manager.search_nodes_by_name(keyword)
            results["entities"].extend(entities[:max_results])
        
        # Search for relationships involving keyword entities
        for entity in results["entities"]:
            relationships = self.graph_manager.get_node_relationships(entity.get("name", ""))
            results["relationships"].extend(relationships)
        
        # Remove duplicates and limit results
        results["entities"] = self._deduplicate_list(results["entities"], "name")[:max_results]
        results["relationships"] = self._deduplicate_list(results["relationships"], "id")[:max_results]
        
        return results
    
    def _get_semantic_matches(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Get semantically similar content using embeddings."""
        if not self.embedding_model:
            return []
        
        try:
            # Get all nodes with text content
            all_nodes = self.graph_manager.get_all_nodes()
            
            if not all_nodes:
                return []
            
            # Create text representations of nodes
            node_texts = []
            for node in all_nodes:
                text = self._create_node_text(node)
                node_texts.append(text)
            
            # Generate embeddings
            query_embedding = self.embedding_model.encode([query])
            node_embeddings = self.embedding_model.encode(node_texts)
            
            # Calculate similarities
            similarities = np.dot(query_embedding, node_embeddings.T)[0]
            
            # Get top matches
            top_indices = np.argsort(similarities)[-max_results:][::-1]
            
            matches = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Minimum similarity threshold
                    matches.append({
                        "node": all_nodes[idx],
                        "similarity": float(similarities[idx]),
                        "text": node_texts[idx]
                    })
            
            return matches
            
        except Exception as e:
            logger.error(f"Error in semantic matching: {e}")
            return []
    
    def _create_node_text(self, node: Dict[str, Any]) -> str:
        """Create a text representation of a node for embedding."""
        text_parts = []
        
        # Add name
        if node.get("name"):
            text_parts.append(node["name"])
        
        # Add type
        if node.get("type"):
            text_parts.append(f"Type: {node['type']}")
        
        # Add properties
        properties = node.get("properties", {})
        for key, value in properties.items():
            if isinstance(value, str) and len(value) < 100:
                text_parts.append(f"{key}: {value}")
        
        return " | ".join(text_parts)
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query."""
        # Simple keyword extraction (can be enhanced with NLP)
        stop_words = {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
            "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
            "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
            "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
            "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
            "while", "of", "at", "by", "for", "with", "through", "during", "before", "after",
            "above", "below", "up", "down", "in", "out", "on", "off", "over", "under", "again",
            "further", "then", "once"
        }
        
        words = query.lower().split()
        keywords = [word.strip(".,!?;:") for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:5]  # Limit to top 5 keywords
    
    def _deduplicate_list(self, items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
        """Remove duplicates from list based on key."""
        seen = set()
        unique_items = []
        
        for item in items:
            identifier = item.get(key)
            if identifier and identifier not in seen:
                seen.add(identifier)
                unique_items.append(item)
        
        return unique_items
    
    def _get_user_context(self) -> Optional[Dict[str, Any]]:
        """Get context about the user from the graph."""
        try:
            user_node = self.graph_manager.get_node("User")
            if not user_node:
                return None
            
            # Get user's relationships
            relationships = self.graph_manager.get_node_relationships("User")
            
            # Categorize relationships
            context = {
                "skills": [],
                "preferences": [],
                "goals": [],
                "organizations": [],
                "locations": [],
                "memories": []
            }
            
            for rel in relationships:
                target_type = rel.get("target_type", "").lower()
                rel_type = rel.get("type", "")
                
                if target_type == "skill" or rel_type == "SKILLED_IN":
                    context["skills"].append(rel)
                elif target_type == "preference" or rel_type in ["LIKES", "DISLIKES"]:
                    context["preferences"].append(rel)
                elif target_type == "goal" or rel_type == "WANTS_TO":
                    context["goals"].append(rel)
                elif target_type == "organization" or rel_type == "WORKS_AT":
                    context["organizations"].append(rel)
                elif target_type == "location" or rel_type == "LIVES_IN":
                    context["locations"].append(rel)
                elif target_type == "memory" or rel_type == "REMEMBERS":
                    context["memories"].append(rel)
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return None
    
    def get_conversation_context(self, recent_conversations: List[str], max_context: int = 3) -> str:
        """
        Generate context string from recent conversations.
        
        Args:
            recent_conversations: List of recent conversation strings
            max_context: Maximum number of conversations to include
            
        Returns:
            Context string for GPT
        """
        if not recent_conversations:
            return ""
        
        # Take the most recent conversations
        recent = recent_conversations[-max_context:]
        
        context_parts = []
        for i, conv in enumerate(recent, 1):
            context_parts.append(f"Previous conversation {i}: {conv}")
        
        return "\n".join(context_parts)
    
    def find_related_entities(self, entity_name: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Find entities related to a given entity within specified depth.
        
        Args:
            entity_name: Name of the starting entity
            max_depth: Maximum relationship depth to explore
            
        Returns:
            List of related entities with relationship paths
        """
        try:
            related_entities = []
            visited = set()
            queue = [(entity_name, 0, [])]  # (entity, depth, path)
            
            while queue and len(related_entities) < 20:  # Limit results
                current_entity, depth, path = queue.pop(0)
                
                if current_entity in visited or depth > max_depth:
                    continue
                
                visited.add(current_entity)
                
                # Get relationships for current entity
                relationships = self.graph_manager.get_node_relationships(current_entity)
                
                for rel in relationships:
                    target = rel.get("target_name")
                    if target and target not in visited:
                        new_path = path + [rel]
                        related_entities.append({
                            "entity": target,
                            "depth": depth + 1,
                            "path": new_path,
                            "relationship_type": rel.get("type")
                        })
                        
                        if depth + 1 < max_depth:
                            queue.append((target, depth + 1, new_path))
            
            return related_entities
            
        except Exception as e:
            logger.error(f"Error finding related entities: {e}")
            return []
