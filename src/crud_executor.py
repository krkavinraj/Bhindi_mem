"""
CRUD operations executor for the knowledge graph.
Handles Create, Read, Update, Delete operations based on parsed conversations.
"""
import logging
from typing import List, Dict, Any, Optional
from .gpt_parser import ParsedConversation, Entity, Relationship
from .graph_manager import GraphManager

logger = logging.getLogger(__name__)

class CRUDExecutor:
    """Executes CRUD operations on the knowledge graph."""
    
    def __init__(self, graph_manager: GraphManager):
        """
        Initialize CRUD executor.
        
        Args:
            graph_manager: GraphManager instance for database operations
        """
        self.graph_manager = graph_manager
    
    def execute_parsed_conversation(self, parsed: ParsedConversation) -> Dict[str, Any]:
        """
        Execute CRUD operations based on parsed conversation.
        
        Args:
            parsed: ParsedConversation object with extracted information
            
        Returns:
            Dictionary with execution results
        """
        try:
            result = {
                "intent": parsed.intent,
                "success": False,
                "message": "",
                "entities_processed": 0,
                "relationships_processed": 0,
                "data": None
            }
            
            if parsed.intent == "CREATE":
                return self._execute_create(parsed, result)
            elif parsed.intent == "READ" or parsed.intent == "QUERY":
                return self._execute_read(parsed, result)
            elif parsed.intent == "UPDATE":
                return self._execute_update(parsed, result)
            elif parsed.intent == "DELETE":
                return self._execute_delete(parsed, result)
            else:
                result["message"] = f"Unknown intent: {parsed.intent}"
                return result
                
        except Exception as e:
            logger.error(f"Error executing CRUD operation: {e}")
            return {
                "intent": parsed.intent,
                "success": False,
                "message": f"Error: {str(e)}",
                "entities_processed": 0,
                "relationships_processed": 0,
                "data": None
            }
    
    def _execute_create(self, parsed: ParsedConversation, result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CREATE operations."""
        entities_created = 0
        relationships_created = 0
        
        # Create entities
        for entity in parsed.entities:
            if self._should_process_entity(entity):
                success = self.graph_manager.create_or_update_node(
                    entity.name,
                    entity.type,
                    entity.properties
                )
                if success:
                    entities_created += 1
                    logger.info(f"Created/Updated entity: {entity.name} ({entity.type})")
        
        # Create relationships
        for relationship in parsed.relationships:
            if self._should_process_relationship(relationship):
                success = self.graph_manager.create_relationship(
                    relationship.from_entity,
                    relationship.to_entity,
                    relationship.type,
                    relationship.properties
                )
                if success:
                    relationships_created += 1
                    logger.info(f"Created relationship: {relationship.from_entity} -> {relationship.to_entity}")
        
        result.update({
            "success": entities_created > 0 or relationships_created > 0,
            "message": f"Created {entities_created} entities and {relationships_created} relationships",
            "entities_processed": entities_created,
            "relationships_processed": relationships_created
        })
        
        return result
    
    def _execute_read(self, parsed: ParsedConversation, result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute READ/QUERY operations."""
        # Determine what to query based on entities mentioned
        query_results = []
        
        if not parsed.entities:
            # General query - get overview
            stats = self.graph_manager.get_graph_statistics()
            result.update({
                "success": True,
                "message": "Retrieved graph overview",
                "data": stats
            })
            return result
        
        # Query specific entities
        for entity in parsed.entities:
            if entity.name.lower() != "user":  # Skip generic user entity for queries
                node_data = self.graph_manager.get_node_with_relationships(entity.name)
                if node_data:
                    query_results.append(node_data)
        
        # If no specific entities, query by type or properties
        if not query_results and parsed.entities:
            for entity in parsed.entities:
                if entity.type and entity.type != "Person":
                    nodes = self.graph_manager.get_nodes_by_type(entity.type)
                    query_results.extend(nodes)
        
        result.update({
            "success": len(query_results) > 0,
            "message": f"Found {len(query_results)} results",
            "data": query_results,
            "entities_processed": len(query_results)
        })
        
        return result
    
    def _execute_update(self, parsed: ParsedConversation, result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute UPDATE operations."""
        entities_updated = 0
        relationships_updated = 0
        
        # Update entities
        for entity in parsed.entities:
            if self._should_process_entity(entity) and entity.name.lower() != "user":
                # Check if entity exists
                existing = self.graph_manager.get_node(entity.name)
                if existing:
                    success = self.graph_manager.update_node_properties(
                        entity.name,
                        entity.properties
                    )
                    if success:
                        entities_updated += 1
                        logger.info(f"Updated entity: {entity.name}")
        
        # Update relationships (by recreating with new properties)
        for relationship in parsed.relationships:
            if self._should_process_relationship(relationship):
                # Delete old relationship and create new one
                self.graph_manager.delete_relationship(
                    relationship.from_entity,
                    relationship.to_entity,
                    relationship.type
                )
                success = self.graph_manager.create_relationship(
                    relationship.from_entity,
                    relationship.to_entity,
                    relationship.type,
                    relationship.properties
                )
                if success:
                    relationships_updated += 1
                    logger.info(f"Updated relationship: {relationship.from_entity} -> {relationship.to_entity}")
        
        result.update({
            "success": entities_updated > 0 or relationships_updated > 0,
            "message": f"Updated {entities_updated} entities and {relationships_updated} relationships",
            "entities_processed": entities_updated,
            "relationships_processed": relationships_updated
        })
        
        return result
    
    def _execute_delete(self, parsed: ParsedConversation, result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DELETE operations."""
        entities_deleted = 0
        relationships_deleted = 0
        
        # Delete relationships first
        for relationship in parsed.relationships:
            if self._should_process_relationship(relationship):
                success = self.graph_manager.delete_relationship(
                    relationship.from_entity,
                    relationship.to_entity,
                    relationship.type
                )
                if success:
                    relationships_deleted += 1
                    logger.info(f"Deleted relationship: {relationship.from_entity} -> {relationship.to_entity}")
        
        # Delete entities (except User)
        for entity in parsed.entities:
            if (self._should_process_entity(entity) and 
                entity.name.lower() not in ["user", "me", "i"]):
                success = self.graph_manager.delete_node(entity.name)
                if success:
                    entities_deleted += 1
                    logger.info(f"Deleted entity: {entity.name}")
        
        result.update({
            "success": entities_deleted > 0 or relationships_deleted > 0,
            "message": f"Deleted {entities_deleted} entities and {relationships_deleted} relationships",
            "entities_processed": entities_deleted,
            "relationships_processed": relationships_deleted
        })
        
        return result
    
    def _should_process_entity(self, entity: Entity) -> bool:
        """
        Determine if an entity should be processed.
        
        Args:
            entity: Entity to check
            
        Returns:
            True if entity should be processed
        """
        # Skip entities with very low confidence
        if entity.confidence < 0.3:
            return False
        
        # Skip empty or invalid names
        if not entity.name or not entity.name.strip():
            return False
        
        # Skip very generic entities
        generic_names = ["thing", "stuff", "something", "anything", "everything"]
        if entity.name.lower() in generic_names:
            return False
        
        return True
    
    def _should_process_relationship(self, relationship: Relationship) -> bool:
        """
        Determine if a relationship should be processed.
        
        Args:
            relationship: Relationship to check
            
        Returns:
            True if relationship should be processed
        """
        # Skip relationships with very low confidence
        if relationship.confidence < 0.3:
            return False
        
        # Skip relationships with empty entities
        if (not relationship.from_entity or not relationship.to_entity or
            not relationship.from_entity.strip() or not relationship.to_entity.strip()):
            return False
        
        # Skip self-relationships
        if relationship.from_entity.lower() == relationship.to_entity.lower():
            return False
        
        return True
    
    def get_execution_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of multiple CRUD executions.
        
        Args:
            results: List of execution results
            
        Returns:
            Summary dictionary
        """
        total_entities = sum(r.get("entities_processed", 0) for r in results)
        total_relationships = sum(r.get("relationships_processed", 0) for r in results)
        successful_operations = sum(1 for r in results if r.get("success", False))
        
        intent_counts = {}
        for result in results:
            intent = result.get("intent", "unknown")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        return {
            "total_operations": len(results),
            "successful_operations": successful_operations,
            "total_entities_processed": total_entities,
            "total_relationships_processed": total_relationships,
            "intent_breakdown": intent_counts,
            "success_rate": successful_operations / len(results) if results else 0.0
        }
