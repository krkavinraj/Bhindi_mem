"""
Neo4j graph manager for handling all database operations.
Manages nodes, relationships, and queries for the knowledge graph.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from neo4j import GraphDatabase
from .config import config, NODE_COLORS, RELATIONSHIP_TYPES

logger = logging.getLogger(__name__)

class GraphManager:
    """Manages Neo4j graph database operations."""
    
    def __init__(self):
        """Initialize Neo4j connection."""
        self.driver = None
        self.connect()
    
    def connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                config.neo4j.uri,
                auth=(config.neo4j.username, config.neo4j.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j: {e}")
            logger.warning("Running in demo mode without Neo4j")
            self.driver = None
            # Initialize in-memory storage for demo mode
            self._demo_nodes = []
            self._demo_relationships = []
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")
    
    def _demo_create_or_update_node(self, name: str, node_type: str, properties: Dict[str, Any]) -> bool:
        """Demo mode: Create or update node in memory."""
        # Find existing node
        for i, node in enumerate(self._demo_nodes):
            if node.get('name') == name:
                # Update existing node
                self._demo_nodes[i].update({
                    'type': node_type,
                    **properties,
                    'updated_at': 'demo_timestamp'
                })
                return True
        
        # Create new node
        self._demo_nodes.append({
            'name': name,
            'type': node_type,
            **properties,
            'created_at': 'demo_timestamp'
        })
        return True
    
    def create_or_update_node(self, name: str, node_type: str, properties: Dict[str, Any]) -> bool:
        """
        Create a new node or update existing one.
        
        Args:
            name: Node name (unique identifier)
            node_type: Type of node (Person, Concept, etc.)
            properties: Node properties
            
        Returns:
            True if successful, False otherwise
        """
        if not self.driver:
            return self._demo_create_or_update_node(name, node_type, properties)
        
        try:
            with self.driver.session() as session:
                # Merge node (create if not exists, update if exists)
                query = """
                MERGE (n {name: $name})
                SET n.type = $node_type
                SET n += $properties
                SET n.updated_at = datetime()
                RETURN n
                """
                
                result = session.run(query, {
                    "name": name,
                    "node_type": node_type,
                    "properties": properties
                })
                
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"Error creating/updating node {name}: {e}")
            return False
    
    def create_relationship(
        self,
        from_node: str,
        to_node: str,
        rel_type: str,
        properties: Dict[str, Any] = None
    ) -> bool:
        """
        Create a relationship between two nodes.
        
        Args:
            from_node: Source node name
            to_node: Target node name
            rel_type: Relationship type
            properties: Relationship properties
            
        Returns:
            True if successful, False otherwise
        """
        if not self.driver:
            logger.error("No database connection")
            return False
        
        if properties is None:
            properties = {}
        
        try:
            with self.driver.session() as session:
                # Create relationship
                query = """
                MATCH (a {name: $from_node})
                MATCH (b {name: $to_node})
                MERGE (a)-[r:%s]->(b)
                SET r += $properties
                SET r.created_at = datetime()
                RETURN r
                """ % rel_type
                
                result = session.run(query, {
                    "from_node": from_node,
                    "to_node": to_node,
                    "properties": properties
                })
                
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"Error creating relationship {from_node} -> {to_node}: {e}")
            return False
    
    def get_node(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a node by name.
        
        Args:
            name: Node name
            
        Returns:
            Node data dictionary or None
        """
        if not self.driver:
            return None
        
        try:
            with self.driver.session() as session:
                query = "MATCH (n {name: $name}) RETURN n"
                result = session.run(query, {"name": name})
                record = result.single()
                
                if record:
                    node = record["n"]
                    return dict(node)
                return None
                
        except Exception as e:
            logger.error(f"Error getting node {name}: {e}")
            return None
    
    def get_node_with_relationships(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a node with all its relationships.
        
        Args:
            name: Node name
            
        Returns:
            Dictionary with node and relationships data
        """
        if not self.driver:
            return None
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (n {name: $name})
                OPTIONAL MATCH (n)-[r]-(connected)
                RETURN n, collect({
                    relationship: r,
                    connected_node: connected,
                    direction: CASE 
                        WHEN startNode(r) = n THEN 'outgoing'
                        ELSE 'incoming'
                    END
                }) as relationships
                """
                
                result = session.run(query, {"name": name})
                record = result.single()
                
                if record:
                    node_data = {
                        "node": dict(record["n"]),
                        "relationships": []
                    }
                    
                    for rel_data in record["relationships"]:
                        if rel_data["relationship"]:
                            node_data["relationships"].append({
                                "type": type(rel_data["relationship"]).__name__,
                                "properties": dict(rel_data["relationship"]),
                                "connected_node": dict(rel_data["connected_node"]),
                                "direction": rel_data["direction"]
                            })
                    
                    return node_data
                return None
                
        except Exception as e:
            logger.error(f"Error getting node with relationships {name}: {e}")
            return None
    
    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """
        Get all nodes in the graph.
        
        Returns:
            List of all nodes
        """
        if not self.driver:
            return []
        
        try:
            with self.driver.session() as session:
                query = "MATCH (n) RETURN n ORDER BY n.name"
                result = session.run(query)
                
                nodes = []
                for record in result:
                    nodes.append(dict(record["n"]))
                
                return nodes
                
        except Exception as e:
            logger.error(f"Error getting all nodes: {e}")
            return []
    
    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """
        Get all nodes of a specific type.
        
        Args:
            node_type: Type of nodes to retrieve
            
        Returns:
            List of nodes of the specified type
        """
        if not self.driver:
            return []
        
        try:
            with self.driver.session() as session:
                query = "MATCH (n {type: $node_type}) RETURN n ORDER BY n.name"
                result = session.run(query, {"node_type": node_type})
                
                nodes = []
                for record in result:
                    nodes.append(dict(record["n"]))
                
                return nodes
                
        except Exception as e:
            logger.error(f"Error getting nodes by type {node_type}: {e}")
            return []
    
    def search_nodes_by_name(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search nodes by name (case-insensitive partial match).
        
        Args:
            search_term: Term to search for
            
        Returns:
            List of matching nodes
        """
        if not self.driver:
            return []
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (n) 
                WHERE toLower(n.name) CONTAINS toLower($search_term)
                RETURN n 
                ORDER BY n.name
                LIMIT 20
                """
                result = session.run(query, {"search_term": search_term})
                
                nodes = []
                for record in result:
                    nodes.append(dict(record["n"]))
                
                return nodes
                
        except Exception as e:
            logger.error(f"Error searching nodes by name {search_term}: {e}")
            return []
    
    def get_node_relationships(self, name: str) -> List[Dict[str, Any]]:
        """
        Get all relationships for a node.
        
        Args:
            name: Node name
            
        Returns:
            List of relationships
        """
        if not self.driver:
            return []
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (n {name: $name})-[r]-(connected)
                RETURN type(r) as relationship_type, 
                       properties(r) as properties,
                       connected.name as connected_name,
                       connected.type as connected_type,
                       CASE 
                           WHEN startNode(r) = n THEN 'outgoing'
                           ELSE 'incoming'
                       END as direction
                """
                
                result = session.run(query, {"name": name})
                
                relationships = []
                for record in result:
                    relationships.append({
                        "type": record["relationship_type"],
                        "properties": dict(record["properties"]),
                        "target_name": record["connected_name"],
                        "target_type": record["connected_type"],
                        "direction": record["direction"]
                    })
                
                return relationships
                
        except Exception as e:
            logger.error(f"Error getting relationships for {name}: {e}")
            return []
    
    def update_node_properties(self, name: str, properties: Dict[str, Any]) -> bool:
        """
        Update properties of an existing node.
        
        Args:
            name: Node name
            properties: Properties to update
            
        Returns:
            True if successful, False otherwise
        """
        if not self.driver:
            return False
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (n {name: $name})
                SET n += $properties
                SET n.updated_at = datetime()
                RETURN n
                """
                
                result = session.run(query, {
                    "name": name,
                    "properties": properties
                })
                
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"Error updating node {name}: {e}")
            return False
    
    def delete_node(self, name: str) -> bool:
        """
        Delete a node and all its relationships.
        
        Args:
            name: Node name
            
        Returns:
            True if successful, False otherwise
        """
        if not self.driver:
            return False
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (n {name: $name})
                DETACH DELETE n
                """
                
                session.run(query, {"name": name})
                return True
                
        except Exception as e:
            logger.error(f"Error deleting node {name}: {e}")
            return False
    
    def delete_relationship(self, from_node: str, to_node: str, rel_type: str) -> bool:
        """
        Delete a specific relationship.
        
        Args:
            from_node: Source node name
            to_node: Target node name
            rel_type: Relationship type
            
        Returns:
            True if successful, False otherwise
        """
        if not self.driver:
            return False
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (a {name: $from_node})-[r:%s]->(b {name: $to_node})
                DELETE r
                """ % rel_type
                
                session.run(query, {
                    "from_node": from_node,
                    "to_node": to_node
                })
                return True
                
        except Exception as e:
            logger.error(f"Error deleting relationship {from_node} -> {to_node}: {e}")
            return False
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the graph.
        
        Returns:
            Dictionary with graph statistics
        """
        if not self.driver:
            # Demo mode statistics
            node_types = {}
            for node in getattr(self, '_demo_nodes', []):
                node_type = node.get('type', 'Unknown')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            return {
                "nodes": len(getattr(self, '_demo_nodes', [])),
                "relationships": len(getattr(self, '_demo_relationships', [])),
                "node_types": node_types,
                "relationship_types": {}
            }
        
        try:
            with self.driver.session() as session:
                # Get node count
                node_result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = node_result.single()["node_count"]
                
                # Get relationship count
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_count = rel_result.single()["rel_count"]
                
                # Get node types
                type_result = session.run("""
                    MATCH (n) 
                    WHERE n.type IS NOT NULL
                    RETURN n.type as type, count(n) as count
                    ORDER BY count DESC
                """)
                node_types = {record["type"]: record["count"] for record in type_result}
                
                # Get relationship types
                rel_type_result = session.run("""
                    MATCH ()-[r]->() 
                    RETURN type(r) as rel_type, count(r) as count
                    ORDER BY count DESC
                """)
                relationship_types = {record["rel_type"]: record["count"] for record in rel_type_result}
                
                return {
                    "nodes": node_count,
                    "relationships": rel_count,
                    "node_types": node_types,
                    "relationship_types": relationship_types
                }
                
        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return {"error": str(e)}
    
    def get_graph_data_for_visualization(self, limit: int = 100) -> Dict[str, Any]:
        """
        Get graph data formatted for visualization.
        
        Args:
            limit: Maximum number of nodes to return
            
        Returns:
            Dictionary with nodes and edges for visualization
        """
        if not self.driver:
            # Demo mode visualization
            nodes = []
            for node in getattr(self, '_demo_nodes', []):
                node_type = node.get('type', 'Default')
                nodes.append({
                    "id": node.get('name', 'Unknown'),
                    "label": node.get('name', 'Unknown'),
                    "type": node_type,
                    "color": NODE_COLORS.get(node_type, NODE_COLORS["Default"]),
                    "properties": {k: v for k, v in node.items() if k not in ['name', 'type']}
                })
            
            return {
                "nodes": nodes,
                "edges": getattr(self, '_demo_relationships', []),
                "total_nodes": len(nodes),
                "total_edges": len(getattr(self, '_demo_relationships', []))
            }
        
        try:
            with self.driver.session() as session:
                # Get nodes with limit
                node_query = f"""
                MATCH (n) 
                RETURN n.name as name, 
                       n.type as type, 
                       properties(n) as properties
                LIMIT {limit}
                """
                node_result = session.run(node_query)
                
                nodes = []
                node_names = set()
                
                for record in node_result:
                    name = record["name"]
                    node_type = record["type"] or "Default"
                    properties = dict(record["properties"])
                    
                    nodes.append({
                        "id": name,
                        "label": name,
                        "type": node_type,
                        "color": NODE_COLORS.get(node_type, NODE_COLORS["Default"]),
                        "properties": properties
                    })
                    node_names.add(name)
                
                # Get relationships between the selected nodes
                if node_names:
                    rel_query = """
                    MATCH (a)-[r]->(b)
                    WHERE a.name IN $node_names AND b.name IN $node_names
                    RETURN a.name as source, 
                           b.name as target, 
                           type(r) as type,
                           properties(r) as properties
                    """
                    rel_result = session.run(rel_query, {"node_names": list(node_names)})
                    
                    edges = []
                    for record in rel_result:
                        edges.append({
                            "source": record["source"],
                            "target": record["target"],
                            "type": record["type"],
                            "label": record["type"],
                            "properties": dict(record["properties"])
                        })
                else:
                    edges = []
                
                return {
                    "nodes": nodes,
                    "edges": edges,
                    "total_nodes": len(nodes),
                    "total_edges": len(edges)
                }
                
        except Exception as e:
            logger.error(f"Error getting graph data for visualization: {e}")
            return {"nodes": [], "edges": [], "error": str(e)}
