"""Graph Maintenance Tools for Neo4j operations."""

import logging
import uuid
from typing import Any

from ...core.embeddings.base import EmbeddingProvider
from ..database.neo4j_client import Neo4jConnection

logger = logging.getLogger(__name__)


class GraphMaintenanceTools:
    """Tool class for graph maintenance operations exposed to OpenAI API."""
    
    def __init__(self, neo4j_client: Neo4jConnection, embedding_provider: EmbeddingProvider):
        """Initialize with Neo4j client and embedding provider.
        
        Args:
            neo4j_client: Neo4j connection instance
            embedding_provider: Embedding provider for vector operations
        """
        self.neo4j_client = neo4j_client
        self.embedding_provider = embedding_provider
    
    async def search_similar_nodes(self, query: str, threshold: float = 0.8, user_id: str = None) -> str:
        """Search for similar nodes in the graph using vector similarity.
        
        Args:
            query: Search query string
            threshold: Similarity threshold (0.0-1.0)
            
        Returns:
            Formatted string summary of similar nodes
        """
        try:
            async with self.embedding_provider as provider:
                query_embedding = await provider.embed_text(query)
            
            # Ensure Neo4j connection is active
            if not self.neo4j_client.driver:
                await self.neo4j_client.connect()
            
            results = await self.neo4j_client.query_vector_index(
                query_embedding=query_embedding,
                k=10,
                user_id=user_id,
                similarity_threshold=threshold
            )
            
            if not results:
                return f"No similar nodes found for query '{query}' with threshold {threshold}"
            
            summary_parts = [f"Found {len(results)} similar nodes for '{query}':"]
            
            for result in results:
                entity = result.get('entity', {}) if result else {}
                score = result.get('score', 0.0) if result else 0.0
                
                node_id = entity.get('entity_id', 'Unknown')
                name = entity.get('name', 'Unnamed')
                entity_type = entity.get('entity_type', 'Unknown')
                properties = entity.get('properties', {})
                
                # Handle case where properties might be None
                if properties is None:
                    properties = {}
                
                prop_str = ", ".join([f"{k}: {v}" for k, v in properties.items()]) if properties else "No properties"
                summary_parts.append(
                    f"- ID: {node_id}, Name: {name}, Type: {entity_type}, "
                    f"Similarity: {score:.3f}, Properties: {{{prop_str}}}"
                )
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Search similar nodes failed: {e}")
            return f"Error: Failed to search similar nodes - {str(e)}"
    
    async def upsert_node(
        self, 
        name: str, 
        label: str, 
        properties: dict[str, Any] = None, 
        merge_id: str = None,
        user_id: str = None
    ) -> str:
        """Create or update a node in the graph.
        
        Args:
            name: Node name
            label: Node label/type
            properties: Node properties dictionary
            merge_id: Optional existing node ID to update
            
        Returns:
            Descriptive success or error message
        """
        try:
            # Handle None properties
            if properties is None:
                properties = {}
            
            # Generate embedding for the node content
            embedding_text = f"{name} {' '.join(str(v) for v in properties.values())}"
            async with self.embedding_provider as provider:
                embedding = await provider.embed_text(embedding_text)
            
            # Ensure Neo4j connection is active
            if not self.neo4j_client.driver:
                await self.neo4j_client.connect()
            
            if merge_id and merge_id.strip():
                # Update existing node
                cypher = """
                MATCH (n)
                WHERE n.entity_id = $merge_id
                SET n.name = $name,
                    n.entity_type = $label,
                    n.embedding = $embedding,
                    n.last_updated = datetime()
                """
                
                parameters = {
                    "merge_id": merge_id,
                    "name": name,
                    "label": label,
                    "embedding": embedding
                }
                
                # Add user_id if provided
                if user_id:
                    cypher += "\nSET n.user_id = $user_id"
                    parameters["user_id"] = user_id
                
                # Only set properties if they exist and are not empty
                if properties:
                    cypher += "\nSET n.properties = $properties"
                    parameters["properties"] = properties
                
                cypher += "\nRETURN n.entity_id as id, n.name as name"
                
                result = await self.neo4j_client.execute_write_query(cypher, parameters)
                
                if result:
                    return f"Success: Updated node '{name}' (ID: {merge_id})"
                else:
                    return f"Error: Node with ID {merge_id} not found"
            
            else:
                # Create new node
                entity_id = str(uuid.uuid4())
                
                cypher = f"""
                MERGE (n:{label} {{name: $name}})
                SET n.entity_id = $entity_id,
                    n.entity_type = $label,
                    n.name = $name,
                    n.embedding = $embedding,
                    n.created_at = datetime(),
                    n.last_updated = datetime()
                """
                
                parameters = {
                    "name": name,
                    "label": label,
                    "entity_id": entity_id,
                    "embedding": embedding
                }
                
                # Add user_id if provided
                if user_id:
                    cypher += "\nSET n.user_id = $user_id"
                    parameters["user_id"] = user_id
                
                # Only set properties if they exist and are not empty
                if properties:
                    cypher += "\nSET n.properties = $properties"
                    parameters["properties"] = properties
                
                cypher += "\nRETURN n.entity_id as id, n.name as name"
                
                result = await self.neo4j_client.execute_write_query(cypher, parameters)
                
                if result:
                    return f"Success: Created node '{name}' (ID: {entity_id})"
                else:
                    return f"Error: Failed to create node '{name}'"
                    
        except Exception as e:
            logger.error(f"Upsert node failed: {e}")
            return f"Error: Failed to upsert node '{name}' - {str(e)}"
    
    async def create_relationship(
        self, 
        from_name: str, 
        to_name: str, 
        relation_type: str, 
        properties: dict[str, Any] = None
    ) -> str:
        """Create a relationship between two nodes.
        
        Args:
            from_name: Source node name
            to_name: Target node name
            relation_type: Relationship type
            properties: Relationship properties
            
        Returns:
            Descriptive success or error message
        """
        try:
            # Handle None properties
            if properties is None:
                properties = {}
            
            # Ensure Neo4j connection is active
            if not self.neo4j_client.driver:
                await self.neo4j_client.connect()
            
            # Find nodes by name (fuzzy match)
            cypher = """
            MATCH (from_node) WHERE from_node.name = $from_name
            MATCH (to_node) WHERE to_node.name = $to_name
            WITH from_node, to_node
            WHERE from_node IS NOT NULL AND to_node IS NOT NULL
            MERGE (from_node)-[r:%s]->(to_node)
            SET r.created_at = CASE WHEN r.created_at IS NULL THEN datetime() ELSE r.created_at END,
                r.last_updated = datetime()
            """ % relation_type
            
            parameters = {
                "from_name": from_name,
                "to_name": to_name
            }
            
            # Only add properties if they exist and are not empty
            if properties:
                cypher += "\nSET r += $properties"
                parameters["properties"] = properties
            
            cypher += "\nRETURN from_node.name as from_name, to_node.name as to_name, type(r) as rel_type"
            
            result = await self.neo4j_client.execute_write_query(cypher, parameters)
            
            if result:
                return f"Success: Created relationship '{relation_type}' from '{from_name}' to '{to_name}'"
            else:
                # Try to find which nodes exist
                check_cypher = """
                OPTIONAL MATCH (from_node) WHERE from_node.name = $from_name
                OPTIONAL MATCH (to_node) WHERE to_node.name = $to_name
                RETURN from_node IS NOT NULL as from_exists, to_node IS NOT NULL as to_exists
                """
                
                check_result = await self.neo4j_client.execute_query(check_cypher, {
                    "from_name": from_name,
                    "to_name": to_name
                })
                
                if check_result:
                    from_exists = check_result[0].get('from_exists', False)
                    to_exists = check_result[0].get('to_exists', False)
                    
                    if not from_exists and not to_exists:
                        return f"Error: Both nodes '{from_name}' and '{to_name}' not found"
                    elif not from_exists:
                        return f"Error: Source node '{from_name}' not found"
                    elif not to_exists:
                        return f"Error: Target node '{to_name}' not found"
                
                return f"Error: Failed to create relationship between '{from_name}' and '{to_name}'"
                
        except Exception as e:
            logger.error(f"Create relationship failed: {e}")
            return f"Error: Failed to create relationship '{relation_type}' - {str(e)}"