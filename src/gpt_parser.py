"""
Azure GPT-4o powered conversation parser for entity and relationship extraction.
"""
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from openai import AzureOpenAI
from .config import config

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Represents an extracted entity."""
    name: str
    type: str
    properties: Dict[str, Any]
    confidence: float = 0.0

@dataclass
class Relationship:
    """Represents an extracted relationship."""
    from_entity: str
    to_entity: str
    type: str
    properties: Dict[str, Any]
    confidence: float = 0.0

@dataclass
class ParsedConversation:
    """Result of conversation parsing."""
    intent: str
    entities: List[Entity]
    relationships: List[Relationship]
    confidence: float
    raw_response: str

class GPTParser:
    """Azure GPT-4o powered conversation parser."""
    
    def __init__(self):
        """Initialize the GPT parser with Azure OpenAI client."""
        # Extract base endpoint from full URL if needed
        base_endpoint = config.azure_openai.endpoint
        if '/openai/deployments/' in base_endpoint:
            base_endpoint = base_endpoint.split('/openai/deployments/')[0]
        
        self.client = AzureOpenAI(
            api_key=config.azure_openai.api_key,
            api_version=config.azure_openai.api_version,
            azure_endpoint=base_endpoint
        )
        self.deployment_name = config.azure_openai.deployment_name
        
    def parse_conversation(self, user_input: str, context: Optional[str] = None) -> ParsedConversation:
        """
        Parse user conversation to extract entities, relationships, and intent.
        
        Args:
            user_input: The user's message
            context: Optional context from previous conversations
            
        Returns:
            ParsedConversation object with extracted information
        """
        try:
            # Create the system prompt for entity and relationship extraction
            system_prompt = self._create_system_prompt()
            
            # Create user prompt with context
            user_prompt = self._create_user_prompt(user_input, context)
            
            # Call Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=config.azure_openai.max_tokens,
                temperature=config.azure_openai.temperature,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            raw_response = response.choices[0].message.content
            parsed_data = json.loads(raw_response)
            
            return self._create_parsed_conversation(parsed_data, raw_response)
            
        except Exception as e:
            logger.error(f"Error parsing conversation: {e}")
            return ParsedConversation(
                intent="unknown",
                entities=[],
                relationships=[],
                confidence=0.0,
                raw_response=str(e)
            )
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for GPT-4o."""
        return """You are an expert knowledge graph entity and relationship extractor. 
        Your task is to analyze user conversations and extract:
        1. Intent (CREATE, READ, UPDATE, DELETE, or QUERY)
        2. Entities (nodes in the knowledge graph)
        3. Relationships (edges between entities)

        Entity Types Available:
        - Person: People and their attributes
        - Concept: Abstract ideas, topics, subjects
        - Event: Occurrences, experiences, meetings
        - Preference: Likes, dislikes, opinions
        - Location: Places, geographical entities
        - Organization: Companies, institutions, groups
        - Skill: Abilities, competencies, talents
        - Goal: Objectives, aspirations, targets
        - Memory: Specific memories, experiences

        Relationship Types Available:
        KNOWS, LIKES, DISLIKES, WORKS_AT, LIVES_IN, ATTENDED, SKILLED_IN, 
        WANTS_TO, REMEMBERS, RELATED_TO, PART_OF, CREATED, LEARNED

        Return your analysis as a JSON object with this exact structure:
        {
            "intent": "CREATE|READ|UPDATE|DELETE|QUERY",
            "entities": [
                {
                    "name": "entity_name",
                    "type": "entity_type",
                    "properties": {"key": "value"},
                    "confidence": 0.95
                }
            ],
            "relationships": [
                {
                    "from_entity": "entity1_name",
                    "to_entity": "entity2_name", 
                    "type": "relationship_type",
                    "properties": {"key": "value"},
                    "confidence": 0.90
                }
            ],
            "confidence": 0.85
        }

        Guidelines:
        - Always include a "User" entity for the person speaking
        - Extract specific, meaningful entities (avoid generic terms)
        - Infer reasonable properties from context
        - Use high confidence (0.8+) for explicit information
        - Use medium confidence (0.5-0.8) for inferred information
        - Use CREATE intent for new information sharing
        - Use QUERY intent for questions about existing information
        - Use UPDATE intent for modifying existing information
        - Use DELETE intent for removing information"""

    def _create_user_prompt(self, user_input: str, context: Optional[str] = None) -> str:
        """Create the user prompt with input and context."""
        prompt = f"User Input: {user_input}\n\n"
        
        if context:
            prompt += f"Previous Context: {context}\n\n"
        
        prompt += "Please analyze this conversation and extract entities, relationships, and intent as specified."
        
        return prompt
    
    def _create_parsed_conversation(self, parsed_data: Dict[str, Any], raw_response: str) -> ParsedConversation:
        """Create ParsedConversation object from parsed JSON data."""
        # Extract entities
        entities = []
        for entity_data in parsed_data.get("entities", []):
            entities.append(Entity(
                name=entity_data.get("name", ""),
                type=entity_data.get("type", ""),
                properties=entity_data.get("properties", {}),
                confidence=entity_data.get("confidence", 0.0)
            ))
        
        # Extract relationships
        relationships = []
        for rel_data in parsed_data.get("relationships", []):
            relationships.append(Relationship(
                from_entity=rel_data.get("from_entity", ""),
                to_entity=rel_data.get("to_entity", ""),
                type=rel_data.get("type", ""),
                properties=rel_data.get("properties", {}),
                confidence=rel_data.get("confidence", 0.0)
            ))
        
        return ParsedConversation(
            intent=parsed_data.get("intent", "unknown"),
            entities=entities,
            relationships=relationships,
            confidence=parsed_data.get("confidence", 0.0),
            raw_response=raw_response
        )

class IntentClassifier:
    """Classifies user intent for CRUD operations."""
    
    INTENT_KEYWORDS = {
        "CREATE": ["tell", "add", "create", "new", "i am", "i like", "i work", "i live", "i want"],
        "READ": ["show", "find", "get", "what", "who", "where", "when", "how", "list"],
        "UPDATE": ["change", "update", "modify", "edit", "correct", "fix"],
        "DELETE": ["remove", "delete", "forget", "clear", "stop"],
        "QUERY": ["?", "what", "who", "where", "when", "why", "how", "tell me about"]
    }
    
    @classmethod
    def classify_intent(cls, user_input: str) -> str:
        """
        Classify user intent based on keywords and patterns.
        
        Args:
            user_input: The user's message
            
        Returns:
            Intent classification (CREATE, READ, UPDATE, DELETE, QUERY)
        """
        user_input_lower = user_input.lower()
        
        # Check for question marks (strong indicator of QUERY)
        if "?" in user_input:
            return "QUERY"
        
        # Score each intent based on keyword matches
        intent_scores = {}
        for intent, keywords in cls.INTENT_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in user_input_lower)
            intent_scores[intent] = score
        
        # Return the intent with highest score, default to CREATE
        if max(intent_scores.values()) > 0:
            return max(intent_scores, key=intent_scores.get)
        
        return "CREATE"  # Default for statements about self

class EntityExtractor:
    """Extracts and validates entities from parsed data."""
    
    VALID_ENTITY_TYPES = {
        "Person", "Concept", "Event", "Preference", "Location", 
        "Organization", "Skill", "Goal", "Memory"
    }
    
    @classmethod
    def validate_entity(cls, entity: Entity) -> bool:
        """
        Validate if an entity is properly formed.
        
        Args:
            entity: Entity to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not entity.name or not entity.type:
            return False
        
        if entity.type not in cls.VALID_ENTITY_TYPES:
            return False
        
        if entity.confidence < 0.3:  # Minimum confidence threshold
            return False
        
        return True
    
    @classmethod
    def extract_entities_from_text(cls, text: str) -> List[Entity]:
        """
        Simple rule-based entity extraction as fallback.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Always add User entity
        entities.append(Entity(
            name="User",
            type="Person",
            properties={},
            confidence=1.0
        ))
        
        # Simple patterns for common entities
        text_lower = text.lower()
        
        # Skills pattern
        skill_patterns = ["i know", "i can", "i'm good at", "skilled in"]
        for pattern in skill_patterns:
            if pattern in text_lower:
                # Extract skill name (simplified)
                skill_text = text_lower.split(pattern)[1].split(".")[0].strip()
                if skill_text:
                    entities.append(Entity(
                        name=skill_text.title(),
                        type="Skill",
                        properties={"category": "general"},
                        confidence=0.7
                    ))
        
        return entities
