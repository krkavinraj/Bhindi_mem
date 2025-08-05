"""
Natural language response generator using Azure GPT-4o.
Generates human-like responses based on CRUD operation results and graph context.
"""
import logging
from typing import Dict, Any, List, Optional
from openai import AzureOpenAI
from .config import config
from .gpt_parser import ParsedConversation

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Generates natural language responses using Azure GPT-4o."""
    
    def __init__(self):
        """Initialize the response generator with Azure OpenAI client."""
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
    
    def generate_response(
        self,
        user_input: str,
        crud_result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        parsed_conversation: Optional[ParsedConversation] = None
    ) -> str:
        """
        Generate a natural language response based on CRUD operation results.
        
        Args:
            user_input: Original user input
            crud_result: Result from CRUD operation execution
            context: Additional context from memory retriever
            parsed_conversation: Parsed conversation data
            
        Returns:
            Natural language response string
        """
        try:
            # Create system prompt for response generation
            system_prompt = self._create_response_system_prompt()
            
            # Create user prompt with all context
            user_prompt = self._create_response_user_prompt(
                user_input, crud_result, context, parsed_conversation
            )
            
            # Generate response using Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=min(config.azure_openai.max_tokens, 500),  # Limit for responses
                temperature=0.7,  # Slightly creative but consistent
            )
            
            generated_response = response.choices[0].message.content.strip()
            
            # Add fallback if response is empty
            if not generated_response:
                generated_response = self._generate_fallback_response(crud_result)
            
            return generated_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_fallback_response(crud_result)
    
    def _create_response_system_prompt(self) -> str:
        """Create system prompt for response generation."""
        return """You are a helpful AI assistant that manages a personal knowledge graph. 
        Your role is to provide natural, conversational responses about knowledge graph operations.

        Guidelines for responses:
        1. Be conversational and friendly
        2. Acknowledge what the user shared or asked
        3. Summarize what was added/found/updated/deleted in the knowledge graph
        4. Be specific about entities and relationships when relevant
        5. Ask follow-up questions when appropriate
        6. Keep responses concise but informative
        7. Use a warm, personal tone
        8. If errors occurred, explain them helpfully

        Response style examples:
        - "I've added that you're skilled in guitar playing! I can see you've been learning for 3 years."
        - "I found 5 skills in your knowledge graph: Python, Guitar, Cooking, Photography, and Running."
        - "I've updated your job information. You're now working at Google as a software engineer."
        - "I couldn't find any information about that topic in your knowledge graph yet."

        Always be helpful and encouraging about building their personal knowledge graph."""

    def _create_response_user_prompt(
        self,
        user_input: str,
        crud_result: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        parsed_conversation: Optional[ParsedConversation]
    ) -> str:
        """Create user prompt with all context for response generation."""
        prompt_parts = []
        
        # Add user input
        prompt_parts.append(f"User said: \"{user_input}\"")
        
        # Add CRUD operation result
        prompt_parts.append(f"\nOperation performed: {crud_result.get('intent', 'unknown')}")
        prompt_parts.append(f"Success: {crud_result.get('success', False)}")
        prompt_parts.append(f"Message: {crud_result.get('message', '')}")
        
        if crud_result.get('entities_processed', 0) > 0:
            prompt_parts.append(f"Entities processed: {crud_result['entities_processed']}")
        
        if crud_result.get('relationships_processed', 0) > 0:
            prompt_parts.append(f"Relationships processed: {crud_result['relationships_processed']}")
        
        # Add data if available
        if crud_result.get('data'):
            prompt_parts.append(f"Data returned: {self._format_data_for_prompt(crud_result['data'])}")
        
        # Add parsed entities and relationships
        if parsed_conversation:
            if parsed_conversation.entities:
                entity_names = [e.name for e in parsed_conversation.entities if e.name.lower() != 'user']
                if entity_names:
                    prompt_parts.append(f"Entities mentioned: {', '.join(entity_names)}")
            
            if parsed_conversation.relationships:
                rel_descriptions = []
                for rel in parsed_conversation.relationships:
                    rel_descriptions.append(f"{rel.from_entity} {rel.type} {rel.to_entity}")
                if rel_descriptions:
                    prompt_parts.append(f"Relationships: {', '.join(rel_descriptions)}")
        
        # Add relevant context
        if context:
            if context.get('user_profile'):
                profile = context['user_profile']
                if profile.get('skills'):
                    prompt_parts.append(f"User's known skills: {len(profile['skills'])} skills")
                if profile.get('preferences'):
                    prompt_parts.append(f"User's preferences: {len(profile['preferences'])} preferences")
        
        prompt_parts.append("\nGenerate a natural, conversational response to the user:")
        
        return "\n".join(prompt_parts)
    
    def _format_data_for_prompt(self, data: Any) -> str:
        """Format data for inclusion in prompt."""
        if isinstance(data, dict):
            if 'nodes' in data and 'relationships' in data:
                # Graph statistics
                return f"{data.get('nodes', 0)} nodes, {data.get('relationships', 0)} relationships"
            else:
                # Single node or other dict
                return str(data)[:200]  # Limit length
        elif isinstance(data, list):
            if len(data) == 0:
                return "No results found"
            elif len(data) == 1:
                return f"1 result: {str(data[0])[:100]}"
            else:
                return f"{len(data)} results found"
        else:
            return str(data)[:200]
    
    def _generate_fallback_response(self, crud_result: Dict[str, Any]) -> str:
        """Generate a fallback response when GPT generation fails."""
        intent = crud_result.get('intent', 'unknown')
        success = crud_result.get('success', False)
        
        if not success:
            return f"I had trouble processing that. {crud_result.get('message', '')}"
        
        if intent == "CREATE":
            entities = crud_result.get('entities_processed', 0)
            relationships = crud_result.get('relationships_processed', 0)
            return f"I've added {entities} new entities and {relationships} relationships to your knowledge graph!"
        
        elif intent in ["READ", "QUERY"]:
            data = crud_result.get('data')
            if isinstance(data, list):
                return f"I found {len(data)} results for your query."
            elif isinstance(data, dict) and 'nodes' in data:
                return f"Your knowledge graph has {data.get('nodes', 0)} nodes and {data.get('relationships', 0)} relationships."
            else:
                return "I found some information for you!"
        
        elif intent == "UPDATE":
            entities = crud_result.get('entities_processed', 0)
            return f"I've updated {entities} entities in your knowledge graph."
        
        elif intent == "DELETE":
            entities = crud_result.get('entities_processed', 0)
            return f"I've removed {entities} entities from your knowledge graph."
        
        return "I've processed your request successfully!"
    
    def generate_graph_summary(self, graph_stats: Dict[str, Any]) -> str:
        """
        Generate a summary of the current knowledge graph state.
        
        Args:
            graph_stats: Graph statistics dictionary
            
        Returns:
            Natural language summary
        """
        try:
            system_prompt = """You are summarizing a personal knowledge graph. 
            Create a brief, friendly summary of what's in the graph.
            Focus on the most interesting aspects and encourage the user."""
            
            user_prompt = f"""Graph Statistics:
            - Total nodes: {graph_stats.get('nodes', 0)}
            - Total relationships: {graph_stats.get('relationships', 0)}
            - Node types: {graph_stats.get('node_types', {})}
            - Relationship types: {graph_stats.get('relationship_types', {})}
            
            Create a friendly summary of this knowledge graph:"""
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating graph summary: {e}")
            nodes = graph_stats.get('nodes', 0)
            relationships = graph_stats.get('relationships', 0)
            return f"Your knowledge graph contains {nodes} entities and {relationships} connections!"
    
    def generate_suggestions(self, context: Dict[str, Any]) -> List[str]:
        """
        Generate conversation suggestions based on current context.
        
        Args:
            context: Current graph context
            
        Returns:
            List of suggested conversation starters
        """
        suggestions = []
        
        try:
            # Get user profile from context
            user_profile = context.get('user_profile', {})
            
            # Suggest based on what's missing or could be expanded
            if not user_profile.get('skills'):
                suggestions.append("Tell me about your skills or hobbies")
            
            if not user_profile.get('goals'):
                suggestions.append("What are your current goals or aspirations?")
            
            if not user_profile.get('preferences'):
                suggestions.append("What do you like or dislike?")
            
            if not user_profile.get('organizations'):
                suggestions.append("Where do you work or study?")
            
            # Add some general suggestions
            suggestions.extend([
                "Ask me about your knowledge graph",
                "Tell me about a recent experience",
                "What would you like to learn?"
            ])
            
            return suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return [
                "Tell me about yourself",
                "What are your interests?",
                "Show me my knowledge graph",
                "What do you know about me?"
            ]
