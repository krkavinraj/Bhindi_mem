"""
Main Streamlit application for the Bhindi Knowledge Graph system.
Provides an interactive chat interface with real-time graph visualization.
"""
import streamlit as st
import logging
import json
from typing import Dict, Any, List
import time
from streamlit_agraph import agraph, Node, Edge, Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
try:
    from src.config import config
    from src.graph_manager import GraphManager
    from src.gpt_parser import GPTParser
    from src.crud_executor import CRUDExecutor
    from src.memory_retriever import MemoryRetriever
    from src.response_generator import ResponseGenerator
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🧠 Bhindi Knowledge Graph",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
    }
    
    .assistant-message {
        background-color: #e8f4fd;
        border-left-color: #4ecdc4;
    }
    
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class KnowledgeGraphApp:
    """Main application class for the Knowledge Graph system."""
    
    def __init__(self):
        """Initialize the application components."""
        self.initialize_session_state()
        self.initialize_components()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'graph_data' not in st.session_state:
            st.session_state.graph_data = {"nodes": [], "edges": []}
        
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        if 'graph_stats' not in st.session_state:
            st.session_state.graph_stats = {"nodes": 0, "relationships": 0}
        
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
    
    def initialize_components(self):
        """Initialize the core components."""
        if not st.session_state.initialized:
            try:
                # Validate configuration
                config.validate()
                
                # Initialize components and store in session state
                st.session_state.graph_manager = GraphManager()
                st.session_state.gpt_parser = GPTParser()
                st.session_state.crud_executor = CRUDExecutor(st.session_state.graph_manager)
                st.session_state.memory_retriever = MemoryRetriever(st.session_state.graph_manager)
                st.session_state.response_generator = ResponseGenerator()
                
                st.session_state.initialized = True
                logger.info("Application components initialized successfully")
                
            except Exception as e:
                st.error(f"Failed to initialize components: {e}")
                st.stop()
    
    @property
    def graph_manager(self):
        """Get graph manager from session state."""
        return st.session_state.get('graph_manager')
    
    @property
    def gpt_parser(self):
        """Get GPT parser from session state."""
        return st.session_state.get('gpt_parser')
    
    @property
    def crud_executor(self):
        """Get CRUD executor from session state."""
        return st.session_state.get('crud_executor')
    
    @property
    def memory_retriever(self):
        """Get memory retriever from session state."""
        return st.session_state.get('memory_retriever')
    
    @property
    def response_generator(self):
        """Get response generator from session state."""
        return st.session_state.get('response_generator')
    
    def render_header(self):
        """Render the application header."""
        st.markdown("""
        <div class="main-header">
            <h1>🧠 Bhindi Knowledge Graph</h1>
            <p>Your Personal AI-Powered Memory System</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with controls and statistics."""
        with st.sidebar:
            st.header("📊 Graph Statistics")
            
            # Update graph statistics
            try:
                stats = self.graph_manager.get_graph_statistics()
                st.session_state.graph_stats = stats
            except Exception as e:
                st.error(f"Error getting stats: {e}")
                stats = {"nodes": 0, "relationships": 0, "node_types": {}, "relationship_types": {}}
            
            # Display main statistics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="metric-value">{stats.get('nodes', 0)}</div>
                    <div class="metric-label">Nodes</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="metric-value">{stats.get('relationships', 0)}</div>
                    <div class="metric-label">Relationships</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Node types breakdown
            if stats.get('node_types'):
                st.subheader("🏷️ Node Types")
                for node_type, count in stats['node_types'].items():
                    st.write(f"**{node_type}**: {count}")
            
            # Relationship types breakdown
            if stats.get('relationship_types'):
                st.subheader("🔗 Relationship Types")
                for rel_type, count in list(stats['relationship_types'].items())[:5]:
                    st.write(f"**{rel_type}**: {count}")
            
            st.divider()
            
            # Controls
            st.subheader("⚙️ Controls")
            
            if st.button("🔄 Refresh Graph", use_container_width=True):
                self.update_graph_data()
                st.rerun()
            
            if st.button("📈 Show Graph Summary", use_container_width=True):
                self.show_graph_summary()
            
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_history = []
                st.rerun()
            
            # Configuration info
            st.subheader("🔧 Configuration")
            st.write(f"**Model**: {config.azure_openai.model}")
            st.write(f"**Database**: Connected" if self.graph_manager.driver else "Disconnected")
    
    def render_chat_interface(self):
        """Render the main chat interface."""
        st.header("💬 Chat with your Knowledge Graph")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Tell me about yourself, ask questions, or share experiences..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Process the message and generate response
            with st.chat_message("assistant"):
                with st.spinner("Processing your message..."):
                    response = self.process_user_message(prompt)
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Update graph data
            self.update_graph_data()
            
            # Rerun to update the interface
            st.rerun()
    
    def process_user_message(self, user_input: str) -> str:
        """
        Process user message through the complete pipeline.
        
        Args:
            user_input: User's message
            
        Returns:
            Generated response
        """
        try:
            # Get context from memory retriever
            context = self.memory_retriever.retrieve_relevant_context(user_input)
            
            # Parse the conversation
            conversation_context = self.memory_retriever.get_conversation_context(
                st.session_state.conversation_history
            )
            
            parsed = self.gpt_parser.parse_conversation(user_input, conversation_context)
            
            # Execute CRUD operations
            crud_result = self.crud_executor.execute_parsed_conversation(parsed)
            
            # Generate natural language response
            response = self.response_generator.generate_response(
                user_input, crud_result, context, parsed
            )
            
            # Add to conversation history
            st.session_state.conversation_history.append(user_input)
            
            # Keep only last 10 conversations for context
            if len(st.session_state.conversation_history) > 10:
                st.session_state.conversation_history = st.session_state.conversation_history[-10:]
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"I encountered an error processing your message: {str(e)}"
    
    def update_graph_data(self):
        """Update the graph data for visualization."""
        try:
            graph_data = self.graph_manager.get_graph_data_for_visualization()
            st.session_state.graph_data = graph_data
        except Exception as e:
            logger.error(f"Error updating graph data: {e}")
    
    def show_graph_summary(self):
        """Show a summary of the current graph."""
        try:
            stats = self.graph_manager.get_graph_statistics()
            summary = self.response_generator.generate_graph_summary(stats)
            
            st.info(f"📊 **Graph Summary**: {summary}")
            
        except Exception as e:
            st.error(f"Error generating summary: {e}")
    
    def render_graph_visualization(self):
        """Render the interactive graph visualization."""
        st.header("🌐 Knowledge Graph Visualization")
        
        # Update graph data in real-time
        self.update_graph_data()
        
        # Check if we have graph data
        if not st.session_state.graph_data.get("nodes"):
            st.info("No graph data to display yet. Start a conversation to build your knowledge graph!")
            return
        
        nodes_data = st.session_state.graph_data.get("nodes", [])
        edges_data = st.session_state.graph_data.get("edges", [])
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Nodes", len(nodes_data))
        
        with col2:
            st.metric("Total Edges", len(edges_data))
        
        with col3:
            if nodes_data:
                node_types = {}
                for node in nodes_data:
                    node_type = node.get("type", "Unknown")
                    node_types[node_type] = node_types.get(node_type, 0) + 1
                st.metric("Node Types", len(node_types))
        
        # Create interactive graph
        if nodes_data:
            # Convert data to streamlit-agraph format
            nodes = []
            edges = []
            
            # Define colors for different node types
            node_colors = {
                "Person": "#FF6B6B",
                "Concept": "#4ECDC4", 
                "Event": "#45B7D1",
                "Skill": "#96CEB4",
                "Preference": "#FFEAA7",
                "Location": "#DDA0DD",
                "Organization": "#98D8C8",
                "Default": "#95A5A6"
            }
            
            # Create nodes
            for node_data in nodes_data:
                node_id = str(node_data.get("id", node_data.get("label", "unknown")))
                node_label = node_data.get("label", node_id)
                node_type = node_data.get("type", "Default")
                node_color = node_colors.get(node_type, node_colors["Default"])
                
                # Create tooltip with node properties
                properties = node_data.get("properties", {})
                title_parts = [f"Type: {node_type}"]
                for key, value in properties.items():
                    if key not in ['id', 'label', 'type'] and value:
                        title_parts.append(f"{key}: {value}")
                title = "\n".join(title_parts)
                
                nodes.append(Node(
                    id=node_id,
                    label=node_label,
                    size=25,
                    color=node_color,
                    title=title,
                    font={"color": "white", "size": 12}
                ))
            
            # Create edges
            for edge_data in edges_data:
                source = str(edge_data.get("source", ""))
                target = str(edge_data.get("target", ""))
                edge_type = edge_data.get("type", "RELATED")
                
                if source and target:
                    edges.append(Edge(
                        source=source,
                        target=target,
                        label=edge_type,
                        color="#888888",
                        width=2
                    ))
            
            # Configure the graph
            config = Config(
                width=800,
                height=600,
                directed=True,
                physics=True,
                hierarchical=False,
                nodeHighlightBehavior=True,
                highlightColor="#F7A7A6",
                collapsible=False,
                node={
                    "labelProperty": "label",
                    "renderLabel": True
                },
                link={
                    "labelProperty": "label",
                    "renderLabel": True
                },
                maxZoom=2,
                minZoom=0.1,
                initialZoom=1,
                d3={
                    "alphaTarget": 0.03,
                    "gravity": -300,
                    "linkDistance": 100,
                    "linkStrength": 1,
                    "disableLinkForce": False
                }
            )
            
            # Render the interactive graph
            if nodes:
                selected_node = agraph(nodes=nodes, edges=edges, config=config)
                
                # Show selected node details
                if selected_node:
                    st.subheader("Selected Node Details")
                    selected_data = next((n for n in nodes_data if str(n.get("id", n.get("label"))) == selected_node), None)
                    if selected_data:
                        st.json(selected_data)
            else:
                st.warning("No valid nodes found for visualization")
        
        # Auto-refresh option
        if st.sidebar.checkbox("Auto-refresh Graph", value=True):
            time.sleep(1)
            st.rerun()
    
    def run(self):
        """Run the main application."""
        self.render_header()
        
        # Create main layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.render_chat_interface()
        
        with col2:
            self.render_graph_visualization()
        
        # Render sidebar
        self.render_sidebar()

def main():
    """Main function to run the Streamlit app."""
    try:
        app = KnowledgeGraphApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()
