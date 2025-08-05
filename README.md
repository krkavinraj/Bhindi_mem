# 🧠 Bhindi Knowledge Graph

An intelligent memory graph system that uses GPT-4o to understand conversations, extract entities/relationships, and perform intelligent CRUD operations on a Neo4j knowledge graph with real-time visualization.

## 🌟 Features

- **Azure GPT-4o Powered NLP**: Advanced conversation parsing and entity extraction
- **Real-time Graph Visualization**: Interactive network graph that updates as you chat
- **Color-coded Nodes**: Different colors for Person, Concept, Event, Preference, etc.
- **Interactive Elements**: Click nodes to see details, drag to rearrange
- **Auto-refresh**: Graph updates automatically after each conversation
- **Live Statistics**: Real-time counts of different node types and relationships
- **Natural Language Interface**: Perform CRUD operations through conversation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Neo4j Database
- Azure OpenAI Service with GPT-4o deployment

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd bhindi-knowledge-graph
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

4. Start Neo4j database (Docker):
```bash
docker run \
    --name neo4j \
    -p7474:7474 -p7687:7687 \
    -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -v $HOME/neo4j/import:/var/lib/neo4j/import \
    -v $HOME/neo4j/plugins:/plugins \
    --env NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

5. Run the application:
```bash
streamlit run streamlit_app.py
```

## 📁 Project Structure

```
bhindi-knowledge-graph/
├── src/
│   ├── config.py              # Configuration management
│   ├── graph_manager.py       # Neo4j operations
│   ├── gpt_parser.py          # GPT-4o conversation parsing
│   ├── crud_executor.py       # CRUD operation execution
│   ├── memory_retriever.py    # Contextual graph querying
│   └── response_generator.py  # Natural language responses
├── data/
│   ├── schemas/               # Graph schemas
│   └── examples/              # Sample data
├── tests/                     # Test files
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Dependencies
└── .env.example             # Environment template
```

## 🎯 Usage

1. **Start a Conversation**: Type any message about yourself, your interests, or experiences
2. **Watch the Graph Grow**: See entities and relationships appear in real-time
3. **Explore Connections**: Click on nodes to see details and connections
4. **Ask Questions**: Query your knowledge graph using natural language

### Example Conversations

- "I love playing guitar and I've been learning it for 3 years"
- "I work at Google as a software engineer"
- "I want to learn machine learning by the end of this year"
- "Show me all my skills"
- "What do I like?"

## 🔧 Configuration

Edit `.env` file to configure:

- **Neo4j**: Database connection settings
- **Azure OpenAI**: API key, endpoint, and deployment settings
- **App**: Visualization and behavior settings

## 🧪 Testing

Run tests:
```bash
python -m pytest tests/
```

## 📊 Graph Schema

The system supports various node types:
- **Person**: People and their attributes
- **Concept**: Abstract ideas and topics
- **Event**: Occurrences and experiences
- **Preference**: Likes, dislikes, and preferences
- **Location**: Places and geographical entities
- **Organization**: Companies, institutions
- **Skill**: Abilities and competencies
- **Goal**: Objectives and aspirations
- **Memory**: Specific memories and experiences

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

For issues and questions, please open a GitHub issue or contact the development team.
