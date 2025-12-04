import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

interface Message {
  id: number;
  text: string;
  sender: 'user' | 'assistant';
  timestamp: string;
}

interface ChatResponse {
  response: string;
  user_id: string;
  session_id: string;
  timestamp: string;
  status: string;
  metadata: any;
}

interface MemoryData {
  tier1_short_term: any;
  tier2_episodic: any[];
  tier3_semantic: any[];
  tier4_procedural: any[];
  total_entities: number;
  total_relationships: number;
}

interface GraphNode {
  id: string;
  name: string;
  type: string;
  x?: number;
  y?: number;
  color?: string;
}

interface GraphEdge {
  from: string;
  to: string;
  relationship: string;
  color?: string;
}

// Transform semantic memory data into graph format
const transformToGraphData = (semanticData: any): { nodes: GraphNode[]; edges: GraphEdge[] } => {
  console.log('Semantic data received:', semanticData); // Debug log
  
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const nodeMap = new Set<string>();
  
  // Color mapping for different entity types
  const typeColors: { [key: string]: string } = {
    'person': '#ff6b6b',
    'people': '#ff6b6b',
    'user': '#ff6b6b',
    'place': '#4ecdc4',
    'location': '#4ecdc4', 
    'concept': '#45b7d1',
    'event': '#96ceb4',
    'object': '#feca57',
    'thing': '#feca57',
    'unknown': '#ddd',
    'default': '#ddd'
  };

  // Handle different data structures
  let entities = [];
  let relationships = [];

  if (Array.isArray(semanticData)) {
    // Old format: array of entities only
    entities = semanticData;
    console.log('Old format: entity array only');
  } else if (semanticData && typeof semanticData === 'object') {
    // New format: object with entities and relationships
    entities = semanticData.entities || [];
    relationships = semanticData.relationships || [];
    console.log(`New format: ${entities.length} entities, ${relationships.length} relationships`);
  }

  // Debug: Log what we actually have
  console.log('Entities found:', entities.length);
  console.log('Relationships found:', relationships.length);
  
  // If we have entities but no relationships, create some basic connections for visualization
  if (entities.length > 0 && relationships.length === 0) {
    console.log('Found entities but no relationships - creating basic connections for visualization');
    
    // Create simple relationships between some entities for demo
    for (let i = 0; i < entities.length - 1; i++) {
      if (Math.random() > 0.7) { // 30% chance of connection
        relationships.push({
          from_entity: entities[i].name,
          to_entity: entities[i + 1].name,
          relationship_type: 'RELATED_TO',
          weight: 0.5
        });
      }
    }
    
    console.log(`Created ${relationships.length} basic relationships for visualization`);
  }
  
  if (entities.length === 0 && relationships.length === 0) {
    console.log('No entities or relationships found - empty graph');
    return { nodes: [], edges: [] };
  }

  // Create a better circular layout for entities
  const centerX = 250;
  const centerY = 150;
  const radius = Math.min(100, Math.max(50, entities.length * 8));

  // Process entities first
  entities.forEach((item: any, index: number) => {
    console.log(`Processing entity ${index}:`, item);
    
    if (item.name && !nodeMap.has(item.name)) {
      // Calculate position in a circle for better layout
      const angle = (index / entities.length) * 2 * Math.PI;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      
      nodes.push({
        id: item.name,
        name: item.name,
        type: item.type || 'unknown',
        color: typeColors[item.type?.toLowerCase()] || typeColors.default,
        x: x,
        y: y
      });
      nodeMap.add(item.name);
    }

    // Check if entity has embedded relationship data (old format compatibility)
    const relationship = item.relationship || item.relation || item.edge_type || item.predicate;
    const relatedEntity = item.related_entity || item.target || item.object || item.to;
    const relatedType = item.related_type || item.target_type || item.object_type;
    
    if (relationship && relatedEntity) {
      console.log(`Found embedded relationship: ${item.name} --[${relationship}]--> ${relatedEntity}`);
      
      // Add related entity node if it doesn't exist
      if (!nodeMap.has(relatedEntity)) {
        const angle = Math.random() * 2 * Math.PI;
        const x = centerX + (radius + 60) * Math.cos(angle);
        const y = centerY + (radius + 60) * Math.sin(angle);
        
        nodes.push({
          id: relatedEntity,
          name: relatedEntity,
          type: relatedType || 'unknown',
          color: typeColors[relatedType?.toLowerCase()] || typeColors.default,
          x: x,
          y: y
        });
        nodeMap.add(relatedEntity);
      }

      edges.push({
        from: item.name,
        to: relatedEntity,
        relationship: relationship,
        color: '#999'
      });
    }
  });

  // Process separate relationships array (new format)
  relationships.forEach((rel: any, index: number) => {
    console.log(`Processing relationship ${index}:`, rel);
    
    const from = rel.from_entity || rel.from || rel.source || rel.subject;
    const to = rel.to_entity || rel.to || rel.target || rel.object;
    const relationship = rel.relationship_type || rel.type || rel.predicate || rel.relationship || rel.relation;
    
    if (from && to && relationship) {
      console.log(`Found relationship: ${from} --[${relationship}]--> ${to}`);
      
      // Make sure both nodes exist
      [from, to].forEach(entityName => {
        if (!nodeMap.has(entityName)) {
          const angle = Math.random() * 2 * Math.PI;
          const x = centerX + radius * Math.cos(angle);
          const y = centerY + radius * Math.sin(angle);
          
          nodes.push({
            id: entityName,
            name: entityName,
            type: 'unknown',
            color: typeColors.default,
            x: x,
            y: y
          });
          nodeMap.add(entityName);
        }
      });

      edges.push({
        from: from,
        to: to,
        relationship: relationship,
        color: '#999'
      });
    }
  });

  console.log(`Final graph data: ${nodes.length} nodes, ${edges.length} edges`);
  console.log('Nodes:', nodes);
  console.log('Edges:', edges);
  return { nodes, edges };
};

// Custom SVG Graph Component
const InteractiveGraph: React.FC<{ 
  nodes: GraphNode[]; 
  edges: GraphEdge[]; 
}> = ({ nodes, edges }) => {
  const [draggedNode, setDraggedNode] = useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [nodePositions, setNodePositions] = useState<{ [id: string]: { x: number; y: number } }>({});

  // Initialize node positions
  useEffect(() => {
    const positions: { [id: string]: { x: number; y: number } } = {};
    nodes.forEach(node => {
      positions[node.id] = { x: node.x || 0, y: node.y || 0 };
    });
    setNodePositions(positions);
  }, [nodes]);

  const handleMouseDown = (nodeId: string) => {
    setDraggedNode(nodeId);
  };

  const handleMouseMove = (event: React.MouseEvent<SVGElement>) => {
    if (draggedNode) {
      const rect = event.currentTarget.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      
      setNodePositions(prev => ({
        ...prev,
        [draggedNode]: { x, y }
      }));
    }
  };

  const handleMouseUp = () => {
    setDraggedNode(null);
  };

  const handleNodeClick = (nodeId: string) => {
    console.log('Clicked node:', nodeId);
  };

  // Find node position by ID
  const getNodePosition = (nodeId: string) => {
    return nodePositions[nodeId] || { x: 0, y: 0 };
  };

  return (
    <div style={{ width: '100%', height: '300px', border: '1px solid #e0e0e0', borderRadius: '8px', backgroundColor: '#fafafa' }}>
      <svg
        width="100%"
        height="100%"
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: draggedNode ? 'grabbing' : 'default' }}
      >
        {/* Render edges first (behind nodes) */}
        {edges.map((edge, index) => {
          const fromPos = getNodePosition(edge.from);
          const toPos = getNodePosition(edge.to);
          
          // Skip edge if nodes don't exist or are at origin
          if (!fromPos || !toPos || (fromPos.x === 0 && fromPos.y === 0) || (toPos.x === 0 && toPos.y === 0)) {
            return null;
          }
          
          return (
            <g key={`edge-${index}`}>
              <line
                x1={fromPos.x}
                y1={fromPos.y}
                x2={toPos.x}
                y2={toPos.y}
                stroke={edge.color || '#999'}
                strokeWidth="2"
                markerEnd="url(#arrowhead)"
                opacity="0.8"
              />
              {/* Edge label */}
              <text
                x={(fromPos.x + toPos.x) / 2}
                y={(fromPos.y + toPos.y) / 2 - 5}
                fill="#666"
                fontSize="10"
                fontWeight="500"
                textAnchor="middle"
                style={{ pointerEvents: 'none', userSelect: 'none' }}
              >
                {edge.relationship}
              </text>
            </g>
          );
        })}

        {/* Arrow marker definition */}
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon
              points="0 0, 10 3.5, 0 7"
              fill="#999"
            />
          </marker>
        </defs>

        {/* Render nodes */}
        {nodes.map((node) => {
          const pos = getNodePosition(node.id);
          const isHovered = hoveredNode === node.id;
          const isDragged = draggedNode === node.id;
          
          return (
            <g key={node.id}>
              <circle
                cx={pos.x}
                cy={pos.y}
                r={isHovered || isDragged ? "18" : "15"}
                fill={node.color}
                stroke="#fff"
                strokeWidth="2"
                style={{ 
                  cursor: 'grab',
                  transition: isDragged ? 'none' : 'all 0.2s ease',
                  filter: isHovered ? 'brightness(1.1)' : 'none'
                }}
                onMouseDown={() => handleMouseDown(node.id)}
                onMouseEnter={() => setHoveredNode(node.id)}
                onMouseLeave={() => setHoveredNode(null)}
                onClick={() => handleNodeClick(node.id)}
              />
              {/* Node label */}
              <text
                x={pos.x}
                y={pos.y + 25}
                fill="#333"
                fontSize="11"
                fontWeight="500"
                textAnchor="middle"
                style={{ pointerEvents: 'none', userSelect: 'none' }}
              >
                {node.name}
              </text>
              {/* Node type */}
              <text
                x={pos.x}
                y={pos.y + 37}
                fill="#666"
                fontSize="9"
                textAnchor="middle"
                style={{ pointerEvents: 'none', userSelect: 'none' }}
              >
                ({node.type})
              </text>
              
              {/* Hover tooltip */}
              {isHovered && (
                <g>
                  <rect
                    x={pos.x - 40}
                    y={pos.y - 35}
                    width="80"
                    height="20"
                    fill="rgba(0,0,0,0.8)"
                    rx="4"
                    style={{ pointerEvents: 'none' }}
                  />
                  <text
                    x={pos.x}
                    y={pos.y - 20}
                    fill="white"
                    fontSize="10"
                    textAnchor="middle"
                    style={{ pointerEvents: 'none' }}
                  >
                    {node.name}
                  </text>
                </g>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
};

const MemoryDashboard: React.FC<{ 
  isOpen: boolean; 
  onClose: () => void; 
  userId: string;
  memoryData: MemoryData | null;
  onRefresh: () => void;
}> = ({ isOpen, onClose, userId, memoryData, onRefresh }) => {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  if (!isOpen) return null;
  
  // Transform semantic data for graph visualization
  const graphData = memoryData?.tier3_semantic ? transformToGraphData(memoryData.tier3_semantic) : { nodes: [], edges: [] };

  const handleDeleteConfirm = async () => {
    setIsDeleting(true);
    try {
      const response = await axios.delete(`/api/memory/user/${userId}`);
      console.log('Delete response:', response.data);
      
      if (response.data.success) {
        alert(`Memory deletion completed!\n\nDeleted ${response.data.total_deleted} items from ${response.data.tiers_cleared}/${response.data.total_tiers} memory tiers.`);
        onRefresh(); // Refresh the dashboard to show empty state
      } else {
        alert(`Deletion failed: ${response.data.message || response.data.error}`);
      }
    } catch (error) {
      console.error('Error deleting memory:', error);
      alert('Failed to delete memory data. Please try again.');
    } finally {
      setIsDeleting(false);
      setShowDeleteConfirm(false);
    }
  };

  const handleDeleteCancel = () => {
    setShowDeleteConfirm(false);
  };

  return (
    <div className="memory-modal">
      <div className="memory-dashboard">
        <div className="memory-header">
          <h2>🧠 Memory Dashboard - {userId}</h2>
          <div className="memory-controls">
            <button 
              onClick={() => setShowDeleteConfirm(true)} 
              className="delete-btn" 
              title="Delete All Memory Data"
              disabled={isDeleting}
            >
              {isDeleting ? '⏳' : '🗑️'}
            </button>
            <button onClick={onRefresh} className="refresh-btn" title="Refresh Memory">
              🔄
            </button>
            <button onClick={onClose} className="close-btn" title="Close">
              ✕
            </button>
          </div>
        </div>
        
        <div className="memory-content">
          {!memoryData ? (
            <div className="memory-loading">
              <div className="loading-spinner">⏳</div>
              <p>Loading memory data...</p>
            </div>
          ) : (
            <div className="memory-tiers">
              {/* Tier 1: Short-term Memory (Redis) */}
              <div className="memory-tier tier-1">
                <div className="tier-header">
                  <h3>🔥 Tier 1: Short-term Memory</h3>
                  <span className="tier-tech">Redis</span>
                </div>
                <div className="tier-content">
                  <div className="memory-stats">
                    <div className="stat">
                      <span className="stat-label">Session Data:</span>
                      <span className="stat-value">{memoryData.tier1_short_term?.active ? 'Active' : 'Inactive'}</span>
                    </div>
                    <div className="stat">
                      <span className="stat-label">Current Context:</span>
                      <span className="stat-value">{memoryData.tier1_short_term?.context_loaded ? 'Loaded' : 'Empty'}</span>
                    </div>
                  </div>
                  <p className="tier-description">
                    Temporary session data, conversation context, and immediate working memory.
                  </p>
                </div>
              </div>

              {/* Tier 2: Episodic Memory (PostgreSQL + pgvector) */}
              <div className="memory-tier tier-2">
                <div className="tier-header">
                  <h3>📖 Tier 2: Episodic Memory</h3>
                  <span className="tier-tech">PostgreSQL + pgvector</span>
                </div>
                <div className="tier-content">
                  <div className="memory-stats">
                    <div className="stat">
                      <span className="stat-label">Episodes:</span>
                      <span className="stat-value">{memoryData.tier2_episodic?.length || 0}</span>
                    </div>
                    <div className="stat">
                      <span className="stat-label">Vector Embeddings:</span>
                      <span className="stat-value">Stored</span>
                    </div>
                  </div>
                  <p className="tier-description">
                    Conversation history with semantic embeddings for context retrieval.
                  </p>
                  {memoryData.tier2_episodic?.length > 0 && (
                    <div className="memory-items">
                      {memoryData.tier2_episodic.slice(0, 3).map((episode: any, idx: number) => (
                        <div key={idx} className="memory-item episodic">
                          <div className="item-timestamp">{new Date(episode.timestamp).toLocaleString()}</div>
                          <div className="item-content">{episode.content?.substring(0, 100)}...</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Tier 3: Semantic Memory (Neo4j) */}
              <div className="memory-tier tier-3">
                <div className="tier-header">
                  <h3>🕸️ Tier 3: Semantic Memory</h3>
                  <span className="tier-tech">Neo4j Knowledge Graph</span>
                </div>
                <div className="tier-content">
                  <div className="memory-stats">
                    <div className="stat">
                      <span className="stat-label">Entities:</span>
                      <span className="stat-value">{memoryData.total_entities || 0}</span>
                    </div>
                    <div className="stat">
                      <span className="stat-label">Relationships:</span>
                      <span className="stat-value">{memoryData.total_relationships || 0}</span>
                    </div>
                  </div>
                  <p className="tier-description">
                    Knowledge graph of entities, concepts, and relationships extracted from conversations.
                  </p>
                  <div className="graph-container">
                    <InteractiveGraph 
                      nodes={graphData.nodes} 
                      edges={graphData.edges} 
                    />
                    <div className="graph-info">
                      <div style={{ fontSize: '12px', color: '#666', marginBottom: '8px' }}>
                        Graph: {graphData.nodes.length} nodes, {graphData.edges.length} edges
                      </div>
                      <div className="graph-legend">
                        <div className="legend-item"><span className="legend-color" style={{backgroundColor: '#ff6b6b'}}></span>Person</div>
                        <div className="legend-item"><span className="legend-color" style={{backgroundColor: '#4ecdc4'}}></span>Place</div>
                        <div className="legend-item"><span className="legend-color" style={{backgroundColor: '#45b7d1'}}></span>Concept</div>
                        <div className="legend-item"><span className="legend-color" style={{backgroundColor: '#96ceb4'}}></span>Event</div>
                        <div className="legend-item"><span className="legend-color" style={{backgroundColor: '#feca57'}}></span>Object</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Tier 4: Procedural Memory (PostgreSQL) */}
              <div className="memory-tier tier-4">
                <div className="tier-header">
                  <h3>⚙️ Tier 4: Procedural Memory</h3>
                  <span className="tier-tech">PostgreSQL Rules</span>
                </div>
                <div className="tier-content">
                  <div className="memory-stats">
                    <div className="stat">
                      <span className="stat-label">Rules:</span>
                      <span className="stat-value">{memoryData.tier4_procedural?.length || 0}</span>
                    </div>
                    <div className="stat">
                      <span className="stat-label">Preferences:</span>
                      <span className="stat-value">Learned</span>
                    </div>
                  </div>
                  <p className="tier-description">
                    Behavioral instructions, user preferences, and learned patterns.
                  </p>
                  {memoryData.tier4_procedural?.length > 0 && (
                    <div className="memory-items">
                      {memoryData.tier4_procedural.slice(0, 3).map((rule: any, idx: number) => (
                        <div key={idx} className="memory-item procedural">
                          <div className="rule-title">{rule.title}</div>
                          <div className="rule-description">{rule.instruction}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Delete Confirmation Dialog */}
      {showDeleteConfirm && (
        <div className="delete-modal">
          <div className="delete-dialog">
            <div className="delete-header">
              <h3>⚠️ Delete All Memory Data</h3>
            </div>
            <div className="delete-content">
              <p><strong>This action cannot be undone!</strong></p>
              <p>This will permanently delete:</p>
              <ul>
                <li>🔥 <strong>Short-term memory</strong> (Redis sessions)</li>
                <li>📖 <strong>Episodic memory</strong> (Conversation history)</li>
                <li>🕸️ <strong>Semantic memory</strong> (Knowledge graph)</li>
                <li>⚙️ <strong>Procedural memory</strong> (Rules & preferences)</li>
                <li>💬 <strong>Active chat sessions</strong></li>
              </ul>
              <p>Are you sure you want to delete <strong>all</strong> memory data for user <code>{userId}</code>?</p>
            </div>
            <div className="delete-actions">
              <button 
                onClick={handleDeleteCancel} 
                className="delete-cancel-btn"
                disabled={isDeleting}
              >
                Cancel
              </button>
              <button 
                onClick={handleDeleteConfirm} 
                className="delete-confirm-btn"
                disabled={isDeleting}
              >
                {isDeleting ? 'Deleting...' : 'Delete Everything'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const UserIdInput: React.FC<{ onSubmit: (userId: string) => void }> = ({ onSubmit }) => {
  const [inputUserId, setInputUserId] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputUserId.trim()) {
      onSubmit(inputUserId.trim());
    }
  };

  return (
    <div className="user-id-modal">
      <div className="user-id-card">
        <h2>🔐 Welcome to Umbranet Governor</h2>
        <p>Enter your User ID to continue your conversation or start a new one.</p>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={inputUserId}
            onChange={(e) => setInputUserId(e.target.value)}
            placeholder="Enter your User ID (e.g., alex, john_doe, etc.)"
            className="user-id-input"
            autoFocus
          />
          <button 
            type="submit" 
            disabled={!inputUserId.trim()}
            className="user-id-submit"
          >
            Continue
          </button>
        </form>
        <div className="user-id-info">
          <p><strong>💡 Tip:</strong> Use the same User ID to continue previous conversations.</p>
          <p><strong>🧠 Memory:</strong> Your AI remembers everything across sessions!</p>
        </div>
      </div>
    </div>
  );
};

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [userId, setUserId] = useState('');
  const [sessionId] = useState(`session_${Date.now()}`);
  const [isUserIdSet, setIsUserIdSet] = useState(false);
  const [showUserIdInput, setShowUserIdInput] = useState(false);
  const [showMemoryDashboard, setShowMemoryDashboard] = useState(false);
  const [memoryData, setMemoryData] = useState<MemoryData | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Initialize user ID from localStorage or show input
  useEffect(() => {
    const savedUserId = localStorage.getItem('umbranet_user_id');
    if (savedUserId) {
      setUserId(savedUserId);
      setIsUserIdSet(true);
    } else {
      setShowUserIdInput(true);
    }
  }, []);

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ 
        behavior: 'smooth',
        block: 'end',
        inline: 'nearest'
      });
    }
  };

  useEffect(() => {
    // Small delay to ensure DOM is updated before scrolling
    const timer = setTimeout(() => {
      scrollToBottom();
    }, 100);
    
    return () => clearTimeout(timer);
  }, [messages]);

  // Function to clean AI response from system info
  const cleanAIResponse = (response: string): string => {
    // Remove system info section (everything after "---")
    const cleanedResponse = response.split('---')[0].trim();
    return cleanedResponse;
  };

  // Function to fetch memory data for dashboard
  const fetchMemoryData = async () => {
    if (!userId) return;
    
    try {
      // Fetch real memory data from all tiers
      const [tier1Response, tier2Response, tier3Response, tier4Response, statsResponse] = await Promise.all([
        axios.get(`/api/memory/redis/${userId}`),
        axios.get(`/api/memory/episodic/${userId}`),
        axios.get(`/api/memory/semantic/${userId}`),
        axios.get(`/api/memory/procedural/${userId}`),
        axios.get(`/api/memory/stats/${userId}`)
      ]);

      let semanticData = tier3Response.data;

      const realMemoryData: MemoryData = {
        tier1_short_term: tier1Response.data,
        tier2_episodic: tier2Response.data.episodes || [],
        tier3_semantic: semanticData,
        tier4_procedural: tier4Response.data.rules || [],
        total_entities: statsResponse.data.entity_count || 0,
        total_relationships: statsResponse.data.relationship_count || 0
      };
      
      setMemoryData(realMemoryData);
    } catch (error) {
      console.error('Error fetching memory data:', error);
      // Fallback to basic data if API fails
      const fallbackData: MemoryData = {
        tier1_short_term: { active: false },
        tier2_episodic: [],
        tier3_semantic: [],
        tier4_procedural: [],
        total_entities: 0,
        total_relationships: 0
      };
      setMemoryData(fallbackData);
    }
  };

  // Remove unused mock functions - now using real API endpoints in fetchMemoryData

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading || !isUserIdSet || !userId) return;

    const userMessage: Message = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    // Keep focus on input after sending message
    setTimeout(() => {
      if (inputRef.current) {
        inputRef.current.focus();
      }
    }, 100);

    try {
      const response = await axios.post<ChatResponse>('/api/chat', {
        message: inputValue,
        user_id: userId,
        session_id: sessionId,
      });

      const assistantMessage: Message = {
        id: Date.now() + 1,
        text: cleanAIResponse(response.data.response),
        sender: 'assistant',
        timestamp: response.data.timestamp,
      };

      setMessages(prev => [...prev, assistantMessage]);
      
      // Refocus input after receiving response
      setTimeout(() => {
        if (inputRef.current) {
          inputRef.current.focus();
        }
      }, 100);
    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorMessage: Message = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error processing your message. Please try again.',
        sender: 'assistant',
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, errorMessage]);
      
      // Refocus input after error
      setTimeout(() => {
        if (inputRef.current) {
          inputRef.current.focus();
        }
      }, 100);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const handleUserIdSubmit = (inputUserId: string) => {
    const trimmedUserId = inputUserId.trim();
    if (trimmedUserId) {
      setUserId(trimmedUserId);
      setIsUserIdSet(true);
      setShowUserIdInput(false);
      localStorage.setItem('umbranet_user_id', trimmedUserId);
    }
  };

  const handleChangeUser = () => {
    setShowUserIdInput(true);
    setIsUserIdSet(false);
    setMessages([]); // Clear messages when switching users
  };

  const handleLogout = () => {
    localStorage.removeItem('umbranet_user_id');
    setUserId('');
    setIsUserIdSet(false);
    setShowUserIdInput(true);
    setMessages([]);
    setShowMemoryDashboard(false);
    setMemoryData(null);
  };

  const openMemoryDashboard = () => {
    setShowMemoryDashboard(true);
    fetchMemoryData();
  };

  const closeMemoryDashboard = () => {
    setShowMemoryDashboard(false);
  };

  const refreshMemoryData = () => {
    fetchMemoryData();
  };

  return (
    <div className="app">
      {showUserIdInput && (
        <UserIdInput onSubmit={handleUserIdSubmit} />
      )}
      
      {showMemoryDashboard && (
        <MemoryDashboard 
          isOpen={showMemoryDashboard}
          onClose={closeMemoryDashboard}
          userId={userId}
          memoryData={memoryData}
          onRefresh={refreshMemoryData}
        />
      )}
      
      <header className="app-header">
        <div className="header-content">
          <div className="header-title">
            <h1>🧠 Umbranet Governor</h1>
            <p>Headless AI Operating System</p>
          </div>
          {isUserIdSet && (
            <div className="user-controls">
              <div className="current-user">
                <span className="user-label">👤 {userId}</span>
              </div>
              <div className="user-actions">
                <button onClick={openMemoryDashboard} className="memory-btn" title="Memory Dashboard">
                  🧠
                </button>
                <button onClick={handleChangeUser} className="change-user-btn" title="Switch User">
                  🔄
                </button>
                <button onClick={handleLogout} className="logout-btn" title="Logout">
                  🚪
                </button>
              </div>
            </div>
          )}
        </div>
      </header>

      {isUserIdSet && (
        <div className="chat-container">
          <div className="messages-container">
            {messages.length === 0 && (
              <div className="welcome-message">
                <h3>Welcome back, {userId}!</h3>
                <p>Your personal AI assistant with persistent memory across conversations.</p>
                <p>I remember our previous conversations. Start typing to continue...</p>
              </div>
            )}

          {messages.map((message) => (
            <div
              key={message.id}
              className={`message ${message.sender === 'user' ? 'user-message' : 'assistant-message'}`}
            >
              <div className="message-content">
                {message.text}
              </div>
              <div className="message-timestamp">
                {new Date(message.timestamp).toLocaleTimeString()}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="message assistant-message loading">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <textarea
            ref={inputRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message here... (Press Enter to send)"
            className="message-input"
            rows={3}
            disabled={isLoading}
          />
          <button
            onClick={sendMessage}
            disabled={!inputValue.trim() || isLoading}
            className="send-button"
          >
            {isLoading ? '⏳' : '📤'}
          </button>
          </div>
        </div>
      )}

      {isUserIdSet && (
        <footer className="app-footer">
          <p>Session: {sessionId}</p>
          <p>🧠 4-Tier Memory: Redis • PostgreSQL • Neo4j • Procedural</p>
        </footer>
      )}
    </div>
  );
};

export default App;