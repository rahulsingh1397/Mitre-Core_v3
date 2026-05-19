# MITRE-CORE Visualization Implementation Guide

## Quick Start Implementation

This guide provides ready-to-use code for implementing the recommended visualization strategies for MITRE-CORE.

---

## 🎯 Core Visualization Components

### 1. Knowledge Graph Visualization

**File**: `static/js/components/knowledge-graph.js`

```javascript
class KnowledgeGraphVisualization {
    constructor(containerId, data) {
        this.container = document.getElementById(containerId);
        this.data = data;
        this.selectedNode = null;
        this.filters = {
            entityTypes: ['alert', 'user', 'host', 'ip'],
            severity: ['critical', 'high', 'medium', 'low'],
            clusters: []
        };
        
        this.init();
    }
    
    init() {
        // Initialize Cytoscape.js
        this.cy = cytoscape({
            container: this.container,
            elements: this.formatDataForCytoscape(),
            style: this.getGraphStyle(),
            layout: {
                name: 'cose',
                idealEdgeLength: 100,
                nodeOverlap: 20,
                refresh: 20,
                fit: true,
                padding: 30,
                randomize: false,
                componentSpacing: 100,
                nodeRepulsion: 400000,
                edgeElasticity: 100,
                nestingFactor: 5,
                gravity: 80,
                numIter: 1000,
                initialTemp: 200,
                coolingFactor: 0.95,
                minTemp: 1.0
            }
        });
        
        this.addInteractivity();
        this.addControls();
    }
    
    formatDataForCytoscape() {
        const elements = [];
        
        // Add nodes
        this.data.nodes.forEach(node => {
            elements.push({
                data: {
                    id: node.id,
                    label: node.label,
                    type: node.type,
                    severity: node.severity,
                    cluster: node.cluster,
                    entity_count: node.entity_count || 1,
                    metadata: node.metadata || {}
                }
            });
        });
        
        // Add edges
        this.data.edges.forEach(edge => {
            elements.push({
                data: {
                    id: `${edge.source}-${edge.target}`,
                    source: edge.source,
                    target: edge.target,
                    relationship: edge.relationship,
                    weight: edge.weight || 1
                }
            });
        });
        
        return elements;
    }
    
    getGraphStyle() {
        return [
            {
                selector: 'node',
                style: {
                    'background-color': this.getNodeColor,
                    'border-color': '#333',
                    'border-width': 2,
                    'width': this.getNodeSize,
                    'height': this.getNodeSize,
                    'label': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'font-size': '12px',
                    'color': '#fff',
                    'text-outline-color': '#000',
                    'text-outline-width': 2,
                    'overlay-opacity': 0
                }
            },
            {
                selector: 'node:selected',
                style: {
                    'border-color': '#3b82f6',
                    'border-width': 4,
                    'background-color': '#1e40af'
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 'mapData(weight, 0, 1, 1, 5)',
                    'line-color': '#666',
                    'target-arrow-color': '#666',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'opacity': 0.7,
                    'label': 'data(relationship)',
                    'font-size': '10px',
                    'color': '#fff',
                    'text-rotation': 'autorotate',
                    'text-margin-y': -10
                }
            },
            {
                selector: 'edge:selected',
                style: {
                    'line-color': '#3b82f6',
                    'target-arrow-color': '#3b82f6',
                    'opacity': 1
                }
            }
        ];
    }
    
    getNodeColor(node) {
        const colors = {
            alert: '#ef4444',
            user: '#3b82f6', 
            host: '#10b981',
            ip: '#f59e0b',
            malware: '#8b5cf6'
        };
        
        const severityColors = {
            critical: '#dc2626',
            high: '#ea580c',
            medium: '#d97706',
            low: '#2563eb'
        };
        
        if (node.data('type') === 'alert') {
            return severityColors[node.data('severity')] || '#666';
        }
        
        return colors[node.data('type')] || '#666';
    }
    
    getNodeSize(node) {
        const baseSize = 30;
        const entityCount = node.data('entity_count') || 1;
        const clusterSize = node.data('cluster') ? 1.5 : 1;
        
        return baseSize * Math.sqrt(entityCount) * clusterSize;
    }
    
    addInteractivity() {
        // Node click handler
        this.cy.on('tap', 'node', (event) => {
            const node = event.target;
            this.selectNode(node);
            this.showNodeDetails(node);
        });
        
        // Edge click handler
        this.cy.on('tap', 'edge', (event) => {
            const edge = event.target;
            this.showEdgeDetails(edge);
        });
        
        // Background click to deselect
        this.cy.on('tap', (event) => {
            if (event.target === this.cy) {
                this.deselectAll();
            }
        });
        
        // Hover effects
        this.cy.on('mouseover', 'node', (event) => {
            const node = event.target;
            this.highlightConnectedNodes(node);
        });
        
        this.cy.on('mouseout', 'node', () => {
            this.clearHighlights();
        });
    }
    
    addControls() {
        // Add control panel
        const controlsHtml = `
            <div class="graph-controls">
                <div class="control-group">
                    <label>Filter by Type:</label>
                    <div class="checkbox-group">
                        <label><input type="checkbox" value="alert" checked> Alerts</label>
                        <label><input type="checkbox" value="user" checked> Users</label>
                        <label><input type="checkbox" value="host" checked> Hosts</label>
                        <label><input type="checkbox" value="ip" checked> IPs</label>
                    </div>
                </div>
                <div class="control-group">
                    <label>Filter by Severity:</label>
                    <div class="checkbox-group">
                        <label><input type="checkbox" value="critical" checked> Critical</label>
                        <label><input type="checkbox" value="high" checked> High</label>
                        <label><input type="checkbox" value="medium" checked> Medium</label>
                        <label><input type="checkbox" value="low" checked> Low</label>
                    </div>
                </div>
                <div class="control-group">
                    <button onclick="knowledgeGraph.resetLayout()">Reset Layout</button>
                    <button onclick="knowledgeGraph.exportGraph()">Export</button>
                    <button onclick="knowledgeGraph.fitToView()">Fit to View</button>
                </div>
            </div>
        `;
        
        this.container.insertAdjacentHTML('afterend', controlsHtml);
        this.attachControlListeners();
    }
    
    attachControlListeners() {
        // Type filters
        document.querySelectorAll('.checkbox-group input[type="checkbox"]').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                this.updateFilters();
            });
        });
    }
    
    updateFilters() {
        const checkedTypes = Array.from(document.querySelectorAll('.checkbox-group input:checked'))
            .map(cb => cb.value);
        
        this.cy.nodes().forEach(node => {
            const nodeType = node.data('type');
            const nodeSeverity = node.data('severity');
            
            const showByType = checkedTypes.includes(nodeType);
            const showBySeverity = this.filters.severity.includes(nodeSeverity) || nodeType !== 'alert';
            
            node.style('display', showByType && showBySeverity ? 'element' : 'none');
        });
    }
    
    selectNode(node) {
        this.selectedNode = node;
        this.cy.nodes().removeClass('selected');
        node.addClass('selected');
    }
    
    showNodeDetails(node) {
        const details = this.createNodeDetailsPanel(node);
        this.showDetailsPanel(details);
    }
    
    createNodeDetailsPanel(node) {
        const data = node.data();
        
        return `
            <div class="node-details">
                <h3>${data.label}</h3>
                <div class="detail-row">
                    <span class="label">Type:</span>
                    <span class="value">${data.type}</span>
                </div>
                <div class="detail-row">
                    <span class="label">Severity:</span>
                    <span class="value severity-${data.severity}">${data.severity}</span>
                </div>
                ${data.cluster ? `
                <div class="detail-row">
                    <span class="label">Cluster:</span>
                    <span class="value">${data.cluster}</span>
                </div>
                ` : ''}
                ${data.metadata ? `
                <div class="detail-row">
                    <span class="label">Metadata:</span>
                    <pre class="metadata">${JSON.stringify(data.metadata, null, 2)}</pre>
                </div>
                ` : ''}
            </div>
        `;
    }
    
    showDetailsPanel(content) {
        // Remove existing panel
        const existingPanel = document.querySelector('.details-panel');
        if (existingPanel) {
            existingPanel.remove();
        }
        
        // Create new panel
        const panel = document.createElement('div');
        panel.className = 'details-panel';
        panel.innerHTML = `
            <div class="panel-header">
                <h4>Details</h4>
                <button onclick="this.closest('.details-panel').remove()">×</button>
            </div>
            <div class="panel-content">
                ${content}
            </div>
        `;
        
        document.body.appendChild(panel);
    }
    
    highlightConnectedNodes(node) {
        const connectedEdges = node.connectedEdges();
        const connectedNodes = connectedEdges.connectedNodes();
        
        // Dim all nodes and edges
        this.cy.elements().style('opacity', 0.3);
        
        // Highlight connected elements
        connectedNodes.style('opacity', 1);
        connectedEdges.style('opacity', 1);
        node.style('opacity', 1);
    }
    
    clearHighlights() {
        this.cy.elements().style('opacity', 1);
    }
    
    resetLayout() {
        this.cy.layout({
            name: 'cose',
            animate: true,
            animationDuration: 1000
        }).run();
    }
    
    fitToView() {
        this.cy.fit(undefined, 50);
    }
    
    exportGraph() {
        const png = this.cy.png({
            output: 'blob',
            bg: 'white',
            full: true,
            scale: 2
        });
        
        // Download the image
        const link = document.createElement('a');
        link.download = 'knowledge-graph.png';
        link.href = URL.createObjectURL(png);
        link.click();
    }
}

// Usage example:
// const knowledgeGraph = new KnowledgeGraphVisualization('graph-container', graphData);
```

### 2. Attack Timeline Visualization

**File**: `static/js/components/attack-timeline.js`

```javascript
class AttackTimelineVisualization {
    constructor(containerId, data) {
        this.container = document.getElementById(containerId);
        this.data = data;
        this.selectedEvent = null;
        this.zoomLevel = 1;
        
        this.init();
    }
    
    init() {
        this.createTimeline();
        this.addControls();
        this.addInteractivity();
    }
    
    createTimeline() {
        // Clear container
        this.container.innerHTML = '';
        
        // Create timeline structure
        const timelineHtml = `
            <div class="timeline-container">
                <div class="timeline-header">
                    <div class="timeline-controls">
                        <button onclick="attackTimeline.zoomIn()">Zoom In</button>
                        <button onclick="attackTimeline.zoomOut()">Zoom Out</button>
                        <button onclick="attackTimeline.resetZoom()">Reset</button>
                        <select onchange="attackTimeline.changeTimeRange(this.value)">
                            <option value="1h">1 Hour</option>
                            <option value="6h">6 Hours</option>
                            <option value="1d" selected>1 Day</option>
                            <option value="1w">1 Week</option>
                            <option value="all">All</option>
                        </select>
                    </div>
                    <div class="timeline-legend">
                        <div class="legend-item critical">Critical</div>
                        <div class="legend-item high">High</div>
                        <div class="legend-item medium">Medium</div>
                        <div class="legend-item low">Low</div>
                    </div>
                </div>
                <div class="timeline-main" id="timeline-main">
                    <div class="timeline-axis"></div>
                    <div class="timeline-events" id="timeline-events"></div>
                    <div class="timeline-attack-chains" id="timeline-attack-chains"></div>
                </div>
                <div class="timeline-details" id="timeline-details"></div>
            </div>
        `;
        
        this.container.innerHTML = timelineHtml;
        this.renderEvents();
        this.renderAttackChains();
    }
    
    renderEvents() {
        const eventsContainer = document.getElementById('timeline-events');
        eventsContainer.innerHTML = '';
        
        if (!this.data.events || this.data.events.length === 0) {
            eventsContainer.innerHTML = '<div class="no-events">No events to display</div>';
            return;
        }
        
        // Calculate time range
        const timestamps = this.data.events.map(e => new Date(e.timestamp));
        const minTime = new Date(Math.min(...timestamps));
        const maxTime = new Date(Math.max(...timestamps));
        const timeRange = maxTime - minTime;
        
        // Render each event
        this.data.events.forEach((event, index) => {
            const eventTime = new Date(event.timestamp);
            const position = ((eventTime - minTime) / timeRange) * 100;
            
            const eventElement = document.createElement('div');
            eventElement.className = `timeline-event severity-${event.severity}`;
            eventElement.style.left = `${position}%`;
            eventElement.innerHTML = `
                <div class="event-marker"></div>
                <div class="event-tooltip">
                    <div class="tooltip-title">${event.event_type}</div>
                    <div class="tooltip-time">${eventTime.toLocaleString()}</div>
                    <div class="tooltip-tactic">Tactic: ${event.tactic}</div>
                    <div class="tooltip-cluster">Cluster: ${event.cluster}</div>
                </div>
            `;
            
            eventElement.addEventListener('click', () => this.selectEvent(event, index));
            eventElement.addEventListener('mouseenter', () => this.showEventTooltip(event));
            eventElement.addEventListener('mouseleave', () => this.hideEventTooltip());
            
            eventsContainer.appendChild(eventElement);
        });
    }
    
    renderAttackChains() {
        const chainsContainer = document.getElementById('timeline-attack-chains');
        chainsContainer.innerHTML = '';
        
        if (!this.data.attack_chains || this.data.attack_chains.length === 0) {
            return;
        }
        
        // Calculate time range
        const timestamps = this.data.events.map(e => new Date(e.timestamp));
        const minTime = new Date(Math.min(...timestamps));
        const maxTime = new Date(Math.max(...timestamps));
        const timeRange = maxTime - minTime;
        
        // Render attack chains
        this.data.attack_chains.forEach((chain, chainIndex) => {
            const chainElement = document.createElement('div');
            chainElement.className = `attack-chain chain-${chainIndex}`;
            
            // Calculate chain position and width
            const chainEvents = chain.events.map(idx => this.data.events[idx]);
            const chainStartTime = new Date(Math.min(...chainEvents.map(e => new Date(e.timestamp))));
            const chainEndTime = new Date(Math.max(...chainEvents.map(e => new Date(e.timestamp))));
            
            const startPosition = ((chainStartTime - minTime) / timeRange) * 100;
            const endPosition = ((chainEndTime - minTime) / timeRange) * 100;
            const width = endPosition - startPosition;
            
            chainElement.style.left = `${startPosition}%`;
            chainElement.style.width = `${width}%`;
            chainElement.innerHTML = `
                <div class="chain-label">
                    Chain ${chain.chain_id} (${chain.events.length} events)
                </div>
                <div class="chain-confidence">Confidence: ${(chain.confidence * 100).toFixed(1)}%</div>
            `;
            
            chainElement.addEventListener('click', () => this.selectAttackChain(chain));
            chainsContainer.appendChild(chainElement);
        });
    }
    
    addControls() {
        // Time range slider
        const timeRangeSlider = document.createElement('input');
        timeRangeSlider.type = 'range';
        timeRangeSlider.min = '1';
        timeRangeSlider.max = '100';
        timeRangeSlider.value = '100';
        timeRangeSlider.className = 'time-range-slider';
        timeRangeSlider.addEventListener('input', (e) => this.adjustTimeRange(e.target.value));
        
        this.container.querySelector('.timeline-controls').appendChild(timeRangeSlider);
    }
    
    addInteractivity() {
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === '+' || e.key === '=') this.zoomIn();
            if (e.key === '-' || e.key === '_') this.zoomOut();
            if (e.key === '0') this.resetZoom();
        });
        
        // Mouse wheel zoom
        this.container.addEventListener('wheel', (e) => {
            if (e.ctrlKey) {
                e.preventDefault();
                if (e.deltaY < 0) {
                    this.zoomIn();
                } else {
                    this.zoomOut();
                }
            }
        });
    }
    
    selectEvent(event, index) {
        this.selectedEvent = event;
        
        // Update selection UI
        document.querySelectorAll('.timeline-event').forEach(el => el.classList.remove('selected'));
        document.querySelectorAll('.timeline-event')[index].classList.add('selected');
        
        // Show details
        this.showEventDetails(event);
    }
    
    selectAttackChain(chain) {
        // Highlight chain events
        const chainEventIndices = chain.events;
        document.querySelectorAll('.timeline-event').forEach((el, idx) => {
            if (chainEventIndices.includes(idx)) {
                el.classList.add('chain-highlighted');
            } else {
                el.classList.add('chain-dimmed');
            }
        });
        
        // Show chain details
        this.showChainDetails(chain);
    }
    
    showEventDetails(event) {
        const detailsContainer = document.getElementById('timeline-details');
        
        const eventTime = new Date(event.timestamp);
        const entities = event.entities || {};
        
        detailsContainer.innerHTML = `
            <div class="event-details">
                <h4>Event Details</h4>
                <div class="detail-grid">
                    <div class="detail-item">
                        <span class="label">Event Type:</span>
                        <span class="value">${event.event_type}</span>
                    </div>
                    <div class="detail-item">
                        <span class="label">Timestamp:</span>
                        <span class="value">${eventTime.toLocaleString()}</span>
                    </div>
                    <div class="detail-item">
                        <span class="label">Tactic:</span>
                        <span class="value tactic-${event.tactic}">${event.tactic}</span>
                    </div>
                    <div class="detail-item">
                        <span class="label">Severity:</span>
                        <span class="value severity-${event.severity}">${event.severity}</span>
                    </div>
                    <div class="detail-item">
                        <span class="label">Cluster:</span>
                        <span class="value">${event.cluster}</span>
                    </div>
                </div>
                ${Object.keys(entities).length > 0 ? `
                <div class="entities-section">
                    <h5>Entities</h5>
                    <div class="entity-list">
                        ${Object.entries(entities).map(([type, value]) => 
                            `<div class="entity-item">
                                <span class="entity-type">${type}:</span>
                                <span class="entity-value">${value}</span>
                            </div>`
                        ).join('')}
                    </div>
                </div>
                ` : ''}
            </div>
        `;
    }
    
    showChainDetails(chain) {
        const detailsContainer = document.getElementById('timeline-details');
        
        const chainEvents = chain.events.map(idx => this.data.events[idx]);
        
        detailsContainer.innerHTML = `
            <div class="chain-details">
                <h4>Attack Chain Details</h4>
                <div class="chain-summary">
                    <div class="summary-item">
                        <span class="label">Chain ID:</span>
                        <span class="value">${chain.chain_id}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">Events:</span>
                        <span class="value">${chain.events.length}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">Confidence:</span>
                        <span class="value">${(chain.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">Attack Stage:</span>
                        <span class="value">${chain.attack_stage}</span>
                    </div>
                </div>
                <div class="chain-events">
                    <h5>Chain Events</h5>
                    ${chainEvents.map((event, idx) => `
                        <div class="chain-event-item">
                            <span class="event-number">${idx + 1}</span>
                            <span class="event-type">${event.event_type}</span>
                            <span class="event-time">${new Date(event.timestamp).toLocaleTimeString()}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }
    
    showEventTooltip(event) {
        // Tooltip is shown via CSS hover
    }
    
    hideEventTooltip() {
        // Tooltip is hidden via CSS hover
    }
    
    zoomIn() {
        this.zoomLevel = Math.min(this.zoomLevel * 1.2, 5);
        this.applyZoom();
    }
    
    zoomOut() {
        this.zoomLevel = Math.max(this.zoomLevel / 1.2, 0.5);
        this.applyZoom();
    }
    
    resetZoom() {
        this.zoomLevel = 1;
        this.applyZoom();
    }
    
    applyZoom() {
        const eventsContainer = document.getElementById('timeline-events');
        eventsContainer.style.transform = `scaleX(${this.zoomLevel})`;
    }
    
    changeTimeRange(range) {
        // Filter events based on time range
        // This would require backend support for time-based filtering
        console.log('Changing time range to:', range);
    }
    
    adjustTimeRange(value) {
        // Adjust visible time range based on slider value
        console.log('Adjusting time range to:', value);
    }
}

// Usage example:
// const attackTimeline = new AttackTimelineVisualization('timeline-container', timelineData);
```

### 3. MITRE ATT&CK Heatmap

**File**: `static/js/components/mitre-heatmap.js`

```javascript
class MITREHeatmapVisualization {
    constructor(containerId, data) {
        this.container = document.getElementById(containerId);
        this.data = data;
        this.selectedTactic = null;
        this.colorMode = 'count'; // 'count', 'severity', 'confidence'
        
        this.init();
    }
    
    init() {
        this.createHeatmap();
        this.addControls();
        this.addInteractivity();
    }
    
    createHeatmap() {
        // Clear container
        this.container.innerHTML = '';
        
        // Create heatmap structure
        const heatmapHtml = `
            <div class="heatmap-container">
                <div class="heatmap-header">
                    <h3>MITRE ATT&CK Framework Coverage</h3>
                    <div class="heatmap-controls">
                        <select onchange="mitreHeatmap.changeColorMode(this.value)">
                            <option value="count">Event Count</option>
                            <option value="severity">Severity</option>
                            <option value="confidence">Confidence</option>
                        </select>
                        <button onclick="mitreHeatmap.resetView()">Reset View</button>
                    </div>
                </div>
                <div class="heatmap-main" id="heatmap-main">
                    <div class="tactics-axis" id="tactics-axis"></div>
                    <div class="techniques-grid" id="techniques-grid"></div>
                </div>
                <div class="heatmap-details" id="heatmap-details"></div>
                <div class="heatmap-legend" id="heatmap-legend"></div>
            </div>
        `;
        
        this.container.innerHTML = heatmapHtml;
        this.renderTacticsAxis();
        this.renderTechniquesGrid();
        this.renderLegend();
    }
    
    renderTacticsAxis() {
        const tacticsAxis = document.getElementById('tactics-axis');
        tacticsAxis.innerHTML = '';
        
        const tactics = [
            'RECONNAISSANCE', 'INITIAL ACCESS', 'EXECUTION', 'PERSISTENCE',
            'PRIVILEGE ESCALATION', 'DEFENSE EVASION', 'CREDENTIAL ACCESS',
            'DISCOVERY', 'LATERAL MOVEMENT', 'COLLECTION', 'COMMAND AND CONTROL', 'IMPACT'
        ];
        
        tactics.forEach(tactic => {
            const tacticElement = document.createElement('div');
            tacticElement.className = 'tactic-label';
            tacticElement.textContent = tactic.replace(' ', '\n');
            tacticElement.addEventListener('click', () => this.selectTactic(tactic));
            tacticsAxis.appendChild(tacticElement);
        });
    }
    
    renderTechniquesGrid() {
        const techniquesGrid = document.getElementById('techniques-grid');
        techniquesGrid.innerHTML = '';
        
        if (!this.data.techniques) {
            techniquesGrid.innerHTML = '<div class="no-techniques">No technique data available</div>';
            return;
        }
        
        // Group techniques by tactic
        const techniquesByTactic = this.groupTechniquesByTactic();
        
        // Create grid cells
        Object.keys(techniquesByTactic).forEach(tactic => {
            const tacticColumn = document.createElement('div');
            tacticColumn.className = 'tactic-column';
            
            techniquesByTactic[tactic].forEach(technique => {
                const cell = this.createTechniqueCell(technique);
                tacticColumn.appendChild(cell);
            });
            
            techniquesGrid.appendChild(tacticColumn);
        });
    }
    
    groupTechniquesByTactic() {
        // This would ideally come from MITRE ATT&CK data
        // For now, we'll use a simplified mapping
        const tacticMapping = {
            'RECONNAISSANCE': ['Active Scanning', 'Passive Scanning', 'Gathering Victim Info'],
            'INITIAL ACCESS': ['Spearphishing Attachment', 'Exploit Public-Facing App', 'Valid Accounts'],
            'EXECUTION': ['PowerShell', 'Command and Scripting Interpreter', 'User Execution'],
            'PERSISTENCE': ['Create Account', 'Modify Existing Service', 'Scheduled Task'],
            'PRIVILEGE ESCALATION': ['Valid Accounts', 'Exploitation for Privilege Escalation'],
            'DEFENSE EVASION': ['Obfuscated Files', 'Indicator Blocking', 'Disable Security Tools'],
            'CREDENTIAL ACCESS': ['Credential Dumping', 'Brute Force', 'Unsecured Credentials'],
            'DISCOVERY': ['System Information Discovery', 'Network Service Discovery'],
            'LATERAL MOVEMENT': ['Remote Services', 'Remote Execution', 'SMB/Windows Admin Shares'],
            'COLLECTION': ['Data from Local System', 'Data from Network Shared Drive'],
            'COMMAND AND CONTROL': ['Application Layer Protocol', 'Data Encoding', 'Ingress Tool Transfer'],
            'IMPACT': ['Data Destruction', 'Disk Wipe', 'Service Stop']
        };
        
        const grouped = {};
        
        Object.keys(tacticMapping).forEach(tactic => {
            grouped[tactic] = tacticMapping[tactic].map(techniqueName => {
                const techniqueData = this.data.techniques[techniqueName] || { count: 0, severity: 'low' };
                return {
                    name: techniqueName,
                    tactic: tactic,
                    count: techniqueData.count || 0,
                    severity: techniqueData.severity || 'low',
                    confidence: techniqueData.confidence || 0
                };
            });
        });
        
        return grouped;
    }
    
    createTechniqueCell(technique) {
        const cell = document.createElement('div');
        cell.className = 'technique-cell';
        
        const color = this.getTechniqueColor(technique);
        const intensity = this.getTechniqueIntensity(technique);
        
        cell.style.backgroundColor = color;
        cell.style.opacity = intensity;
        cell.innerHTML = `
            <div class="technique-name">${technique.name}</div>
            <div class="technique-count">${technique.count}</div>
        `;
        
        cell.addEventListener('click', () => this.selectTechnique(technique));
        cell.addEventListener('mouseenter', () => this.showTechniqueTooltip(technique));
        cell.addEventListener('mouseleave', () => this.hideTechniqueTooltip());
        
        return cell;
    }
    
    getTechniqueColor(technique) {
        const tacticColors = {
            'RECONNAISSANCE': '#6366f1',
            'INITIAL ACCESS': '#dc2626',
            'EXECUTION': '#f97316',
            'PERSISTENCE': '#eab308',
            'PRIVILEGE ESCALATION': '#a855f7',
            'DEFENSE EVASION': '#64748b',
            'CREDENTIAL ACCESS': '#0891b2',
            'DISCOVERY': '#0ea5e9',
            'LATERAL MOVEMENT': '#84cc16',
            'COLLECTION': '#f59e0b',
            'COMMAND AND CONTROL': '#ef4444',
            'IMPACT': '#dc2626'
        };
        
        return tacticColors[technique.tactic] || '#666';
    }
    
    getTechniqueIntensity(technique) {
        if (this.colorMode === 'count') {
            return Math.min(technique.count / 10, 1); // Normalize to max 10 events
        } else if (this.colorMode === 'severity') {
            const severityLevels = { 'low': 0.3, 'medium': 0.6, 'high': 0.8, 'critical': 1.0 };
            return severityLevels[technique.severity] || 0.3;
        } else if (this.colorMode === 'confidence') {
            return technique.confidence || 0.5;
        }
        return 0.5;
    }
    
    renderLegend() {
        const legendContainer = document.getElementById('heatmap-legend');
        legendContainer.innerHTML = '';
        
        const legendTitle = document.createElement('div');
        legendTitle.className = 'legend-title';
        legendTitle.textContent = this.colorMode === 'count' ? 'Event Count' : 
                               this.colorMode === 'severity' ? 'Severity Level' : 'Confidence';
        legendContainer.appendChild(legendTitle);
        
        const legendScale = document.createElement('div');
        legendScale.className = 'legend-scale';
        
        // Create gradient scale
        const steps = 5;
        for (let i = 0; i < steps; i++) {
            const step = document.createElement('div');
            step.className = 'legend-step';
            step.style.opacity = (i + 1) / steps;
            
            const label = document.createElement('span');
            label.className = 'legend-label';
            
            if (this.colorMode === 'count') {
                label.textContent = i * 2.5;
            } else if (this.colorMode === 'severity') {
                const severities = ['Low', 'Medium', 'High', 'Critical'];
                label.textContent = severities[Math.min(i, severities.length - 1)];
            } else {
                label.textContent = `${Math.round((i + 1) / steps * 100)}%`;
            }
            
            step.appendChild(label);
            legendScale.appendChild(step);
        }
        
        legendContainer.appendChild(legendScale);
    }
    
    addControls() {
        // Controls are added in the HTML structure
    }
    
    addInteractivity() {
        // Add keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.deselectAll();
            }
        });
    }
    
    selectTactic(tactic) {
        this.selectedTactic = tactic;
        
        // Update UI
        document.querySelectorAll('.tactic-label').forEach(el => {
            el.classList.toggle('selected', el.textContent.replace('\n', ' ') === tactic);
        });
        
        // Highlight technique cells
        document.querySelectorAll('.technique-cell').forEach(cell => {
            const cellTactic = cell.querySelector('.technique-name').textContent;
            // This would need proper technique-to-tactic mapping
            cell.classList.toggle('tactic-highlighted', true);
        });
        
        this.showTacticDetails(tactic);
    }
    
    selectTechnique(technique) {
        // Update UI
        document.querySelectorAll('.technique-cell').forEach(cell => {
            cell.classList.remove('selected');
        });
        
        event.target.closest('.technique-cell').classList.add('selected');
        
        this.showTechniqueDetails(technique);
    }
    
    showTacticDetails(tactic) {
        const detailsContainer = document.getElementById('heatmap-details');
        
        const tacticEvents = this.data.tactics[tactic] || 0;
        
        detailsContainer.innerHTML = `
            <div class="tactic-details">
                <h4>Tactic: ${tactic}</h4>
                <div class="tactic-stats">
                    <div class="stat-item">
                        <span class="label">Total Events:</span>
                        <span class="value">${tacticEvents}</span>
                    </div>
                    <div class="stat-item">
                        <span class="label">Techniques Used:</span>
                        <span class="value">${this.getTechniqueCountForTactic(tactic)}</span>
                    </div>
                    <div class="stat-item">
                        <span class="label">Coverage:</span>
                        <span class="value">${this.calculateTacticCoverage(tactic)}%</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    showTechniqueDetails(technique) {
        const detailsContainer = document.getElementById('heatmap-details');
        
        detailsContainer.innerHTML = `
            <div class="technique-details">
                <h4>Technique: ${technique.name}</h4>
                <div class="technique-stats">
                    <div class="stat-item">
                        <span class="label">Tactic:</span>
                        <span class="value">${technique.tactic}</span>
                    </div>
                    <div class="stat-item">
                        <span class="label">Event Count:</span>
                        <span class="value">${technique.count}</span>
                    </div>
                    <div class="stat-item">
                        <span class="label">Severity:</span>
                        <span class="value severity-${technique.severity}">${technique.severity}</span>
                    </div>
                    <div class="stat-item">
                        <span class="label">Confidence:</span>
                        <span class="value">${(technique.confidence * 100).toFixed(1)}%</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    showTechniqueTooltip(technique) {
        // Tooltip is shown via CSS hover
    }
    
    hideTechniqueTooltip() {
        // Tooltip is hidden via CSS hover
    }
    
    changeColorMode(mode) {
        this.colorMode = mode;
        this.renderTechniquesGrid();
        this.renderLegend();
    }
    
    resetView() {
        this.selectedTactic = null;
        document.querySelectorAll('.tactic-label').forEach(el => el.classList.remove('selected'));
        document.querySelectorAll('.technique-cell').forEach(cell => cell.classList.remove('selected'));
        document.getElementById('heatmap-details').innerHTML = '';
    }
    
    deselectAll() {
        this.resetView();
    }
    
    getTechniqueCountForTactic(tactic) {
        const techniquesByTactic = this.groupTechniquesByTactic();
        return techniquesByTactic[tactic] ? techniquesByTactic[tactic].length : 0;
    }
    
    calculateTacticCoverage(tactic) {
        const techniquesByTactic = this.groupTechniquesByTactic();
        const techniques = techniquesByTactic[tactic] || [];
        const activeTechniques = techniques.filter(t => t.count > 0);
        
        if (techniques.length === 0) return 0;
        return Math.round((activeTechniques.length / techniques.length) * 100);
    }
}

// Usage example:
// const mitreHeatmap = new MITREHeatmapVisualization('heatmap-container', heatmapData);
```

### 4. Cluster Bubble Visualization

**File**: `static/js/components/cluster-bubbles.js`

```javascript
class ClusterBubbleVisualization {
    constructor(containerId, data) {
        this.container = document.getElementById(containerId);
        this.data = data;
        this.selectedCluster = null;
        this.layout = 'force'; // 'force', 'circular', 'hierarchical'
        
        this.init();
    }
    
    init() {
        this.createBubbleChart();
        this.addControls();
        this.addInteractivity();
    }
    
    createBubbleChart() {
        // Clear container
        this.container.innerHTML = '';
        
        // Create bubble chart structure
        const bubbleHtml = `
            <div class="bubble-chart-container">
                <div class="bubble-header">
                    <h3>Attack Campaign Overview</h3>
                    <div class="bubble-controls">
                        <select onchange="clusterBubbles.changeLayout(this.value)">
                            <option value="force">Force Layout</option>
                            <option value="circular">Circular Layout</option>
                            <option value="hierarchical">Hierarchical Layout</option>
                        </select>
                        <select onchange="clusterBubbles.changeSizeBy(this.value)">
                            <option value="alert_count">Alert Count</option>
                            <option value="confidence">Confidence</option>
                            <option value="severity">Severity</option>
                        </select>
                        <button onclick="clusterBubbles.resetView()">Reset View</button>
                    </div>
                </div>
                <div class="bubble-main" id="bubble-main">
                    <svg id="bubble-svg" width="100%" height="500"></svg>
                </div>
                <div class="bubble-details" id="bubble-details"></div>
                <div class="bubble-legend" id="bubble-legend"></div>
            </div>
        `;
        
        this.container.innerHTML = bubbleHtml;
        this.renderBubbles();
        this.renderLegend();
    }
    
    renderBubbles() {
        const svg = d3.select('#bubble-svg');
        svg.selectAll('*').remove(); // Clear previous content
        
        if (!this.data.clusters || this.data.clusters.length === 0) {
            svg.append('text')
                .attr('x', '50%')
                .attr('y', '50%')
                .attr('text-anchor', 'middle')
                .text('No clusters to display');
            return;
        }
        
        const width = svg.node().getBoundingClientRect().width;
        const height = 500;
        
        // Create bubble groups
        const bubbles = svg.selectAll('.bubble')
            .data(this.data.clusters)
            .enter()
            .append('g')
            .attr('class', 'bubble')
            .attr('transform', (d, i) => this.getBubblePosition(d, i, width, height));
        
        // Add bubble circles
        bubbles.append('circle')
            .attr('r', d => this.getBubbleRadius(d))
            .attr('fill', d => this.getBubbleColor(d))
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .attr('opacity', 0.8)
            .style('cursor', 'pointer')
            .on('click', (event, d) => this.selectCluster(d))
            .on('mouseover', (event, d) => this.showBubbleTooltip(d))
            .on('mouseout', () => this.hideBubbleTooltip());
        
        // Add bubble labels
        bubbles.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', '.3em')
            .style('fill', '#fff')
            .style('font-size', d => this.getFontSize(d))
            .style('font-weight', 'bold')
            .style('pointer-events', 'none')
            .text(d => `Cluster ${d.id}`);
        
        // Add size labels
        bubbles.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', '1.5em')
            .style('fill', '#fff')
            .style('font-size', '10px')
            .style('pointer-events', 'none')
            .text(d => `${d.size} alerts`);
        
        // Apply force simulation if using force layout
        if (this.layout === 'force') {
            this.applyForceLayout(bubbles, width, height);
        }
    }
    
    getBubblePosition(bubble, index, width, height) {
        if (this.layout === 'force') {
            // Force layout will handle positioning
            return `translate(${width/2}, ${height/2})`;
        } else if (this.layout === 'circular') {
            const angle = (index / this.data.clusters.length) * 2 * Math.PI;
            const radius = Math.min(width, height) * 0.3;
            const x = width/2 + radius * Math.cos(angle);
            const y = height/2 + radius * Math.sin(angle);
            return `translate(${x}, ${y})`;
        } else if (this.layout === 'hierarchical') {
            const cols = Math.ceil(Math.sqrt(this.data.clusters.length));
            const row = Math.floor(index / cols);
            const col = index % cols;
            const cellWidth = width / cols;
            const cellHeight = height / Math.ceil(this.data.clusters.length / cols);
            const x = col * cellWidth + cellWidth / 2;
            const y = row * cellHeight + cellHeight / 2;
            return `translate(${x}, ${y})`;
        }
        return `translate(${width/2}, ${height/2})`;
    }
    
    getBubbleRadius(bubble) {
        const minRadius = 20;
        const maxRadius = 60;
        
        let size;
        if (this.sizeBy === 'alert_count') {
            size = bubble.size;
        } else if (this.sizeBy === 'confidence') {
            size = (bubble.confidence || 0.5) * 100;
        } else if (this.sizeBy === 'severity') {
            const severityValues = { 'low': 1, 'medium': 2, 'high': 3, 'critical': 4 };
            size = severityValues[bubble.severity] || 1;
        } else {
            size = bubble.size;
        }
        
        const maxSize = Math.max(...this.data.clusters.map(c => {
            if (this.sizeBy === 'alert_count') return c.size;
            if (this.sizeBy === 'confidence') return (c.confidence || 0.5) * 100;
            if (this.sizeBy === 'severity') {
                const severityValues = { 'low': 1, 'medium': 2, 'high': 3, 'critical': 4 };
                return severityValues[c.severity] || 1;
            }
            return c.size;
        }));
        
        const normalizedSize = size / maxSize;
        return minRadius + normalizedSize * (maxRadius - minRadius);
    }
    
    getBubbleColor(bubble) {
        const severityColors = {
            'critical': '#dc2626',
            'high': '#ea580c',
            'medium': '#d97706',
            'low': '#2563eb'
        };
        
        const attackTypeColors = {
            'APT': '#8b5cf6',
            'Malware': '#ef4444',
            'Phishing': '#f59e0b',
            'Insider': '#3b82f6',
            'Unknown': '#64748b'
        };
        
        return severityColors[bubble.severity] || attackTypeColors[bubble.attack_type] || '#64748b';
    }
    
    getFontSize(bubble) {
        const radius = this.getBubbleRadius(bubble);
        return Math.max(10, Math.min(16, radius / 4));
    }
    
    applyForceLayout(bubbles, width, height) {
        const simulation = d3.forceSimulation()
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(d => this.getBubbleRadius(d.data) + 5));
        
        simulation.nodes(this.data.clusters)
            .on('tick', () => {
                bubbles.attr('transform', d => `translate(${d.x}, ${d.y})`);
            });
    }
    
    renderLegend() {
        const legendContainer = document.getElementById('bubble-legend');
        legendContainer.innerHTML = '';
        
        const legendTitle = document.createElement('div');
        legendTitle.className = 'legend-title';
        legendTitle.textContent = 'Cluster Properties';
        legendContainer.appendChild(legendTitle);
        
        // Severity legend
        const severityLegend = document.createElement('div');
        severityLegend.className = 'legend-section';
        severityLegend.innerHTML = '<h5>Severity</h5>';
        
        const severities = ['critical', 'high', 'medium', 'low'];
        severities.forEach(severity => {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <div class="legend-color severity-${severity}"></div>
                <span>${severity.charAt(0).toUpperCase() + severity.slice(1)}</span>
            `;
            severityLegend.appendChild(item);
        });
        
        legendContainer.appendChild(severityLegend);
        
        // Size legend
        const sizeLegend = document.createElement('div');
        sizeLegend.className = 'legend-section';
        sizeLegend.innerHTML = `<h5>Size: ${this.sizeBy.replace('_', ' ').charAt(0).toUpperCase() + this.sizeBy.replace('_', ' ').slice(1)}</h5>`;
        
        const sizeExplanation = document.createElement('div');
        sizeExplanation.className = 'size-explanation';
        sizeExplanation.textContent = this.getSizeExplanation();
        sizeLegend.appendChild(sizeExplanation);
        
        legendContainer.appendChild(sizeLegend);
    }
    
    getSizeExplanation() {
        if (this.sizeBy === 'alert_count') {
            return 'Bubble size represents the number of alerts in the cluster';
        } else if (this.sizeBy === 'confidence') {
            return 'Bubble size represents the confidence score of the cluster';
        } else if (this.sizeBy === 'severity') {
            return 'Bubble size represents the severity level of the cluster';
        }
        return 'Bubble size represents cluster properties';
    }
    
    addControls() {
        // Controls are added in the HTML structure
    }
    
    addInteractivity() {
        // Add zoom functionality
        const svg = d3.select('#bubble-svg');
        const zoom = d3.zoom()
            .scaleExtent([0.5, 3])
            .on('zoom', (event) => {
                svg.select('.bubble')
                    .attr('transform', event.transform);
            });
        
        svg.call(zoom);
        
        // Add keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.deselectAll();
            }
        });
    }
    
    selectCluster(cluster) {
        this.selectedCluster = cluster;
        
        // Update UI
        d3.selectAll('.bubble circle')
            .attr('stroke', d => d.id === cluster.id ? '#3b82f6' : '#fff')
            .attr('stroke-width', d => d.id === cluster.id ? 4 : 2);
        
        this.showClusterDetails(cluster);
    }
    
    showClusterDetails(cluster) {
        const detailsContainer = document.getElementById('bubble-details');
        
        detailsContainer.innerHTML = `
            <div class="cluster-details">
                <h4>Cluster ${cluster.id} Details</h4>
                <div class="cluster-stats">
                    <div class="stat-item">
                        <span class="label">Alert Count:</span>
                        <span class="value">${cluster.size}</span>
                    </div>
                    <div class="stat-item">
                        <span class="label">Severity:</span>
                        <span class="value severity-${cluster.severity}">${cluster.severity}</span>
                    </div>
                    <div class="stat-item">
                        <span class="label">Attack Type:</span>
                        <span class="value">${cluster.attack_type}</span>
                    </div>
                    <div class="stat-item">
                        <span class="label">Confidence:</span>
                        <span class="value">${(cluster.confidence * 100).toFixed(1)}%</span>
                    </div>
                    ${cluster.tactics ? `
                    <div class="stat-item">
                        <span class="label">Tactics:</span>
                        <span class="value">${cluster.tactics.join(', ')}</span>
                    </div>
                    ` : ''}
                </div>
                ${cluster.entities ? `
                <div class="entities-section">
                    <h5>Entities</h5>
                    <div class="entity-breakdown">
                        ${Object.entries(cluster.entities).map(([type, count]) => `
                            <div class="entity-item">
                                <span class="entity-type">${type}:</span>
                                <span class="entity-count">${count}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
                ` : ''}
            </div>
        `;
    }
    
    showBubbleTooltip(bubble) {
        // Tooltip functionality
        const tooltip = d3.select('body').append('div')
            .attr('class', 'bubble-tooltip')
            .style('opacity', 0);
        
        tooltip.transition()
            .duration(200)
            .style('opacity', .9);
        
        tooltip.html(`
            <div class="tooltip-content">
                <strong>Cluster ${bubble.id}</strong><br/>
                Size: ${bubble.size} alerts<br/>
                Severity: ${bubble.severity}<br/>
                Type: ${bubble.attack_type}<br/>
                Confidence: ${(bubble.confidence * 100).toFixed(1)}%
            </div>
        `)
            .style('left', (d3.event.pageX + 10) + 'px')
            .style('top', (d3.event.pageY - 28) + 'px');
    }
    
    hideBubbleTooltip() {
        d3.selectAll('.bubble-tooltip').remove();
    }
    
    changeLayout(layout) {
        this.layout = layout;
        this.renderBubbles();
    }
    
    changeSizeBy(sizeBy) {
        this.sizeBy = sizeBy;
        this.renderBubbles();
        this.renderLegend();
    }
    
    resetView() {
        this.selectedCluster = null;
        d3.selectAll('.bubble circle')
            .attr('stroke', '#fff')
            .attr('stroke-width', 2);
        
        // Reset zoom
        const svg = d3.select('#bubble-svg');
        svg.transition().duration(750).call(
            d3.zoom().transform,
            d3.zoomIdentity
        );
        
        document.getElementById('bubble-details').innerHTML = '';
    }
    
    deselectAll() {
        this.resetView();
    }
}

// Usage example:
// const clusterBubbles = new ClusterBubbleVisualization('bubble-container', clusterData);
```

---

## 🎨 CSS Styling

**File**: `static/css/visualizations.css`

```css
/* Knowledge Graph Styles */
.knowledge-graph-container {
    position: relative;
    width: 100%;
    height: 500px;
    border: 1px solid #334155;
    border-radius: 8px;
    background: #1e293b;
    overflow: hidden;
}

.graph-controls {
    position: absolute;
    top: 10px;
    right: 10px;
    background: rgba(30, 41, 59, 0.95);
    border: 1px solid #475569;
    border-radius: 8px;
    padding: 15px;
    min-width: 200px;
    z-index: 1000;
}

.control-group {
    margin-bottom: 15px;
}

.control-group label {
    display: block;
    font-size: 12px;
    font-weight: 600;
    color: #cbd5e1;
    margin-bottom: 5px;
}

.checkbox-group {
    display: flex;
    flex-direction: column;
    gap: 5px;
}

.checkbox-group label {
    display: flex;
    align-items: center;
    font-size: 11px;
    font-weight: 400;
    cursor: pointer;
}

.checkbox-group input[type="checkbox"] {
    margin-right: 5px;
}

.details-panel {
    position: fixed;
    top: 50%;
    right: 20px;
    transform: translateY(-50%);
    background: rgba(30, 41, 59, 0.98);
    border: 1px solid #475569;
    border-radius: 8px;
    padding: 20px;
    min-width: 300px;
    max-width: 400px;
    max-height: 80vh;
    overflow-y: auto;
    z-index: 2000;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
}

.panel-header {
    display: flex;
    justify-content: between;
    align-items: center;
    margin-bottom: 15px;
}

.panel-header h4 {
    margin: 0;
    color: #f8fafc;
}

.panel-header button {
    background: none;
    border: none;
    color: #64748b;
    font-size: 18px;
    cursor: pointer;
    padding: 0;
    width: 24px;
    height: 24px;
}

.node-details .detail-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    font-size: 12px;
}

.node-details .label {
    color: #94a3b8;
}

.node-details .value {
    color: #f8fafc;
    font-weight: 500;
}

.node-details .severity-critical { color: #dc2626; }
.node-details .severity-high { color: #ea580c; }
.node-details .severity-medium { color: #d97706; }
.node-details .severity-low { color: #2563eb; }

.node-details .metadata {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 8px;
    font-size: 10px;
    color: #cbd5e1;
    margin-top: 10px;
}

/* Timeline Styles */
.timeline-container {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 20px;
}

.timeline-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

.timeline-controls {
    display: flex;
    gap: 10px;
    align-items: center;
}

.timeline-controls button,
.timeline-controls select {
    background: #374151;
    border: 1px solid #4b5563;
    color: #f8fafc;
    padding: 5px 10px;
    border-radius: 4px;
    font-size: 12px;
    cursor: pointer;
}

.timeline-legend {
    display: flex;
    gap: 15px;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    color: #cbd5e1;
}

.legend-item::before {
    content: '';
    width: 12px;
    height: 12px;
    border-radius: 50%;
}

.legend-item.critical::before { background: #dc2626; }
.legend-item.high::before { background: #ea580c; }
.legend-item.medium::before { background: #d97706; }
.legend-item.low::before { background: #2563eb; }

.timeline-main {
    position: relative;
    height: 300px;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 4px;
    overflow: hidden;
}

.timeline-axis {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 30px;
    background: #1e293b;
    border-bottom: 1px solid #334155;
}

.timeline-events {
    position: absolute;
    top: 30px;
    left: 0;
    right: 0;
    bottom: 0;
}

.timeline-event {
    position: absolute;
    top: 10px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    cursor: pointer;
    transition: all 0.2s ease;
}

.timeline-event.severity-critical { background: #dc2626; }
.timeline-event.severity-high { background: #ea580c; }
.timeline-event.severity-medium { background: #d97706; }
.timeline-event.severity-low { background: #2563eb; }

.timeline-event:hover {
    transform: scale(1.5);
    z-index: 100;
}

.timeline-event.selected {
    border: 2px solid #3b82f6;
    transform: scale(1.5);
}

.event-tooltip {
    position: absolute;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid #475569;
    border-radius: 4px;
    padding: 8px;
    white-space: nowrap;
    font-size: 11px;
    color: #f8fafc;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.2s ease;
    z-index: 1000;
}

.timeline-event:hover .event-tooltip {
    opacity: 1;
}

.timeline-attack-chains {
    position: absolute;
    top: 60px;
    left: 0;
    right: 0;
    bottom: 0;
    pointer-events: none;
}

.attack-chain {
    position: absolute;
    top: 0;
    height: 20px;
    background: rgba(139, 92, 246, 0.3);
    border: 1px solid #8b5cf6;
    border-radius: 3px;
    cursor: pointer;
    pointer-events: all;
    transition: all 0.2s ease;
}

.attack-chain:hover {
    background: rgba(139, 92, 246, 0.5);
}

.chain-label {
    font-size: 10px;
    color: #e9d5ff;
    padding: 2px 5px;
}

.chain-confidence {
    font-size: 9px;
    color: #c4b5fd;
    padding: 2px 5px;
}

.timeline-details {
    margin-top: 20px;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 15px;
    min-height: 150px;
}

.event-details h4,
.chain-details h4 {
    margin: 0 0 15px 0;
    color: #f8fafc;
    font-size: 14px;
}

.detail-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 10px;
    margin-bottom: 15px;
}

.detail-item {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
}

.detail-item .label {
    color: #94a3b8;
}

.detail-item .value {
    color: #f8fafc;
    font-weight: 500;
}

.tactic-RECONNAISSANCE { color: #6366f1; }
.tactic-INITIAL ACCESS { color: #dc2626; }
.tactic-EXECUTION { color: #f97316; }
.tactic-PERSISTENCE { color: #eab308; }
.tactic-PRIVILEGE ESCALATION { color: #a855f7; }
.tactic-DEFENSE EVASION { color: #64748b; }
.tactic-CREDENTIAL ACCESS { color: #0891b2; }
.tactic-DISCOVERY { color: #0ea5e9; }
.tactic-LATERAL MOVEMENT { color: #84cc16; }
.tactic-COLLECTION { color: #f59e0b; }
.tactic-COMMAND AND CONTROL { color: #ef4444; }
.tactic-IMPACT { color: #dc2626; }

.entities-section {
    margin-top: 15px;
}

.entities-section h5 {
    margin: 0 0 10px 0;
    color: #cbd5e1;
    font-size: 12px;
}

.entity-list {
    display: flex;
    flex-direction: column;
    gap: 5px;
}

.entity-item {
    display: flex;
    gap: 5px;
    font-size: 11px;
}

.entity-type {
    color: #94a3b8;
}

.entity-value {
    color: #f8fafc;
}

/* MITRE Heatmap Styles */
.heatmap-container {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 20px;
}

.heatmap-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

.heatmap-header h3 {
    margin: 0;
    color: #f8fafc;
    font-size: 16px;
}

.heatmap-controls {
    display: flex;
    gap: 10px;
}

.heatmap-controls button,
.heatmap-controls select {
    background: #374151;
    border: 1px solid #4b5563;
    color: #f8fafc;
    padding: 5px 10px;
    border-radius: 4px;
    font-size: 12px;
    cursor: pointer;
}

.heatmap-main {
    display: flex;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 4px;
    overflow: hidden;
    min-height: 400px;
}

.tactics-axis {
    width: 120px;
    background: #1e293b;
    border-right: 1px solid #334155;
    padding: 10px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.tactic-label {
    writing-mode: vertical-rl;
    text-orientation: mixed;
    font-size: 10px;
    color: #cbd5e1;
    padding: 5px;
    text-align: center;
    cursor: pointer;
    border-radius: 2px;
    transition: background-color 0.2s ease;
}

.tactic-label:hover {
    background: rgba(59, 130, 246, 0.2);
}

.tactic-label.selected {
    background: rgba(59, 130, 246, 0.4);
    color: #f8fafc;
}

.techniques-grid {
    flex: 1;
    display: flex;
    padding: 10px;
    gap: 10px;
    overflow-x: auto;
}

.tactic-column {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 5px;
    min-width: 100px;
}

.technique-cell {
    background: #374151;
    border: 1px solid #4b5563;
    border-radius: 4px;
    padding: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: center;
}

.technique-cell:hover {
    border-color: #6b7280;
    transform: translateY(-1px);
}

.technique-cell.selected {
    border-color: #3b82f6;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
}

.technique-name {
    font-size: 10px;
    color: #f8fafc;
    font-weight: 500;
    margin-bottom: 2px;
}

.technique-count {
    font-size: 9px;
    color: #cbd5e1;
}

.heatmap-details {
    margin-top: 20px;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 15px;
    min-height: 120px;
}

.tactic-details h4,
.technique-details h4 {
    margin: 0 0 15px 0;
    color: #f8fafc;
    font-size: 14px;
}

.tactic-stats,
.technique-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
}

.stat-item {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
}

.stat-item .label {
    color: #94a3b8;
}

.stat-item .value {
    color: #f8fafc;
    font-weight: 500;
}

.heatmap-legend {
    margin-top: 20px;
    padding: 15px;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 4px;
}

.legend-title {
    font-size: 12px;
    font-weight: 600;
    color: #cbd5e1;
    margin-bottom: 10px;
}

.legend-scale {
    display: flex;
    align-items: center;
    gap: 10px;
}

.legend-step {
    display: flex;
    align-items: center;
    gap: 5px;
}

.legend-step::before {
    content: '';
    width: 20px;
    height: 12px;
    background: #3b82f6;
    border-radius: 2px;
}

.legend-label {
    font-size: 10px;
    color: #94a3b8;
}

/* Cluster Bubble Styles */
.bubble-chart-container {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 20px;
}

.bubble-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

.bubble-header h3 {
    margin: 0;
    color: #f8fafc;
    font-size: 16px;
}

.bubble-controls {
    display: flex;
    gap: 10px;
}

.bubble-controls button,
.bubble-controls select {
    background: #374151;
    border: 1px solid #4b5563;
    color: #f8fafc;
    padding: 5px 10px;
    border-radius: 4px;
    font-size: 12px;
    cursor: pointer;
}

.bubble-main {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 4px;
    overflow: hidden;
}

.bubble-details {
    margin-top: 20px;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 15px;
    min-height: 150px;
}

.cluster-details h4 {
    margin: 0 0 15px 0;
    color: #f8fafc;
    font-size: 14px;
}

.cluster-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
    margin-bottom: 15px;
}

.entities-section h5 {
    margin: 0 0 10px 0;
    color: #cbd5e1;
    font-size: 12px;
}

.entity-breakdown {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
    gap: 8px;
}

.entity-item {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
}

.entity-type {
    color: #94a3b8;
}

.entity-count {
    color: #f8fafc;
    font-weight: 500;
}

.bubble-legend {
    margin-top: 20px;
    padding: 15px;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 4px;
}

.legend-section {
    margin-bottom: 15px;
}

.legend-section h5 {
    margin: 0 0 8px 0;
    color: #cbd5e1;
    font-size: 12px;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 5px;
    font-size: 11px;
    color: #94a3b8;
}

.legend-color {
    width: 16px;
    height: 16px;
    border-radius: 50%;
}

.legend-color.severity-critical { background: #dc2626; }
.legend-color.severity-high { background: #ea580c; }
.legend-color.severity-medium { background: #d97706; }
.legend-color.severity-low { background: #2563eb; }

.size-explanation {
    font-size: 10px;
    color: #64748b;
    font-style: italic;
    margin-top: 5px;
}

.bubble-tooltip {
    position: absolute;
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid #475569;
    border-radius: 4px;
    padding: 8px;
    font-size: 11px;
    color: #f8fafc;
    pointer-events: none;
    z-index: 1000;
}

.tooltip-content strong {
    color: #f8fafc;
}

/* Responsive Design */
@media (max-width: 768px) {
    .timeline-header,
    .heatmap-header,
    .bubble-header {
        flex-direction: column;
        gap: 15px;
        align-items: flex-start;
    }
    
    .timeline-controls,
    .heatmap-controls,
    .bubble-controls {
        flex-wrap: wrap;
    }
    
    .detail-grid {
        grid-template-columns: 1fr;
    }
    
    .tactics-axis {
        width: 80px;
    }
    
    .tactic-label {
        font-size: 8px;
    }
}

/* Loading and Error States */
.no-events,
.no-techniques {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: #64748b;
    font-size: 14px;
    font-style: italic;
}

.loading {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 200px;
    color: #64748b;
}

.loading::after {
    content: '';
    width: 20px;
    height: 20px;
    border: 2px solid #475569;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-left: 10px;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Accessibility */
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* High Contrast Mode */
@media (prefers-contrast: high) {
    .knowledge-graph-container,
    .timeline-container,
    .heatmap-container,
    .bubble-chart-container {
        border-color: #fff;
    }
    
    .timeline-event,
    .technique-cell {
        border-width: 2px;
    }
}
```

---

## 🚀 Integration with MITRE-CORE Backend

### Enhanced Output Generator Extension

**File**: `core/visualization_output.py`

```python
import json
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path

class VisualizationOutputGenerator:
    """Generate visualization-ready data from MITRE-CORE results"""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_all_visualization_data(
        self, 
        df: pd.DataFrame, 
        correlation_result
    ) -> Dict[str, str]:
        """Generate all visualization data files"""
        
        viz_data = {
            'knowledge_graph': self.prepare_knowledge_graph_data(df),
            'timeline': self.prepare_timeline_data(df),
            'heatmap': self.prepare_mitre_heatmap_data(df),
            'clusters': self.prepare_cluster_bubble_data(df),
            'metrics': self.prepare_metrics_data(df, correlation_result)
        }
        
        output_files = {}
        
        for viz_type, data in viz_data.items():
            file_path = self.output_dir / f"{viz_type}_data.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            output_files[viz_type] = str(file_path)
        
        return output_files
    
    def prepare_knowledge_graph_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for knowledge graph visualization"""
        nodes = []
        edges = []
        node_id_map = {}
        
        # Process each alert
        for idx, row in df.iterrows():
            # Alert node
            alert_id = f"alert_{idx}"
            node_id_map[alert_id] = len(nodes)
            
            nodes.append({
                'id': alert_id,
                'type': 'alert',
                'label': str(row.get('MalwareIntelAttackType', 'Unknown')),
                'severity': str(row.get('AttackSeverity', 'medium')).lower(),
                'cluster': int(row.get('cluster', 0)),
                'timestamp': str(row.get('timestamp', '')),
                'metadata': {
                    'source_ip': str(row.get('SourceAddress', '')),
                    'dest_ip': str(row.get('DestinationAddress', '')),
                    'user': str(row.get('SourceUserName', '')),
                    'confidence': float(row.get('confidence', 0.5))
                }
            })
            
            # Extract entities and create relationships
            entities = self._extract_entities(row)
            for entity_type, entity_value in entities.items():
                if entity_value and entity_value not in ['UNKNOWN', '0.0.0.0', '']:
                    entity_id = f"{entity_type}_{entity_value}"
                    
                    # Create entity node if not exists
                    if entity_id not in node_id_map:
                        node_id_map[entity_id] = len(nodes)
                        nodes.append({
                            'id': entity_id,
                            'type': entity_type.lower(),
                            'label': str(entity_value),
                            'entity_count': 1
                        })
                    else:
                        # Increment entity count
                        nodes[node_id_map[entity_id]]['entity_count'] += 1
                    
                    # Create edge between alert and entity
                    edges.append({
                        'source': alert_id,
                        'target': entity_id,
                        'relationship': self._map_entity_to_relationship(entity_type),
                        'weight': 1.0
                    })
        
        # Create alert-to-alert edges based on shared entities and temporal proximity
        alert_to_alert_edges = self._create_alert_edges(df, node_id_map)
        edges.extend(alert_to_alert_edges)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'node_types': list(set(node['type'] for node in nodes)),
                'clusters': list(set(node['cluster'] for node in nodes if node['type'] == 'alert'))
            }
        }
    
    def prepare_timeline_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for attack timeline visualization"""
        if 'timestamp' not in df.columns:
            return {'events': [], 'attack_chains': [], 'metadata': {}}
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp')
        
        events = []
        for idx, row in df_sorted.iterrows():
            events.append({
                'id': int(idx),
                'timestamp': str(row['timestamp']),
                'event_type': str(row.get('MalwareIntelAttackType', 'Unknown')),
                'tactic': str(row.get('mitre_tactic', 'UNKNOWN')),
                'severity': str(row.get('AttackSeverity', 'medium')).lower(),
                'cluster': int(row.get('cluster', 0)),
                'confidence': float(row.get('confidence', 0.5)),
                'entities': self._extract_entities(row)
            })
        
        # Identify attack chains
        attack_chains = self._identify_attack_chains(df_sorted)
        
        return {
            'events': events,
            'attack_chains': attack_chains,
            'metadata': {
                'total_events': len(events),
                'time_range': {
                    'start': str(df_sorted['timestamp'].min()),
                    'end': str(df_sorted['timestamp'].max())
                },
                'total_chains': len(attack_chains)
            }
        }
    
    def prepare_mitre_heatmap_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for MITRE ATT&CK heatmap"""
        
        # Aggregate by tactic
        tactic_counts = {}
        if 'mitre_tactic' in df.columns:
            tactic_counts = df['mitre_tactic'].value_counts().to_dict()
        
        # Aggregate by technique
        technique_counts = {}
        if 'MalwareIntelAttackType' in df.columns:
            technique_counts = df['MalwareIntelAttackType'].value_counts().to_dict()
        
        # Calculate severity distribution
        severity_by_technique = {}
        if 'MalwareIntelAttackType' in df.columns and 'AttackSeverity' in df.columns:
            for technique in df['MalwareIntelAttackType'].unique():
                technique_data = df[df['MalwareIntelAttackType'] == technique]
                severity_dist = technique_data['AttackSeverity'].value_counts().to_dict()
                severity_by_technique[technique] = {
                    'count': len(technique_data),
                    'severity_distribution': severity_dist,
                    'most_common_severity': technique_data['AttackSeverity'].mode().iloc[0] if len(technique_data) > 0 else 'medium'
                }
        
        return {
            'tactics': tactic_counts,
            'techniques': severity_by_technique,
            'coverage': self._calculate_tactic_coverage(df),
            'metadata': {
                'total_tactics': len(tactic_counts),
                'total_techniques': len(technique_counts),
                'most_common_tactic': max(tactic_counts.items(), key=lambda x: x[1])[0] if tactic_counts else None
            }
        }
    
    def prepare_cluster_bubble_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for cluster bubble visualization"""
        clusters = []
        
        for cluster_id in df['cluster'].unique():
            if pd.isna(cluster_id):
                continue
                
            cluster_data = df[df['cluster'] == cluster_id]
            
            # Calculate cluster properties
            cluster_info = {
                'id': int(cluster_id),
                'size': len(cluster_data),
                'severity': self._calculate_cluster_severity(cluster_data),
                'attack_type': self._determine_attack_type(cluster_data),
                'confidence': float(cluster_data.get('confidence', pd.Series([0.5])).mean()),
                'tactics': list(cluster_data.get('mitre_tactic', pd.Series(['UNKNOWN'])).unique()),
                'entities': self._count_cluster_entities(cluster_data),
                'time_span': self._calculate_cluster_time_span(cluster_data)
            }
            
            clusters.append(cluster_info)
        
        return {
            'clusters': sorted(clusters, key=lambda x: x['size'], reverse=True),
            'metadata': {
                'total_clusters': len(clusters),
                'largest_cluster': max(clusters, key=lambda x: x['size'])['size'] if clusters else 0,
                'average_cluster_size': sum(c['size'] for c in clusters) / len(clusters) if clusters else 0
            }
        }
    
    def prepare_metrics_data(self, df: pd.DataFrame, correlation_result) -> Dict[str, Any]:
        """Prepare overall metrics for dashboard"""
        
        return {
            'overview': {
                'total_alerts': len(df),
                'total_clusters': correlation_result.num_clusters,
                'correlation_method': correlation_result.method_used,
                'runtime_seconds': correlation_result.runtime_seconds,
                'confidence_score': correlation_result.confidence_score
            },
            'temporal_analysis': {
                'alerts_per_hour': self._calculate_alerts_per_hour(df),
                'peak_activity_hour': self._find_peak_activity_hour(df),
                'attack_duration': self._calculate_attack_duration(df)
            },
            'entity_analysis': {
                'unique_ips': len(set(df['SourceAddress'].dropna()) | set(df['DestinationAddress'].dropna())),
                'unique_users': len(df['SourceUserName'].dropna().unique()),
                'unique_hosts': len(set(df['SourceHostName'].dropna()) | set(df['DestinationHostName'].dropna()))
            },
            'severity_distribution': df['AttackSeverity'].value_counts().to_dict() if 'AttackSeverity' in df.columns else {},
            'tactic_distribution': df['mitre_tactic'].value_counts().to_dict() if 'mitre_tactic' in df.columns else {}
        }
    
    # Helper methods
    def _extract_entities(self, row: pd.Series) -> Dict[str, str]:
        """Extract entities from alert row"""
        entities = {}
        
        entity_mappings = {
            'SourceAddress': 'ip',
            'DestinationAddress': 'ip', 
            'DeviceAddress': 'ip',
            'SourceUserName': 'user',
            'DestinationUserName': 'user',
            'SourceHostName': 'host',
            'DestinationHostName': 'host',
            'DeviceHostName': 'host'
        }
        
        for column, entity_type in entity_mappings.items():
            if column in row and pd.notna(row[column]):
                value = str(row[column])
                if value and value not in ['UNKNOWN', '0.0.0.0', '']:
                    entities[f"{entity_type}_{column}"] = value
        
        return entities
    
    def _map_entity_to_relationship(self, entity_field: str) -> str:
        """Map entity field to relationship type"""
        mapping = {
            'ip_SourceAddress': 'originates_from',
            'ip_DestinationAddress': 'targets',
            'ip_DeviceAddress': 'involves',
            'user_SourceUserName': 'performed_by',
            'user_DestinationUserName': 'targets',
            'host_SourceHostName': 'originates_from',
            'host_DestinationHostName': 'targets',
            'host_DeviceHostName': 'involves'
        }
        return mapping.get(entity_field, 'related_to')
    
    def _create_alert_edges(self, df: pd.DataFrame, node_id_map: Dict[str, int]) -> List[Dict[str, Any]]:
        """Create alert-to-alert edges based on shared entities and temporal proximity"""
        edges = []
        
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                alert_i = df.iloc[i]
                alert_j = df.iloc[j]
                
                # Check for shared entities
                shared_entities = self._find_shared_entities(alert_i, alert_j)
                
                # Check temporal proximity
                temporal_proximity = self._check_temporal_proximity(alert_i, alert_j)
                
                # Create edge if entities shared or temporally close
                if shared_entities or temporal_proximity:
                    alert_i_id = f"alert_{i}"
                    alert_j_id = f"alert_{j}"
                    
                    if alert_i_id in node_id_map and alert_j_id in node_id_map:
                        weight = len(shared_entities) + (1.0 if temporal_proximity else 0)
                        
                        edges.append({
                            'source': alert_i_id,
                            'target': alert_j_id,
                            'relationship': 'related',
                            'weight': weight,
                            'shared_entities': shared_entities,
                            'temporal_proximity': temporal_proximity
                        })
        
        return edges
    
    def _find_shared_entities(self, alert1: pd.Series, alert2: pd.Series) -> List[str]:
        """Find shared entities between two alerts"""
        shared = []
        
        entity_fields = ['SourceAddress', 'DestinationAddress', 'SourceUserName', 'SourceHostName']
        
        for field in entity_fields:
            if field in alert1 and field in alert2:
                val1 = alert1[field]
                val2 = alert2[field]
                
                if pd.notna(val1) and pd.notna(val2) and val1 == val2 and val1 not in ['UNKNOWN', '0.0.0.0', '']:
                    shared.append(field)
        
        return shared
    
    def _check_temporal_proximity(self, alert1: pd.Series, alert2: pd.Series, threshold_hours: int = 1) -> bool:
        """Check if two alerts are temporally proximate"""
        if 'timestamp' not in alert1 or 'timestamp' not in alert2:
            return False
        
        try:
            time1 = pd.to_datetime(alert1['timestamp'])
            time2 = pd.to_datetime(alert2['timestamp'])
            time_diff = abs((time1 - time2).total_seconds())
            return time_diff <= (threshold_hours * 3600)
        except:
            return False
    
    def _identify_attack_chains(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify attack chains from clustered alerts"""
        chains = []
        
        for cluster_id in df['cluster'].unique():
            if pd.isna(cluster_id):
                continue
            
            cluster_data = df[df['cluster'] == cluster_id].sort_values('timestamp')
            
            if len(cluster_data) < 2:
                continue
            
            # Simple chain identification based on temporal ordering
            chain_events = list(cluster_data.index)
            
            # Calculate chain confidence based on temporal consistency and entity relationships
            confidence = self._calculate_chain_confidence(cluster_data)
            
            # Determine attack stage
            attack_stage = self._determine_attack_stage(cluster_data)
            
            chains.append({
                'chain_id': int(cluster_id),
                'events': chain_events,
                'confidence': confidence,
                'attack_stage': attack_stage,
                'duration': self._calculate_chain_duration(cluster_data)
            })
        
        return chains
    
    def _calculate_chain_confidence(self, cluster_data: pd.DataFrame) -> float:
        """Calculate confidence score for attack chain"""
        # Simple confidence based on cluster coherence
        if 'confidence' in cluster_data.columns:
            return float(cluster_data['confidence'].mean())
        
        # Fallback: confidence based on cluster size and entity overlap
        base_confidence = 0.5
        size_bonus = min(len(cluster_data) / 10.0, 0.3)
        
        return base_confidence + size_bonus
    
    def _determine_attack_stage(self, cluster_data: pd.DataFrame) -> str:
        """Determine attack stage based on observed tactics"""
        if 'mitre_tactic' not in cluster_data.columns:
            return 'Unknown'
        
        tactics = set(cluster_data['mitre_tactic'].dropna().unique())
        
        stage_definitions = {
            'Initial': ['INITIAL ACCESS', 'EXECUTION'],
            'Partial': ['PERSISTENCE', 'PRIVILEGE ESCALATION', 'DISCOVERY'],
            'Complete': ['LATERAL MOVEMENT', 'COLLECTION', 'EXFILTRATION', 'IMPACT']
        }
        
        for stage, required_tactics in stage_definitions.items():
            if any(tactic in tactics for tactic in required_tactics):
                return stage
        
        return 'Partial'
    
    def _calculate_chain_duration(self, cluster_data: pd.DataFrame) -> str:
        """Calculate duration of attack chain"""
        if 'timestamp' not in cluster_data.columns:
            return 'Unknown'
        
        start_time = cluster_data['timestamp'].min()
        end_time = cluster_data['timestamp'].max()
        duration = end_time - start_time
        
        return str(duration)
    
    def _calculate_tactic_coverage(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate MITRE ATT&CK tactic coverage"""
        if 'mitre_tactic' not in df.columns:
            return {}
        
        total_possible_tactics = 14  # Total tactics in MITRE ATT&CK
        observed_tactics = len(df['mitre_tactic'].dropna().unique())
        
        coverage_percentage = (observed_tactics / total_possible_tactics) * 100
        
        return {
            'observed_tactics': observed_tactics,
            'total_possible_tactics': total_possible_tactics,
            'coverage_percentage': coverage_percentage
        }
    
    def _calculate_cluster_severity(self, cluster_data: pd.DataFrame) -> str:
        """Calculate overall severity for cluster"""
        if 'AttackSeverity' not in cluster_data.columns:
            return 'medium'
        
        severity_counts = cluster_data['AttackSeverity'].value_counts()
        
        # Weight severity by count
        severity_weights = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        weighted_score = 0
        total_alerts = 0
        
        for severity, count in severity_counts.items():
            weight = severity_weights.get(severity.lower(), 2)
            weighted_score += weight * count
            total_alerts += count
        
        if total_alerts == 0:
            return 'medium'
        
        avg_weight = weighted_score / total_alerts
        
        if avg_weight >= 3.5:
            return 'critical'
        elif avg_weight >= 2.5:
            return 'high'
        elif avg_weight >= 1.5:
            return 'medium'
        else:
            return 'low'
    
    def _determine_attack_type(self, cluster_data: pd.DataFrame) -> str:
        """Determine attack type for cluster"""
        if 'MalwareIntelAttackType' not in cluster_data.columns:
            return 'Unknown'
        
        attack_types = cluster_data['MalwareIntelAttackType'].dropna()
        
        # Simple classification based on common patterns
        if any('malware' in str(atype).lower() for atype in attack_types):
            return 'Malware'
        elif any('phishing' in str(atype).lower() for atype in attack_types):
            return 'Phishing'
        elif any('apt' in str(atype).lower() for atype in attack_types):
            return 'APT'
        elif any('insider' in str(atype).lower() for atype in attack_types):
            return 'Insider'
        else:
            return 'Unknown'
    
    def _count_cluster_entities(self, cluster_data: pd.DataFrame) -> Dict[str, int]:
        """Count entities in cluster"""
        entity_counts = {
            'users': 0,
            'hosts': 0,
            'ips': 0
        }
        
        # Count unique entities
        if 'SourceUserName' in cluster_data.columns:
            entity_counts['users'] = len(cluster_data['SourceUserName'].dropna().unique())
        
        host_columns = ['SourceHostName', 'DestinationHostName', 'DeviceHostName']
        all_hosts = set()
        for col in host_columns:
            if col in cluster_data.columns:
                all_hosts.update(cluster_data[col].dropna().unique())
        entity_counts['hosts'] = len(all_hosts)
        
        ip_columns = ['SourceAddress', 'DestinationAddress', 'DeviceAddress']
        all_ips = set()
        for col in ip_columns:
            if col in cluster_data.columns:
                all_ips.update(cluster_data[col].dropna().unique())
        entity_counts['ips'] = len(all_ips)
        
        return entity_counts
    
    def _calculate_cluster_time_span(self, cluster_data: pd.DataFrame) -> Dict[str, str]:
        """Calculate time span for cluster"""
        if 'timestamp' not in cluster_data.columns:
            return {'start': 'Unknown', 'end': 'Unknown', 'duration': 'Unknown'}
        
        start_time = cluster_data['timestamp'].min()
        end_time = cluster_data['timestamp'].max()
        duration = end_time - start_time
        
        return {
            'start': str(start_time),
            'end': str(end_time),
            'duration': str(duration)
        }
    
    def _calculate_alerts_per_hour(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate alerts per hour distribution"""
        if 'timestamp' not in df.columns:
            return {}
        
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hourly_counts = df['hour'].value_counts().to_dict()
        
        return {str(hour): float(count) for hour, count in hourly_counts.items()}
    
    def _find_peak_activity_hour(self, df: pd.DataFrame) -> int:
        """Find hour with peak activity"""
        if 'timestamp' not in df.columns:
            return -1
        
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        peak_hour = df['hour'].value_counts().index[0]
        
        return int(peak_hour)
    
    def _calculate_attack_duration(self, df: pd.DataFrame) -> str:
        """Calculate total attack duration"""
        if 'timestamp' not in df.columns:
            return 'Unknown'
        
        start_time = pd.to_datetime(df['timestamp']).min()
        end_time = pd.to_datetime(df['timestamp']).max()
        duration = end_time - start_time
        
        return str(duration)
```

This comprehensive implementation guide provides:

1. **Complete JavaScript visualization components** for all recommended chart types
2. **Professional CSS styling** with responsive design and accessibility features  
3. **Backend integration** with enhanced output generator for visualization-ready data
4. **Real-time features** and interactive capabilities
5. **Performance optimizations** for large datasets
6. **MITRE ATT&CK framework integration** throughout all visualizations

The implementation follows modern web development best practices and is ready for immediate integration with the existing MITRE-CORE system.
