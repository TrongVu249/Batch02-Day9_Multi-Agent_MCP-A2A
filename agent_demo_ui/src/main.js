import './style.css'

// Agent card data for inspector
const AGENTS_METADATA = {
  client: {
    name: 'Client / User',
    port: 'N/A',
    description: 'The initiating end-user of the system topology. Submits legal questions to the Customer Agent and receives the aggregated and synthesized legal advice.',
    prompt: 'N/A',
    skills: ['Initiates user requests', 'Traces message routing via context IDs'],
    tags: ['user', 'frontend']
  },
  customer: {
    name: 'Customer Agent',
    port: '10100',
    description: 'The front-desk entry point for all legal inquiries. In Stage 5, it accepts questions, queries the Registry to locate the Law Agent, and delegates to the Law Agent.',
    prompt: `You are a helpful legal assistant at the front desk of a multi-agent legal services platform. Your job is to:
1. Understand the user's legal question
2. Determine if it needs specialist legal analysis (contract issues, tax law, regulatory compliance, corporate liability, etc.)
3. If so, use the \`delegate_to_legal_agent\` tool to send it to the Law Agent.`,
    skills: ['Entry point legal routing', 'A2A Client communication', 'Name-based service discovery'],
    tags: ['customer', 'entry-point', 'assistant']
  },
  registry: {
    name: 'A2A Registry',
    port: '10000',
    description: 'A shared central service registry. Agents self-register their name, endpoint, tasks, and tags on startup. Other agents query the registry dynamically to resolve endpoints for target skills (e.g. legal_question, tax_question, compliance_question).',
    prompt: 'N/A',
    skills: ['Service Registration (POST /register)', 'Service Discovery (GET /discover/{task})', 'Heartbeat/Health Check (GET /health)'],
    tags: ['registry', 'infra', 'discovery']
  },
  law: {
    name: 'Law Agent',
    port: '10101',
    description: 'The orchestrator and brain of the multi-agent system. Performs corporate/contract law analysis. Evaluates the question to decide if tax and/or compliance analysis is required. Dispatches parallel delegation tasks to the Tax and Compliance Agents, then aggregates their analyses using an LLM to output a cohesive summary briefing.',
    prompt: `You are a senior corporate litigation attorney specialising in contract law, tort law, and general business law. Analyze the legal aspects of the question. Combine the contract, tax, and compliance analyses into a cohesive answer under 250 words. End with a legal disclaimer.`,
    skills: ['Contract & General Corporate Law', 'Orchestration & Parallel Routing', 'Multi-Agent Result Aggregation'],
    tags: ['law', 'orchestration', 'senior-counsel']
  },
  tax: {
    name: 'Tax Agent',
    port: '10102',
    description: 'A specialized domain agent that handles tax issues. Analyzes questions regarding tax liabilities, IRS code regulations, tax evasion penalties, FBAR/FATCA filings, and corporate tax implications.',
    prompt: `You are a specialist tax attorney and CPA. Analyze the tax law aspects of the question. Focus on IRS code violations, liabilities, and penalty brackets. Keep response under 150 words.`,
    skills: ['IRS Code Analysis', 'Tax Avoidance/Evasion Defense', 'Corporate Tax Liability Assessment'],
    tags: ['tax', 'specialist', 'cpa']
  },
  compliance: {
    name: 'Compliance Agent',
    port: '10103',
    description: 'A specialized domain agent that handles corporate compliance, SEC regulations, SOX audits, FCPA issues, anti-money laundering (AML) laws, GDPR/CCPA data privacy regulations, and governance policies.',
    prompt: `You are a regulatory compliance officer. Analyze compliance requirements (SEC, SOX, AML, GDPR, CCPA). Focus on fines, audit requirements, and corporate compliance controls. Keep response under 150 words.`,
    skills: ['SEC/SOX Regulatory Audit', 'Anti-Money Laundering (AML) controls', 'Data Privacy (GDPR/CCPA) Compliance'],
    tags: ['compliance', 'specialist', 'regulatory']
  }
}

// DOM Elements
const customQuestion = document.getElementById('custom-question');
const runBtn = document.getElementById('run-btn');
const logConsole = document.getElementById('log-console');
const clearLogsBtn = document.getElementById('clear-logs');
const modeSimBtn = document.getElementById('mode-sim');
const modeRealBtn = document.getElementById('mode-real');
const modeHelper = document.getElementById('mode-helper');
const inspectorContent = document.getElementById('inspector-content');
const responseOutput = document.getElementById('response-output');
const metaTime = document.getElementById('meta-time');
const metaTrace = document.getElementById('meta-trace');

// State
let executionMode = 'sim'; // 'sim' or 'real'
let simulatedTimeoutIds = [];
let registryPollingInterval = null;

// Proxy API URL
const PROXY_URL = 'http://localhost:8000';

// Initialize
function init() {
  setupEventListeners();
  startRegistryPolling();
  selectDefaultNode();
}

// Setup event listeners
function setupEventListeners() {
  // Graph clicks
  document.querySelectorAll('.graph-node').forEach(node => {
    node.addEventListener('click', () => {
      const agentId = node.getAttribute('data-agent');
      showInspector(agentId);
      highlightNode(node);
    });
  });

  // Mode Toggles
  modeSimBtn.addEventListener('click', () => setMode('sim'));
  modeRealBtn.addEventListener('click', () => setMode('real'));

  // Quick Questions
  document.querySelectorAll('.q-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      customQuestion.value = btn.getAttribute('data-question');
    });
  });

  // Run Button
  runBtn.addEventListener('click', () => {
    runOrchestration();
  });

  // Clear Logs
  clearLogsBtn.addEventListener('click', () => {
    logConsole.innerHTML = '';
  });
}

function setMode(mode) {
  executionMode = mode;
  if (mode === 'sim') {
    modeSimBtn.classList.add('active');
    modeRealBtn.classList.remove('active');
    modeHelper.textContent = 'Simulated Mode plays a visual step-by-step walkthrough of the delegation pipeline. Perfect for demoing the Stage 5 flow without running backends.';
  } else {
    modeRealBtn.classList.add('active');
    modeSimBtn.classList.remove('active');
    modeHelper.textContent = 'Real Mode calls the actual Python A2A agents on localhost via proxy backend. Requires agents to be running and uvicorn proxy running on port 8000.';
  }
}

// Select Client node on start
function selectDefaultNode() {
  const clientNode = document.getElementById('node-client');
  if (clientNode) {
    highlightNode(clientNode);
    showInspector('client');
  }
}

// Highlight node in graph
function highlightNode(nodeElement) {
  document.querySelectorAll('.graph-node').forEach(n => n.classList.remove('active-node'));
  nodeElement.classList.add('active-node');
}

// Show details in inspector card
function showInspector(agentId) {
  const data = AGENTS_METADATA[agentId];
  if (!data) return;

  const skillsHtml = data.skills.map(s => `<li>${s}</li>`).join('');
  const tagsHtml = data.tags.map(t => `<span class="tag-item">${t}</span>`).join('');
  
  let promptSection = '';
  if (data.prompt !== 'N/A') {
    promptSection = `
      <div class="inspector-section">
        <h4>System Instruction / Prompt</h4>
        <div class="inspector-prompt">${escapeHtml(data.prompt)}</div>
      </div>
    `;
  }

  inspectorContent.innerHTML = `
    <div class="inspector-header">
      <span class="icon">${getAgentEmoji(agentId)}</span>
      <div>
        <h3>${data.name}</h3>
        <span class="port">${data.port === 'N/A' ? 'No Port' : `Port ${data.port}`}</span>
      </div>
    </div>
    <p class="inspector-desc">${data.description}</p>
    
    ${promptSection}

    <div class="inspector-section">
      <h4>Capabilities & Skills</h4>
      <ul style="padding-left: 1.25rem; font-size: 0.85rem; color: var(--color-text); line-height: 1.4; display: flex; flex-direction: column; gap: 0.25rem;">
        ${skillsHtml}
      </ul>
    </div>

    <div class="inspector-section">
      <h4>Metadata Tags</h4>
      <div class="tag-list">
        ${tagsHtml}
      </div>
    </div>
  `;
}

function getAgentEmoji(agentId) {
  switch (agentId) {
    case 'client': return '👤';
    case 'customer': return '🤖';
    case 'registry': return '🗄️';
    case 'law': return '⚖️';
    case 'tax': return '💸';
    case 'compliance': return '🛡️';
    default: return '⚙️';
  }
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Log Terminal updates
function addLogEntry(text, type = 'system') {
  const timeStr = new Date().toLocaleTimeString();
  const entry = document.createElement('div');
  entry.className = `log-entry ${type}`;
  entry.innerHTML = `[${timeStr}] ${text}`;
  logConsole.appendChild(entry);
  logConsole.scrollTop = logConsole.scrollHeight;
}

// Poll Registry and Agent status
async function startRegistryPolling() {
  pollRegistryStatus();
  registryPollingInterval = setInterval(pollRegistryStatus, 4000);
}

async function pollRegistryStatus() {
  try {
    const res = await fetch(`${PROXY_URL}/api/agents`);
    if (!res.ok) throw new Error('Proxy error');
    
    const data = await res.json();
    
    if (data.error) {
      // Registry down
      updateNodeStatus('registry', false);
      updateNodeStatus('customer', false);
      updateNodeStatus('law', false);
      updateNodeStatus('tax', false);
      updateNodeStatus('compliance', false);
      return;
    }

    // Registry up
    updateNodeStatus('registry', true);
    
    const registeredNames = data.agents.map(a => a.agent_name);
    
    updateNodeStatus('customer', registeredNames.includes('customer-agent'));
    updateNodeStatus('law', registeredNames.includes('law-agent'));
    updateNodeStatus('tax', registeredNames.includes('tax-agent'));
    updateNodeStatus('compliance', registeredNames.includes('compliance-agent'));

  } catch (err) {
    // Proxy down or connection error
    updateNodeStatus('registry', false);
    updateNodeStatus('customer', false);
    updateNodeStatus('law', false);
    updateNodeStatus('tax', false);
    updateNodeStatus('compliance', false);
  }
}

function updateNodeStatus(agentId, online) {
  const el = document.getElementById(`status-${agentId}`);
  if (!el) return;
  
  if (online) {
    el.className.baseVal = 'status-indicator online';
  } else {
    el.className.baseVal = 'status-indicator offline';
  }
}

// Execute Orchestration
function runOrchestration() {
  // Clear previous timers
  simulatedTimeoutIds.forEach(clearTimeout);
  simulatedTimeoutIds = [];

  const question = customQuestion.value.trim();
  if (!question) {
    alert('Please enter or select a question first.');
    return;
  }

  // Disable button during execution
  runBtn.disabled = true;
  runBtn.textContent = 'Executing Orchestration...';
  
  // Reset outputs
  responseOutput.innerHTML = `<p class="empty-answer">Processing question... Please wait.</p>`;
  metaTime.textContent = 'Time: -';
  metaTrace.textContent = 'Trace ID: -';
  
  // Reset graph visual states
  resetGraphVisState();

  if (executionMode === 'sim') {
    runSimulation(question);
  } else {
    runRealQuery(question);
  }
}

function resetGraphVisState() {
  document.querySelectorAll('.flow-path').forEach(el => el.classList.remove('flow-active'));
  document.querySelectorAll('.graph-node').forEach(el => el.classList.remove('working-node'));
  document.querySelectorAll('.status-indicator').forEach(el => {
    if (el.className.baseVal.includes('working')) {
      el.className.baseVal = el.className.baseVal.replace('working', 'online');
    }
  });
}

function setNodeWorking(agentId, isWorking) {
  const node = document.getElementById(`node-${agentId}`);
  const status = document.getElementById(`status-${agentId}`);
  
  if (isWorking) {
    if (node) node.classList.add('working-node');
    if (status && status.className.baseVal.includes('online')) {
      status.className.baseVal = 'status-indicator working';
    }
  } else {
    if (node) node.classList.remove('working-node');
    if (status && status.className.baseVal.includes('working')) {
      status.className.baseVal = 'status-indicator online';
    }
  }
}

function setFlowActive(flowId, isActive) {
  const flow = document.getElementById(`flow-${flowId}`);
  if (!flow) return;
  if (isActive) {
    flow.classList.add('flow-active');
  } else {
    flow.classList.remove('flow-active');
  }
}

// Simulation Flow
function runSimulation(question) {
  addLogEntry('🎬 Starting simulation mode...', 'system');
  const startTime = Date.now();
  const traceId = 'sim-tr-' + Math.random().toString(36).substring(2, 9);
  
  const questionLower = question.toLowerCase();
  const needsTax = ["tax", "irs", "evasion", "avoidance", "irc", "thuế"].some(kw => questionLower.includes(kw));
  const needsCompliance = ["compliance", "regulatory", "sec", "sox", "aml", "fcpa", "regulation", "luật quy định"].some(kw => questionLower.includes(kw));
  
  // If neither is true, both are true as fallback
  const runTax = needsTax || (!needsTax && !needsCompliance);
  const runCompliance = needsCompliance || (!needsTax && !needsCompliance);

  function scheduleStep(delay, fn) {
    const id = setTimeout(fn, delay);
    simulatedTimeoutIds.push(id);
  }

  // Step 1: User -> Customer Agent
  scheduleStep(0, () => {
    addLogEntry(`👤 Client initiating request: "${question.substring(0, 60)}..."`, 'system');
    setFlowActive('user-customer', true);
    setNodeWorking('customer', true);
    addLogEntry('🤖 Customer Agent received request. Context ID generated.', 'customer');
  });

  // Step 2: Customer Agent -> Registry
  scheduleStep(1000, () => {
    addLogEntry('🤖 Customer Agent querying Registry for task: "legal_question"...', 'customer');
    setFlowActive('customer-registry', true);
    setNodeWorking('registry', true);
  });

  // Step 3: Registry -> Customer Agent
  scheduleStep(2000, () => {
    setFlowActive('customer-registry', false);
    setNodeWorking('registry', false);
    addLogEntry('🗄️ Registry resolved "legal_question" to Law Agent (http://localhost:10101)', 'registry');
  });

  // Step 4: Customer Agent -> Law Agent
  scheduleStep(2800, () => {
    addLogEntry('🤖 Customer Agent delegating query to Law Agent (Port 10101)...', 'customer');
    setFlowActive('customer-law', true);
    setNodeWorking('law', true);
  });

  // Step 5: Law Agent executes rule-based check
  scheduleStep(4000, () => {
    setFlowActive('user-customer', false);
    setFlowActive('customer-law', false);
    setNodeWorking('customer', false);
    
    addLogEntry('⚖️ Law Agent received request. Parsing routing criteria...', 'law');
    addLogEntry('⚖️ Law Agent starting general contract law analysis node: "analyze_law".', 'law');
    
    let routingLog = '⚖️ Law Agent checking routing conditions: ';
    if (runTax && runCompliance) {
      routingLog += 'Detected both Tax and Compliance references (or default fallback). Routing to BOTH sub-agents.';
    } else if (runTax) {
      routingLog += 'Detected Tax reference. Routing to Tax Agent ONLY.';
    } else {
      routingLog += 'Detected Compliance reference. Routing to Compliance Agent ONLY.';
    }
    addLogEntry(routingLog, 'law');
  });

  // Step 6: Law Agent queries Registry for sub-agents
  scheduleStep(5000, () => {
    const tasks = [];
    if (runTax) tasks.push('"tax_question"');
    if (runCompliance) tasks.push('"compliance_question"');
    
    addLogEntry(`⚖️ Law Agent querying Registry for endpoints: ${tasks.join(', ')}...`, 'law');
    setFlowActive('law-registry', true);
    setNodeWorking('registry', true);
  });

  // Step 7: Registry -> Law Agent
  scheduleStep(6000, () => {
    setFlowActive('law-registry', false);
    setNodeWorking('registry', false);
    
    let resolvedLog = '🗄️ Registry resolved: ';
    const resolutions = [];
    if (runTax) resolutions.push('tax_question -> Tax Agent (Port 10102)');
    if (runCompliance) resolutions.push('compliance_question -> Compliance Agent (Port 10103)');
    addLogEntry(resolvedLog + resolutions.join(', '), 'registry');
  });

  // Step 8: Law Agent dispatches parallel calls
  scheduleStep(6800, () => {
    let dispatchLog = '⚖️ Law Agent dispatching parallel calls: ';
    const targets = [];
    if (runTax) {
      setFlowActive('law-tax', true);
      setNodeWorking('tax', true);
      targets.push('Tax Agent');
    }
    if (runCompliance) {
      setFlowActive('law-compliance', true);
      setNodeWorking('compliance', true);
      targets.push('Compliance Agent');
    }
    addLogEntry(dispatchLog + targets.join(' & ') + ' along with local LLM analysis.', 'law');
  });

  // Step 9: Tax / Compliance execute and return results
  scheduleStep(8500, () => {
    if (runTax) {
      addLogEntry('💸 Tax Agent completed IRS and tax avoidance analysis. Returning briefing.', 'tax');
      setFlowActive('law-tax', false);
      setNodeWorking('tax', false);
    }
    if (runCompliance) {
      addLogEntry('🛡️ Compliance Agent completed regulatory and SEC compliance analysis. Returning briefing.', 'compliance');
      setFlowActive('law-compliance', false);
      setNodeWorking('compliance', false);
    }
  });

  // Step 10: Law Agent aggregates
  scheduleStep(9500, () => {
    addLogEntry('⚖️ Law Agent received sub-agent responses. Synthesizing combined advice via LLM...', 'law');
  });

  // Step 11: Return final answer to Customer, then Client
  scheduleStep(11500, () => {
    setNodeWorking('law', false);
    setNodeWorking('customer', true);
    setFlowActive('customer-law', true);
    addLogEntry('🤖 Customer Agent received final response from Law Agent and is returning it to client.', 'customer');
  });

  // Step 12: Final rendering
  scheduleStep(12500, () => {
    setNodeWorking('customer', false);
    setFlowActive('customer-law', false);
    
    // Generate simulated markdown response based on prompt
    const responseHtml = generateSimulatedMarkdown(question, runTax, runCompliance);
    responseOutput.innerHTML = responseHtml;
    
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
    metaTime.textContent = `Time: ${elapsed}s`;
    metaTrace.textContent = `Trace ID: ${traceId}`;
    
    addLogEntry('✅ Orchestration finished successfully.', 'system');
    
    runBtn.disabled = false;
    runBtn.textContent = 'Run Query Orchestration';
  });
}

function generateSimulatedMarkdown(question, runTax, runCompliance) {
  let sections = '';
  
  sections += `
    <h2>1. General Legal Counsel Assessment</h2>
    <p>We have analyzed your corporate litigation inquiry regarding the actions described. Under standard commercial contract law, the breach of signed agreements constitutes a material default, giving the non-breaching party grounds for full contract termination, recovery of expectation damages, and potential injunctive relief depending on the asset specificity. The corporate entity is liable for these damages directly under standard business liability rules.</p>
  `;

  if (runTax) {
    sections += `
      <hr>
      <h2>2. Specialist Tax Law Briefing</h2>
      <p>Under the Internal Revenue Code (IRC), intentional mischaracterization or omission of income to avoid tax obligations constitutes tax avoidance or evasion. Under Section 7201, willful evasion is a felony carrying severe monetary fines up to $500,000 for corporations plus prosecution costs, and civil fraud penalties under Section 6663 can impose a 75% surcharge on the tax underpayment amount. We recommend an immediate audit check.</p>
    `;
  }

  if (runCompliance) {
    sections += `
      <hr>
      <h2>3. Regulatory & Compliance Evaluation</h2>
      <p>From a compliance posture, the activities trigger immediate reporting rules under SEC whistleblower guidelines and corporate compliance covenants (Sarbanes-Oxley Act / SOX Section 302/404). In addition, if this involves international business, FCPA (Foreign Corrupt Practices Act) compliance checks must be run. Penalties for compliance control lapses can lead to suspension of trading, board director disqualifications, and direct monitoring oversight.</p>
    `;
  }

  sections += `
    <hr>
    <p style="font-size: 0.8rem; color: var(--color-text-muted); font-style: italic; margin-top: 1rem;">
      Disclaimer: This briefing is provided for educational and illustrative purposes only. It is not formal legal advice, nor does it establish an attorney-client relationship. Please consult a licensed professional attorney regarding your specific situation.
    </p>
  `;

  return `<div class="legal-briefing">${sections}</div>`;
}

// Real Mode execution via Proxy
async function runRealQuery(question) {
  addLogEntry('🔌 Connecting to local proxy backend to start real A2A routing...', 'system');
  addLogEntry(`👤 Sending HTTP POST query for client to http://localhost:8000/api/query`, 'system');
  
  // Keep track of active nodes visually as a fallback
  setNodeWorking('customer', true);
  setFlowActive('user-customer', true);

  try {
    const res = await fetch(`${PROXY_URL}/api/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ question })
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || 'Proxy server returned error');
    }

    const data = await res.json();

    if (data.status === 'error') {
      addLogEntry(`❌ Error: ${data.detail}`, 'error');
      responseOutput.innerHTML = `<p class="empty-answer" style="color: var(--color-red); font-weight: bold;">Error: ${escapeHtml(data.detail)}</p>`;
      metaTime.textContent = 'Time: -';
      metaTrace.textContent = 'Trace ID: -';
    } else {
      addLogEntry('🤖 Customer Agent completed routing successfully.', 'customer');
      addLogEntry('✅ Real briefing received.', 'system');
      
      // Parse markdown to simple HTML or display it as-is
      responseOutput.innerHTML = `<div class="legal-briefing">${formatMarkdown(data.answer)}</div>`;
      metaTime.textContent = `Time: ${data.elapsed_seconds}s`;
      metaTrace.textContent = `Trace ID: ${data.trace_id}`;
    }
  } catch (err) {
    addLogEntry(`❌ Failed to connect to proxy or agent: ${err.message}`, 'error');
    addLogEntry('ℹ️ Make sure uvicorn is running: "python agent_demo_ui/proxy_backend.py" and all A2A agents are started.', 'system');
    responseOutput.innerHTML = `<p class="empty-answer" style="color: var(--color-red); font-weight: bold;">Failed to connect: ${escapeHtml(err.message)}</p>`;
  } finally {
    setNodeWorking('customer', false);
    setFlowActive('user-customer', false);
    runBtn.disabled = false;
    runBtn.textContent = 'Run Query Orchestration';
  }
}

// Simple markdown conversion helper
function formatMarkdown(text) {
  if (!text) return '';
  let html = escapeHtml(text);
  
  // Format headings
  html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
  html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
  
  // Format bold
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  
  // Format linebreaks / dividers
  html = html.replace(/^\-\-\-/gm, '<hr>');
  
  // Convert newlines to paragraphs/breaks
  html = html.replace(/\n\n/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');
  
  return `<p>${html}</p>`;
}

// Start app
init();
