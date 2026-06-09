# Multi-Agent A2A System Demo Web UI

We will build a premium, highly interactive, and beautiful web interface using Vite, Vanilla JS, and Vanilla CSS to demo the multi-agent interactions of Stage 4 and Stage 5 of the Codelab. 

Additionally, we will build a lightweight Python proxy backend to handle communication with the A2A SDK agents to circumvent CORS restrictions and simplify the frontend HTTP calls.

## User Review Required

> [!IMPORTANT]
> The demo UI will support two modes:
> 1. **Simulation Mode**: A step-by-step visual animation demonstrating how agents discover and delegate to each other, highlighting the exact path of execution on an interactive SVG topology graph.
> 2. **Real Mode**: Sends actual HTTP requests to the running Customer Agent (`http://localhost:10100`) via a lightweight Python proxy server. It will measure real latency and show the final responses.
>
> To use **Real Mode**, you will need to start the A2A agents (e.g. via `./start_all.sh`) and then run our proxy server `python agent_demo_ui/proxy_backend.py`.

## Proposed Changes

We will develop the frontend UI in a self-contained Vite project directory `agent_demo_ui` and provide a proxy backend script inside it.

---

### [Component: Python Proxy Backend]

#### [NEW] [proxy_backend.py](file:///d:/Project/Vin_AI/Lab%209/Batch02-Day9_Multi-Agent_MCP-A2A/agent_demo_ui/proxy_backend.py)
A FastAPI script that exposes:
- `GET /api/agents` to fetch registered agents from the registry at `http://localhost:10000/agents`.
- `POST /api/query` to send questions to the Customer Agent at `http://localhost:10100` using the `A2AClient` (same logic as `test_client.py`).
- Enables CORS to allow calls from the Vite dev server (port 5173).

---

### [Component: Vite Frontend]

#### [MODIFY] [index.html](file:///d:/Project/Vin_AI/Lab%209/Batch02-Day9_Multi-Agent_MCP-A2A/agent_demo_ui/index.html)
A modern, dark-themed index page with Google Fonts (Inter / Outfit) containing:
- Graph viewer section (SVG topology graph with nodes: Client, Customer Agent, Registry, Law Agent, Tax Agent, Compliance Agent).
- Node inspection panel (details on prompts, tools, port of selected agents).
- Live control panel with quick-select questions, input field, and Real vs Simulation toggles.
- Scrolling terminal logs console to print step-by-step agent events.
- Response display showing the final legal briefing, elapsed time, and traces.

#### [MODIFY] [src/style.css](file:///d:/Project/Vin_AI/Lab%209/Batch02-Day9_Multi-Agent_MCP-A2A/agent_demo_ui/src/style.css)
A custom, premium design system using HSL color tokens:
- Sleek dark background with smooth gradients.
- Glassmorphic panels (`backdrop-filter`).
- Keyframe animations for active agent glowing pulses (`@keyframes pulse`).
- SVG connection path animation (marching ants effect / dash-offset flow).
- Responsive layout with Flexbox and CSS Grid.

#### [MODIFY] [src/main.js](file:///d:/Project/Vin_AI/Lab%209/Batch02-Day9_Multi-Agent_MCP-A2A/agent_demo_ui/src/main.js)
State management and interaction logic:
- Polling mechanism to check agent status from `/api/agents` and show green/red lights.
- Interactive graph click handlers to display metadata card for each agent.
- Visual flow orchestration: lighting up nodes and trigger connection pulses step-by-step.
- Integration with both the Mock Simulator and the real endpoint.
- Printing formatted logs to the custom terminal log container.

## Verification Plan

### Automated/Manual Verification
1. **Frontend Dev Server**: Run `npm run dev` in `agent_demo_ui` and view in a browser.
2. **Registry and Agent Connection Check**: Check that registry polling displays online/offline status correctly based on whether `start_all.sh` is running.
3. **Simulation Test**: Click a query in "Simulation Mode" and verify the step-by-step path animations (Registry query, Law Agent delegation, parallel Tax/Compliance calls, aggregation).
4. **Real Query Test**: Start all agents via `./start_all.sh`, run `python agent_demo_ui/proxy_backend.py`, choose "Real Mode" in the web UI, and verify that actual LLM responses are rendered and latency is reported.
