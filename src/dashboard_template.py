DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sarah Admin | Voice Agent Control Center</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=Inter:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --primary: #38bdf8;
            --primary-glow: rgba(56, 189, 248, 0.3);
            --secondary: #818cf8;
            --text-main: #f8fafc;
            --text-dim: #94a3b8;
            --success: #4ade80;
            --danger: #f87171;
            --accent: #f472b6;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            -webkit-font-smoothing: antialiased;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            line-height: 1.5;
            overflow-x: hidden;
        }

        h1, h2, h3 {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
        }

        /* Layout */
        .app-container {
            display: grid;
            grid-template-columns: 260px 1fr;
            min-height: 100vh;
        }

        /* Sidebar */
        aside {
            background-color: #0c1120;
            border-right: 1px solid #1e293b;
            padding: 2rem 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 2rem;
            position: sticky;
            top: 0;
            height: 100vh;
        }

        .logo-area {
            display: flex;
            items-align: center;
            gap: 0.75rem;
            margin-bottom: 2rem;
        }

        .logo-icon {
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-radius: 8px;
            box-shadow: 0 0 15px var(--primary-glow);
        }

        .logo-text {
            font-size: 1.25rem;
            letter-spacing: -0.025em;
        }

        .nav-link {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 1rem;
            border-radius: 10px;
            color: var(--text-dim);
            text-decoration: none;
            transition: all 0.2s;
            cursor: pointer;
        }

        .nav-link:hover {
            color: var(--text-main);
            background: #162033;
        }

        .nav-link.active {
            color: var(--primary);
            background: #1e293b;
            font-weight: 500;
        }

        /* Main Content */
        main {
            padding: 2.5rem;
            overflow-y: auto;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 3rem;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(74, 222, 128, 0.1);
            color: var(--success);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 600;
            border: 1px solid rgba(74, 222, 128, 0.2);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--success);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--success);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.3); opacity: 0.5; }
            100% { transform: scale(1); opacity: 1; }
        }

        /* Grid Cards */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }

        .card {
            background-color: var(--bg-card);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 1.5rem;
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        .card:hover {
            transform: translateY(-4px);
            border-color: rgba(56, 189, 248, 0.2);
        }

        .card-label {
            font-size: 0.875rem;
            color: var(--text-dim);
            margin-bottom: 0.5rem;
            font-weight: 500;
        }

        .card-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-main);
            font-family: 'Outfit';
        }

        /* Terminal Logs */
        .terminal {
            background: #000;
            border-radius: 12px;
            padding: 1.5rem;
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.875rem;
            height: 450px;
            overflow-y: auto;
            border: 1px solid #1e293b;
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.5);
            margin-bottom: 2.5rem;
        }

        .log-line {
            margin-bottom: 4px;
            color: #d1d5db;
        }

        .log-info { color: var(--primary); }
        .log-warn { color: #fbbf24; }
        .log-err { color: var(--danger); }
        .log-time { color: #6b7280; margin-right: 8px; font-size: 0.8rem; }

        /* Forms */
        .section-title {
            margin-bottom: 1.5rem;
            font-size: 1.25rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .config-form {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 1024px) {
            .config-form { grid-template-columns: 1fr; }
            .app-container { grid-template-columns: 1fr; }
            aside { display: none; }
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        label {
            font-size: 0.875rem;
            color: var(--text-dim);
            font-weight: 500;
        }

        input, select {
            background: #0f172a;
            border: 1px solid #334155;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            color: var(--text-main);
            font-family: inherit;
            transition: border-color 0.2s;
        }

        input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 2px var(--primary-glow);
        }

        button {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: #fff;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 4px 12px var(--primary-glow);
        }

        button:hover {
            transform: scale(1.02);
            filter: brightness(1.1);
        }

        button:active {
            transform: scale(0.98);
        }

        /* Sessions Table */
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }

        th {
            text-align: left;
            padding: 1rem;
            color: var(--text-dim);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid #1e293b;
        }

        td {
            padding: 1.25rem 1rem;
            border-bottom: 1px solid #1e293b;
            font-size: 0.875rem;
        }

        tr:last-child td {
            border-bottom: none;
        }

        .badge {
            padding: 0.25rem 0.625rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
        }

        .badge-call { background: rgba(56, 189, 248, 0.1); color: var(--primary); }
        .badge-phone { background: rgba(129, 140, 248, 0.1); color: var(--secondary); }

        .loading-overlay {
            position: fixed;
            inset: 0;
            background: rgba(15, 23, 42, 0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            display: none;
        }

        .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid rgba(255,255,255,0.1);
            border-left-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: var(--bg-card);
            border: 1px solid var(--primary);
            padding: 1rem 1.5rem;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            transform: translateY(150%);
            transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
            z-index: 1001;
        }

        .toast.show { transform: translateY(0); }
    </style>
</head>
<body>
    <div class="app-container">
        <aside>
            <div class="logo-area">
                <div class="logo-icon"></div>
                <h1 class="logo-text">Sarah Admin</h1>
            </div>
            <nav>
                <a href="#" class="nav-link active">Dashboard</a>
                <a href="#" class="nav-link">Call Logs</a>
                <a href="#" class="nav-link">Memory Viewer</a>
                <a href="#" class="nav-link">Account Settings</a>
            </nav>
            <div style="margin-top: auto; border-top: 1px solid #1e293b; padding-top: 1.5rem;">
                <div style="font-size: 0.75rem; color: var(--text-dim);">Build v2.4.0</div>
                <div style="font-size: 0.75rem; color: var(--text-dim);">DeepMind Agentic Framework</div>
            </div>
        </aside>

        <main>
            <header>
                <div>
                    <h2 style="font-size: 1.75rem; margin-bottom: 0.25rem;">Server Overview</h2>
                    <p style="color: var(--text-dim);">Live monitoring and control of Sarah Voice Agent</p>
                </div>
                <div class="status-badge">
                    <div class="status-dot"></div>
                    System Operational
                </div>
            </header>

            <div class="stats-grid">
                <div class="card">
                    <div class="card-label">Active Calls</div>
                    <div class="card-value" id="stat-active-calls">0</div>
                </div>
                <div class="card">
                    <div class="card-label">Today's Total Sessions</div>
                    <div class="card-value" id="stat-total-sessions">12</div>
                </div>
                <div class="card">
                    <div class="card-label">Avg. Latency</div>
                    <div class="card-value">142ms</div>
                </div>
                <div class="card">
                    <div class="card-label">Agent Readiness</div>
                    <div class="card-value" id="stat-ready" style="color: var(--success);">READY</div>
                </div>
            </div>

            <!-- New Section: Manual Outbound Call -->
            <div class="section-title">Initiate Manual Call</div>
            <div class="card" style="margin-bottom: 2.5rem;">
                <div class="config-form" style="grid-template-columns: 1fr auto;">
                    <div class="form-group">
                        <label>Phone Number (with country code)</label>
                        <input type="text" id="manual-phone-number" placeholder="+1234567890">
                    </div>
                    <div style="display: flex; align-items: flex-end;">
                        <button type="button" onclick="initiateManualCall()">🚀 Start Outbound Call</button>
                    </div>
                </div>
            </div>

            <div class="section-title">Active Conversations</div>
            <div class="card" style="padding: 0; margin-bottom: 2.5rem; overflow: hidden;">
                <table>
                    <thead>
                        <tr>
                            <th>Call SID</th>
                            <th>From Number</th>
                            <th>Status</th>
                            <th>Last Transcription</th>
                            <th>Duration</th>
                        </tr>
                    </thead>
                    <tbody id="sessions-table">
                        <!-- Filled by JS -->
                        <tr>
                            <td colspan="5" style="text-align: center; color: var(--text-dim); padding: 2rem;">No active calls</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div class="section-title">System Logs</div>
            <div class="terminal" id="log-terminal">
                <div class="log-line">--- Terminal initialized ---</div>
            </div>

            <div class="section-title">Twilio & API Settings</div>
            <div class="card">
                <form id="config-form" class="config-form">
                    <div class="form-group">
                        <label>Groq API Key</label>
                        <input type="password" name="GROQ_API_KEY" id="cfg-groq" placeholder="gsk_...">
                    </div>
                    <div class="form-group">
                        <label>Twilio Account SID</label>
                        <input type="text" name="TWILIO_ACCOUNT_SID" id="cfg-tw-sid" placeholder="AC...">
                    </div>
                    <div class="form-group">
                        <label>Twilio Auth Token</label>
                        <input type="password" name="TWILIO_AUTH_TOKEN" id="cfg-tw-token" placeholder="...">
                    </div>
                    <div class="form-group">
                        <label>Twilio Phone Number</label>
                        <input type="text" name="TWILIO_PHONE_NUMBER" id="cfg-tw-phone" placeholder="+1...">
                    </div>
                    <div class="form-group">
                        <label>Server URL (Public)</label>
                        <input type="text" name="SERVER_URL" id="cfg-server-url" placeholder="https://...">
                    </div>
                    <div class="form-group">
                        <label>Barge-In Threshold (Sensitivity)</label>
                        <input type="number" name="RMS_BARGE_IN_THRESHOLD" id="cfg-barge" value="600">
                    </div>
                    <div style="grid-column: 1 / -1; display: flex; gap: 1rem; margin-top: 1rem;">
                        <button type="submit">Update Production Config</button>
                        <button type="button" style="background: transparent; border: 1px solid #1e293b; box-shadow: none;" onclick="refreshConfig()">Refresh</button>
                    </div>
                </form>
            </div>
        </main>
    </div>

    <div class="loading-overlay" id="loading">
        <div class="spinner"></div>
    </div>

    <div class="toast" id="toast">Config updated successfully</div>

    <script>
        async function initiateManualCall() {
            const phone = document.getElementById('manual-phone-number').value.trim();
            if (!phone) {
                showToast("Please enter a phone number", true);
                return;
            }

            document.getElementById('loading').style.display = 'flex';
            try {
                const res = await fetch('/dashboard/call', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ phone_number: phone })
                });
                const data = await res.json();
                
                if (data.status === 'success') {
                    showToast("Call initiated! Check logs below.");
                    document.getElementById('manual-phone-number').value = '';
                } else {
                    showToast("Failed: " + (data.message || "Unknown error"), true);
                }
            } catch (err) {
                showToast("Network error", true);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }

        async function fetchStatus() {
            try {
                const res = await fetch('/dashboard/status');
                const data = await res.json();
                
                document.getElementById('stat-active-calls').innerText = data.active_calls;
                document.getElementById('stat-ready').innerText = data.agent_ready ? 'READY' : 'STARTING';
                
                const table = document.getElementById('sessions-table');
                if (data.sessions && data.sessions.length > 0) {
                    table.innerHTML = data.sessions.map(s => `
                        <tr>
                            <td><code>${s.sid.substring(0, 12)}...</code></td>
                            <td><span class="badge badge-phone">${s.phone || 'Unknown'}</span></td>
                            <td><span class="badge badge-call">STREAMING</span></td>
                            <td style="max-width: 300px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--text-dim);">
                                ${s.last_text || '--'}
                            </td>
                            <td>${s.duration || '0s'}</td>
                        </tr>
                    `).join('');
                } else {
                    table.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-dim); padding: 2rem;">No active calls</td></tr>';
                }
            } catch (err) {
                console.error("Status fetch failed", err);
            }
        }

        async function fetchLogs() {
            try {
                const res = await fetch('/dashboard/logs');
                const logs = await res.json();
                const container = document.getElementById('log-terminal');
                
                container.innerHTML = logs.map(line => {
                    let cls = '';
                    if (line.includes('| INFO')) cls = 'log-info';
                    if (line.includes('| WARNING')) cls = 'log-warn';
                    if (line.includes('| ERROR')) cls = 'log-err';
                    
                    // Basic syntax highlighting for timestamps
                    const parts = line.split('|');
                    if (parts.length > 1) {
                        return `<div class="log-line"><span class="log-time">${parts[0].trim()}</span> | <span class="${cls}">${parts[1].trim()}</span> | ${parts.slice(2).join('|')}</div>`;
                    }
                    return `<div class="log-line">${line}</div>`;
                }).join('');
                
                // Only scroll to bottom if user isn't looking up
                if (container.scrollHeight - container.scrollTop - container.clientHeight < 50) {
                    container.scrollTop = container.scrollHeight;
                }
            } catch (err) {
                console.error("Log fetch failed", err);
            }
        }

        async function refreshConfig() {
            try {
                const res = await fetch('/dashboard/config');
                const config = await res.json();
                
                for (const [key, val] of Object.entries(config)) {
                    const el = document.querySelector(`[name="${key}"]`);
                    if (el) el.value = val;
                }
            } catch (err) {
                console.error("Config fetch failed", err);
            }
        }

        document.getElementById('config-form').onsubmit = async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const updates = Object.fromEntries(formData.entries());
            
            document.getElementById('loading').style.display = 'flex';
            
            try {
                const res = await fetch('/dashboard/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(updates)
                });
                
                if (res.ok) {
                    showToast("Configuration saved & reloading...");
                    setTimeout(() => window.location.reload(), 1500);
                } else {
                    showToast("Error updating config", true);
                }
            } catch (err) {
                showToast("Network error", true);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        };

        function showToast(msg, isErr = false) {
            const toast = document.getElementById('toast');
            toast.innerText = msg;
            toast.style.borderColor = isErr ? 'var(--danger)' : 'var(--primary)';
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 3000);
        }

        // Poll every 3 seconds
        setInterval(fetchStatus, 3000);
        setInterval(fetchLogs, 5000);
        
        // Initial load
        fetchStatus();
        fetchLogs();
        refreshConfig();
    </script>
</body>
</html>
"""
