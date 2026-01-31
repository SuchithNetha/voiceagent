"""
Arya Voice Agent Dashboard Templates.

PUBLIC_DASHBOARD: Open to all users - Make calls, view live transcripts, learn about Arya
ADMIN_DASHBOARD: Super Admin only - Full system control, logs, configuration
LOGIN_HTML: Admin login page
"""

PUBLIC_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arya | AI Real Estate Agent</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #070a13;
            --bg-card: #111827;
            --primary: #38bdf8;
            --primary-glow: rgba(56, 189, 248, 0.3);
            --secondary: #818cf8;
            --text-main: #f8fafc;
            --text-dim: #94a3b8;
            --success: #4ade80;
            --border: rgba(255, 255, 255, 0.08);
            --gradient: linear-gradient(135deg, #38bdf8, #818cf8);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Inter', sans-serif; background: var(--bg-dark); color: var(--text-main); min-height: 100vh; }
        h1, h2, h3 { font-family: 'Outfit', sans-serif; }

        /* Header */
        header {
            display: flex; justify-content: space-between; align-items: center;
            padding: 1.5rem 4rem; border-bottom: 1px solid var(--border);
            position: sticky; top: 0; background: rgba(7,10,19,0.9); backdrop-filter: blur(12px); z-index: 100;
        }
        .logo { display: flex; align-items: center; gap: 0.75rem; }
        .logo-icon { width: 36px; height: 36px; background: var(--gradient); border-radius: 10px; }
        .logo h1 { font-size: 1.5rem; font-weight: 600; }
        .admin-btn {
            background: transparent; border: 1px solid var(--border); color: var(--text-dim);
            padding: 0.6rem 1.2rem; border-radius: 8px; font-size: 0.85rem; cursor: pointer; transition: 0.2s;
        }
        .admin-btn:hover { border-color: var(--primary); color: var(--primary); }

        /* Main Layout */
        .main-container { max-width: 1400px; margin: 0 auto; padding: 3rem 4rem; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 2.5rem; }
        @media (max-width: 1000px) { .grid-2 { grid-template-columns: 1fr; } header { padding: 1rem 2rem; } .main-container { padding: 2rem; } }

        /* Cards */
        .card {
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: 20px; overflow: hidden;
        }
        .card-header {
            padding: 1.5rem 2rem; border-bottom: 1px solid var(--border);
            display: flex; justify-content: space-between; align-items: center;
        }
        .card-body { padding: 2rem; }

        /* Hero Call Section */
        .hero-card {
            background: linear-gradient(145deg, #111827, #0c1120);
            border: 1px solid var(--primary-glow);
        }
        .hero-card h2 { font-size: 1.8rem; margin-bottom: 0.5rem; }
        .hero-card p { color: var(--text-dim); font-size: 0.95rem; }

        /* Form */
        .form-group { margin-bottom: 1.5rem; }
        label { display: block; font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
        input {
            width: 100%; background: #080d1a; border: 1px solid #1e293b;
            padding: 1rem 1.2rem; border-radius: 12px; color: white; font-size: 1rem; outline: none; transition: 0.2s;
        }
        input:focus { border-color: var(--primary); box-shadow: 0 0 0 3px var(--primary-glow); }
        .btn-primary {
            width: 100%; background: var(--gradient); border: none; color: white;
            padding: 1rem; border-radius: 12px; font-size: 1rem; font-weight: 700;
            cursor: pointer; transition: 0.2s; display: flex; align-items: center; justify-content: center; gap: 0.5rem;
        }
        .btn-primary:hover { filter: brightness(1.1); transform: translateY(-2px); box-shadow: 0 8px 25px var(--primary-glow); }
        .btn-primary:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }

        /* Status Badge */
        .badge { display: inline-flex; align-items: center; gap: 0.3rem; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700; }
        .badge-success { background: rgba(74, 222, 128, 0.15); color: var(--success); }
        .badge-info { background: rgba(56, 189, 248, 0.15); color: var(--primary); }

        /* About Section */
        .about-section { margin-top: 3rem; }
        .about-section h2 { font-size: 1.5rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
        .feature-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin-top: 1.5rem; }
        .feature-item {
            background: var(--bg-card); border: 1px solid var(--border);
            padding: 1.5rem; border-radius: 16px; transition: 0.2s;
        }
        .feature-item:hover { border-color: var(--primary-glow); transform: translateY(-2px); }
        .feature-item h4 { font-size: 1rem; margin-bottom: 0.5rem; color: var(--primary); }
        .feature-item p { font-size: 0.85rem; color: var(--text-dim); line-height: 1.6; }

        /* Instructions */
        .instructions { background: rgba(56, 189, 248, 0.05); border: 1px solid var(--primary-glow); border-radius: 16px; padding: 1.5rem 2rem; margin-top: 2rem; }
        .instructions h3 { font-size: 1rem; color: var(--primary); margin-bottom: 1rem; }
        .instructions ol { padding-left: 1.2rem; }
        .instructions li { color: var(--text-dim); font-size: 0.9rem; margin-bottom: 0.75rem; line-height: 1.5; }

        /* Live Transcript */
        .transcript-box {
            background: #000; border: 1px solid #1e293b; border-radius: 12px;
            height: 300px; overflow-y: auto; padding: 1.5rem; font-family: 'Fira Code', monospace; font-size: 0.85rem;
        }
        .transcript-empty { color: var(--text-dim); opacity: 0.5; text-align: center; padding: 4rem 0; }
        .turn { margin-bottom: 1rem; }
        .turn-role { font-size: 0.65rem; text-transform: uppercase; font-weight: 800; margin-bottom: 0.3rem; }
        .turn-role.user { color: #818cf8; }
        .turn-role.agent { color: #38bdf8; }
        .turn-content { color: #e2e8f0; line-height: 1.5; }

        /* Footer */
        footer { text-align: center; padding: 2rem; color: var(--text-dim); font-size: 0.8rem; border-top: 1px solid var(--border); margin-top: 4rem; }
    </style>
</head>
<body>
    <header>
        <div class="logo">
            <div class="logo-icon"></div>
            <h1>Arya</h1>
        </div>
        <button class="admin-btn" onclick="window.location.href='/login'">🔐 Admin Login</button>
    </header>

    <div class="main-container">
        <div class="grid-2">
            <!-- Left: Call Initiator -->
            <div>
                <div class="card hero-card">
                    <div class="card-body">
                        <h2>📞 Talk to Arya</h2>
                        <p style="margin-bottom: 2rem">Enter your phone number and Arya will call you to help find your dream property in Madrid.</p>
                        
                        <div class="form-group">
                            <label>Your Phone Number</label>
                            <input type="tel" id="phone-input" placeholder="+91 98765 43210" style="font-size: 1.1rem; letter-spacing: 0.05em;">
                        </div>
                        
                        <button class="btn-primary" id="call-btn" onclick="initiateCall()">
                            <span>🚀</span> Call Me Now
                        </button>
                        
                        <div id="call-status" style="text-align: center; margin-top: 1.5rem; font-size: 0.9rem; min-height: 1.5rem;"></div>
                    </div>
                </div>

                <div class="instructions">
                    <h3>📋 How It Works</h3>
                    <ol>
                        <li><strong>Enter your phone number</strong> with country code (e.g., +91 for India)</li>
                        <li><strong>Click "Call Me Now"</strong> — Arya will call you within seconds</li>
                        <li><strong>Speak naturally</strong> — Tell Arya what kind of property you're looking for</li>
                        <li><strong>Get recommendations</strong> — Arya will search and describe matching properties</li>
                    </ol>
                </div>
            </div>

            <!-- Right: Live Transcript -->
            <div>
                <div class="card">
                    <div class="card-header">
                        <h3 style="font-size: 1.1rem">🎙️ Live Conversation</h3>
                        <div id="live-status" class="badge badge-info">Waiting...</div>
                    </div>
                    <div class="card-body" style="padding: 0;">
                        <div class="transcript-box" id="transcript-box">
                            <div class="transcript-empty">
                                Start a call to see the live conversation here...
                            </div>
                        </div>
                    </div>
                </div>

                <div style="margin-top: 1.5rem; padding: 1rem 1.5rem; background: var(--bg-card); border-radius: 12px; border: 1px solid var(--border);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.85rem; color: var(--text-dim);">Active Calls Right Now</span>
                        <span id="active-count" style="font-size: 1.5rem; font-weight: 700; color: var(--success);">0</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- About Section -->
        <section class="about-section">
            <h2>✨ Meet Arya — Your AI Real Estate Agent</h2>
            <p style="color: var(--text-dim); max-width: 700px; line-height: 1.7;">
                Arya is an advanced AI-powered voice agent specializing in luxury real estate in Madrid. 
                She understands natural conversation, remembers your preferences, and helps you discover 
                properties that match your lifestyle — all through a simple phone call.
            </p>

            <div class="feature-list">
                <div class="feature-item">
                    <h4>🗣️ Natural Conversations</h4>
                    <p>Speak naturally like you would with a human agent. Arya understands context, follow-up questions, and remembers what you discussed.</p>
                </div>
                <div class="feature-item">
                    <h4>🔍 Smart Property Search</h4>
                    <p>Describe what you want — "a modern 2-bedroom with a terrace" — and Arya searches through premium Madrid listings instantly.</p>
                </div>
                <div class="feature-item">
                    <h4>🧠 Personalized Memory</h4>
                    <p>Call back anytime. Arya remembers your preferences, past conversations, and refines recommendations over time.</p>
                </div>
                <div class="feature-item">
                    <h4>⚡ Real-Time Response</h4>
                    <p>No waiting, no hold music. Arya responds in milliseconds with accurate property information and availability.</p>
                </div>
            </div>
        </section>
    </div>

    <footer>
        <p>Powered by Arya Voice Agent • Built with ❤️ for the future of real estate</p>
    </footer>

    <script>
        let pollingInterval = null;

        async function initiateCall() {
            const phone = document.getElementById('phone-input').value.trim();
            const btn = document.getElementById('call-btn');
            const status = document.getElementById('call-status');
            
            if (!phone) {
                status.innerHTML = '<span style="color: #f87171">Please enter a valid phone number</span>';
                return;
            }
            
            btn.disabled = true;
            btn.innerHTML = '🛰️ Connecting...';
            status.innerHTML = '<span style="color: var(--text-dim)">Initiating call...</span>';
            
            try {
                const res = await fetch('/api/call', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ phone_number: phone })
                });
                const data = await res.json();
                
                if (data.status === 'success') {
                    status.innerHTML = '<span style="color: var(--success)">✅ Call initiated! Arya will ring you shortly...</span>';
                    document.getElementById('live-status').innerText = 'Connecting...';
                    document.getElementById('live-status').className = 'badge badge-success';
                    startPolling();
                } else {
                    status.innerHTML = `<span style="color: #f87171">${data.message || 'Failed to initiate call'}</span>`;
                }
            } catch (e) {
                status.innerHTML = '<span style="color: #f87171">Server unreachable. Please try again.</span>';
            }
            
            btn.disabled = false;
            btn.innerHTML = '<span>🚀</span> Call Me Now';
        }

        function startPolling() {
            if (pollingInterval) clearInterval(pollingInterval);
            pollingInterval = setInterval(updateStatus, 2000);
            updateStatus();
        }

        async function updateStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                
                document.getElementById('active-count').innerText = data.active_calls;
                
                const liveStatus = document.getElementById('live-status');
                if (data.active_calls > 0) {
                    liveStatus.innerText = 'LIVE';
                    liveStatus.className = 'badge badge-success';
                } else {
                    liveStatus.innerText = 'Waiting...';
                    liveStatus.className = 'badge badge-info';
                }
                
                // Update transcript
                const box = document.getElementById('transcript-box');
                if (data.sessions && data.sessions.length > 0) {
                    const session = data.sessions[0];
                    if (session.history && session.history.length > 0) {
                        box.innerHTML = session.history.map(t => {
                            const role = t.role || 'unknown';
                            const content = t.content || '';
                            const isUser = role === 'user';
                            return `
                                <div class="turn">
                                    <div class="turn-role ${isUser ? 'user' : 'agent'}">${isUser ? '► You' : '◄ Arya'}</div>
                                    <div class="turn-content">${content}</div>
                                </div>
                            `;
                        }).join('');
                        box.scrollTop = box.scrollHeight;
                    } else {
                        box.innerHTML = '<div class="transcript-empty">Call connected. Waiting for conversation...</div>';
                    }
                }
            } catch (e) {}
        }

        // Start polling on page load
        setInterval(updateStatus, 3000);
        updateStatus();
    </script>
</body>
</html>
"""

ADMIN_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arya Admin | Control Center</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=Inter:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #070a13;
            --bg-card: #111827;
            --primary: #38bdf8;
            --primary-glow: rgba(56, 189, 248, 0.4);
            --secondary: #818cf8;
            --text-main: #f8fafc;
            --text-dim: #94a3b8;
            --success: #4ade80;
            --danger: #f87171;
            --border: rgba(255, 255, 255, 0.08);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Inter', sans-serif; background-color: var(--bg-dark); color: var(--text-main); line-height: 1.5; overflow: hidden; }
        h1, h2, h3 { font-family: 'Outfit', sans-serif; font-weight: 600; }

        .app-container { display: grid; grid-template-columns: 260px 1fr; height: 100vh; }
        aside { background: #0c1120; border-right: 1px solid var(--border); padding: 2rem 1.5rem; display: flex; flex-direction: column; }
        
        .logo-area { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 2rem; }
        .logo-icon { width: 32px; height: 32px; background: linear-gradient(135deg, var(--primary), var(--secondary)); border-radius: 8px; }
        .logo-text { font-size: 1.2rem; }

        .nav-item { padding: 0.75rem 1rem; border-radius: 10px; color: var(--text-dim); cursor: pointer; transition: 0.2s; margin-bottom: 0.25rem; }
        .nav-item:hover { color: var(--text-main); background: rgba(255,255,255,0.05); }
        .nav-item.active { color: var(--primary); background: rgba(56, 189, 248, 0.1); }

        main { display: flex; flex-direction: column; overflow: hidden; }
        .top-bar { padding: 1rem 2rem; background: rgba(17,24,39,0.8); backdrop-filter: blur(10px); border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
        .content-scroll { padding: 2rem; overflow-y: auto; flex: 1; }

        .page-view { display: none; animation: fadeIn 0.3s; }
        .page-view.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

        .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 16px; margin-bottom: 2rem; overflow: hidden; }
        .card-header { padding: 1.2rem 1.5rem; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
        
        table { width: 100%; border-collapse: collapse; }
        th { text-align: left; padding: 1rem; color: var(--text-dim); font-size: 0.7rem; text-transform: uppercase; border-bottom: 1px solid var(--border); }
        td { padding: 1rem; border-bottom: 1px solid var(--border); font-size: 0.85rem; }

        .badge { padding: 0.25rem 0.6rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700; }
        .badge-success { background: rgba(74, 222, 128, 0.1); color: var(--success); }
        .badge-info { background: rgba(56, 189, 248, 0.1); color: var(--primary); }

        .terminal { background: #000; padding: 1.2rem; border-radius: 12px; font-family: 'Fira Code', monospace; font-size: 0.8rem; height: 400px; overflow-y: auto; color: #d1d5db; border: 1px solid #1e293b; }
        .log-line { margin-bottom: 4px; padding: 2px 4px; border-radius: 4px; }
        .log-info { color: var(--primary); }
        .log-err { color: var(--danger); }

        button { background: linear-gradient(135deg, var(--primary), var(--secondary)); color: white; border: none; padding: 0.6rem 1.2rem; border-radius: 8px; font-weight: 600; cursor: pointer; transition: 0.2s; }
        button:hover { filter: brightness(1.1); }
        button.sec { background: rgba(255,255,255,0.05); border: 1px solid var(--border); color: var(--text-dim); }

        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; padding: 1.5rem; }
        .form-group { display: flex; flex-direction: column; gap: 0.5rem; }
        label { font-size: 0.8rem; color: var(--text-dim); font-weight: 600; text-transform: uppercase; }
        input { background: #080d1a; border: 1px solid #1e293b; padding: 0.8rem 1rem; border-radius: 10px; color: white; outline: none; }
        input:focus { border-color: var(--primary); }

        .memory-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1.5rem; }
        .memory-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; border-left: 4px solid var(--primary); }
    </style>
</head>
<body>
    <div class="app-container">
        <aside>
            <div class="logo-area">
                <div class="logo-icon"></div>
                <h1 class="logo-text">Arya Admin</h1>
            </div>
            <nav>
                <div class="nav-item active" onclick="switchTab('overview')">📊 Overview</div>
                <div class="nav-item" onclick="switchTab('logs')">📋 System Logs</div>
                <div class="nav-item" onclick="switchTab('intelligence')">🧠 Intelligence</div>
                <div class="nav-item" onclick="switchTab('config')">⚙️ Configuration</div>
            </nav>
            <div style="margin-top: auto; padding-top: 1rem; border-top: 1px solid var(--border);">
                <div style="font-size: 0.85rem; color: var(--primary); font-weight: 600;">Ghost</div>
                <div style="font-size: 0.7rem; color: var(--text-dim);">SUPER ADMIN</div>
                <button class="sec" style="width: 100%; margin-top: 1rem; font-size: 0.8rem;" onclick="logout()">🚪 Logout</button>
            </div>
        </aside>

        <main>
            <div class="top-bar">
                <h2 id="view-title">Overview</h2>
                <div style="display: flex; gap: 1rem; align-items: center;">
                    <span id="active-badge" class="badge badge-success">0 Active Calls</span>
                    <span id="ready-badge" class="badge badge-info">Agent Ready</span>
                </div>
            </div>

            <div class="content-scroll">
                <!-- OVERVIEW -->
                <div id="view-overview" class="page-view active">
                    <div class="card">
                        <div class="card-header">
                            <h3>Live Sessions</h3>
                            <button class="sec" onclick="updateData()">🔄 Refresh</button>
                        </div>
                        <table>
                            <thead><tr><th>Phone</th><th>Status</th><th>Last Message</th><th>Duration</th></tr></thead>
                            <tbody id="sessions-table">
                                <tr><td colspan="4" style="text-align:center; padding:2rem; opacity:0.5">No active sessions</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- LOGS -->
                <div id="view-logs" class="page-view">
                    <div class="card">
                        <div class="card-header">
                            <h3>System Telemetry</h3>
                            <button class="sec" onclick="fetchLogs()">🔄 Refresh</button>
                        </div>
                        <div style="padding: 1rem;">
                            <div class="terminal" id="log-terminal"></div>
                        </div>
                    </div>
                </div>

                <!-- INTELLIGENCE -->
                <div id="view-intelligence" class="page-view">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2rem">
                        <h3>Historical Sessions</h3>
                        <button class="sec" onclick="loadIntelligence()">🔄 Refresh</button>
                    </div>
                    <div id="intel-grid" class="memory-grid"></div>
                </div>

                <!-- CONFIG -->
                <div id="view-config" class="page-view">
                    <div class="card">
                        <div class="card-header"><h3>Environment Configuration</h3></div>
                        <form id="config-form" class="form-grid">
                            <div class="form-group"><label>Groq API Key</label><input type="password" name="GROQ_API_KEY"></div>
                            <div class="form-group"><label>Sarvam TTS Key</label><input type="password" name="SARVAM_API_KEY"></div>
                            <div class="form-group"><label>Twilio Number</label><input type="text" name="TWILIO_PHONE_NUMBER"></div>
                            <div class="form-group"><label>Server URL</label><input type="text" name="SERVER_URL"></div>
                            <div style="grid-column: 1/-1;"><button type="submit">💾 Save Configuration</button></div>
                        </form>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <script>
        let currentTab = 'overview';

        async function init() {
            setInterval(updateData, 3000);
            updateData();
            fetchConfig();
        }

        async function updateData() {
            try {
                const res = await fetch('/dashboard/status');
                const data = await res.json();
                
                document.getElementById('active-badge').innerText = data.active_calls + ' Active Calls';
                document.getElementById('ready-badge').innerText = data.agent_ready ? 'Agent Ready' : 'Standby';
                
                const table = document.getElementById('sessions-table');
                if (data.sessions && data.sessions.length > 0) {
                    table.innerHTML = data.sessions.map(s => `
                        <tr>
                            <td><code>${s.phone}</code></td>
                            <td><span class="badge badge-success">Active</span></td>
                            <td style="max-width:300px; overflow:hidden; text-overflow:ellipsis;">${s.last_text || '...'}</td>
                            <td>${s.duration}</td>
                        </tr>
                    `).join('');
                } else {
                    table.innerHTML = '<tr><td colspan="4" style="text-align:center; padding:2rem; opacity:0.5">No active sessions</td></tr>';
                }
            } catch (e) {}
        }

        async function fetchLogs() {
            try {
                const res = await fetch('/dashboard/logs');
                const logs = await res.json();
                const terminal = document.getElementById('log-terminal');
                terminal.innerHTML = logs.map(l => {
                    const cls = l.includes('ERROR') ? 'log-err' : (l.includes('INFO') ? 'log-info' : '');
                    return `<div class="log-line ${cls}">${l}</div>`;
                }).join('');
                terminal.scrollTop = terminal.scrollHeight;
            } catch (e) {}
        }

        async function loadIntelligence() {
            const grid = document.getElementById('intel-grid');
            grid.innerHTML = '<div style="opacity:0.5">Loading...</div>';
            try {
                const res = await fetch('/dashboard/historical-sessions');
                const data = await res.json();
                if (data.length === 0) {
                    grid.innerHTML = '<div style="opacity:0.5">No historical sessions found.</div>';
                    return;
                }
                grid.innerHTML = data.map(s => `
                    <div class="memory-card">
                        <div style="font-size:0.7rem; color:var(--text-dim); margin-bottom:0.5rem">${s.phone || 'Unknown'}</div>
                        <div style="font-size:1rem; font-weight:600; margin-bottom:0.75rem">${s.session_id.substring(0, 12)}...</div>
                        <div style="font-size:0.85rem; color:var(--text-dim); margin-bottom:1rem">${s.summary || 'No summary'}</div>
                        <div style="font-size:0.7rem; opacity:0.5">${new Date(s.started_at).toLocaleString()}</div>
                    </div>
                `).join('');
            } catch (e) {
                grid.innerHTML = '<div style="color:var(--danger)">Failed to load intelligence.</div>';
            }
        }

        async function fetchConfig() {
            try {
                const res = await fetch('/dashboard/config');
                const data = await res.json();
                const form = document.getElementById('config-form');
                for (let [k, v] of Object.entries(data)) {
                    if (form.elements[k]) form.elements[k].value = v;
                }
            } catch (e) {}
        }

        function switchTab(id) {
            currentTab = id;
            document.querySelectorAll('.page-view').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            document.getElementById('view-' + id).classList.add('active');
            document.querySelector(`[onclick="switchTab('${id}')"]`).classList.add('active');
            document.getElementById('view-title').innerText = id.charAt(0).toUpperCase() + id.slice(1);
            
            if (id === 'logs') fetchLogs();
            if (id === 'intelligence') loadIntelligence();
        }

        async function logout() {
            await fetch('/logout', { method: 'POST' });
            window.location.href = '/';
        }

        init();
    </script>
</body>
</html>
"""

LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Arya Admin | Login</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600&family=Inter:wght@400;500&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background: #070a13; color: white; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
        .card { background: #111827; padding: 3rem; border-radius: 24px; width: 400px; border: 1px solid rgba(255,255,255,0.08); position: relative; }
        .card::before { content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 4px; background: linear-gradient(to right, #38bdf8, #818cf8); }
        h2 { font-family: 'Outfit'; font-size: 1.8rem; margin-bottom: 0.5rem; text-align: center; }
        p { text-align: center; color: #94a3b8; font-size: 0.9rem; margin-bottom: 2rem; }
        .form-group { margin-bottom: 1.5rem; }
        label { display: block; font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.5rem; text-transform: uppercase; font-weight: 600; }
        input { width: 100%; background: #080d1a; border: 1px solid #1e293b; padding: 1rem; border-radius: 12px; color: white; box-sizing: border-box; outline: none; }
        input:focus { border-color: #38bdf8; }
        button { width: 100%; background: linear-gradient(135deg, #38bdf8, #818cf8); border: none; padding: 1rem; border-radius: 12px; color: white; font-weight: 700; cursor: pointer; font-size: 1rem; }
        button:hover { filter: brightness(1.1); }
        #msg { text-align: center; margin-top: 1.5rem; font-size: 0.85rem; min-height: 1.2rem; }
        .back-link { text-align: center; margin-top: 2rem; font-size: 0.85rem; }
        .back-link a { color: #38bdf8; text-decoration: none; }
    </style>
</head>
<body>
    <div class="card">
        <h2>🔐 Admin Access</h2>
        <p>Super Admin authentication required</p>
        <div class="form-group">
            <label>Username</label>
            <input type="text" id="username" placeholder="simonriley141">
        </div>
        <div class="form-group">
            <label>Password</label>
            <input type="password" id="password" placeholder="••••••••">
        </div>
        <button onclick="login()">Authenticate</button>
        <div id="msg"></div>
        <div class="back-link"><a href="/">← Back to Home</a></div>
    </div>
    <script>
        async function login() {
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            const msg = document.getElementById('msg');
            msg.innerText = 'Authenticating...';
            msg.style.color = '#94a3b8';
            
            try {
                const res = await fetch('/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    window.location.href = '/admin';
                } else {
                    msg.style.color = '#f87171';
                    msg.innerText = data.message || 'Authentication failed';
                }
            } catch (e) {
                msg.style.color = '#f87171';
                msg.innerText = 'Server unreachable';
            }
        }
        
        document.getElementById('password').addEventListener('keyup', e => { if (e.key === 'Enter') login(); });
    </script>
</body>
</html>
"""

# Keep DASHBOARD_HTML as alias for backwards compatibility
DASHBOARD_HTML = ADMIN_DASHBOARD
