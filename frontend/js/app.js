/* ============================================================
   STATE
============================================================ */
const state = {
  sessionId:   crypto.randomUUID(),
  chatHistory: [],
  files: {
    chatImage:    null,
    chatCsv:      null,
    chatPdf:      null,
    analysisCsv:  null,
    sqlSchemaCsv: null,
    augCsv:       null
  },
  aug: { b64:null, filename:null, diagnosis:null, result:null },
  initialized: false,
  imageUploads: []   // [{timestamp}] — tracks image upload times for quota UI
};

const tabLabels = {
  chat:    'Chat Analysis',
  upload:  'Data Upload & Analysis',
  sql:     'SQL Generator',
  augment: 'Data Augmentation'
};

const IMAGE_QUOTA     = 3;
const IMAGE_WINDOW_MS = 6 * 60 * 60 * 1000; // 6 hours
const IMG_QUOTA_KEY   = 'dab_img_quota';

/* ============================================================
   CALLED FROM auth.js AFTER EVERY SUCCESSFUL AUTH
============================================================ */
function onAppEnter(){
  document.getElementById('sessionIdLabel').textContent = state.sessionId.slice(0,8);
  if(state.initialized) return;
  state.initialized = true;
  initApp();
}

/* ============================================================
   SAFE WRAPPER
============================================================ */
function safeRun(label, fn){
  try{ fn(); }
  catch(e){ console.error('[DAB init] ' + label + ' failed:', e); }
}

/* ============================================================
   INIT
============================================================ */
function initApp(){
  safeRun('pingApi', pingApi);
  safeRun('apiBaseInput', ()=>{
    document.getElementById('apiBaseInput').addEventListener('change', pingApi);
  });
  safeRun('nav tabs', ()=>{
    document.querySelectorAll('.nav-tab').forEach(btn=>{
      btn.addEventListener('click', ()=>{
        document.querySelectorAll('.nav-tab').forEach(b=>b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
        document.getElementById('breadcrumbCurrent').textContent = tabLabels[btn.dataset.tab] || '';
        document.querySelector('.sidebar').classList.remove('open');
        document.getElementById('sidebarOverlay').classList.remove('visible');
      });
    });
  });
  safeRun('sidebar toggle', ()=>{
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.getElementById('sidebarOverlay');
    const open  = ()=>{ sidebar.classList.add('open');  overlay.classList.add('visible'); };
    const close = ()=>{ sidebar.classList.remove('open'); overlay.classList.remove('visible'); };

    document.getElementById('sidebarToggle').addEventListener('click', ()=>{
      sidebar.classList.contains('open') ? close() : open();
    });
    overlay.addEventListener('click', close);

    // Close sidebar automatically when a nav tab is tapped on mobile
    document.querySelectorAll('.nav-tab, .chat-item').forEach(()=>{}); // no-op placeholder, real close handled in tab click handler below
  });
  safeRun('darkMode',          wireDarkMode);
  safeRun('exportChat',        wireExportChat);
  safeRun('imageQuota',        initImageQuota);
  safeRun('wireFileDrops',     wireFileDrops);
  safeRun('wireChat',          wireChat);
  safeRun('wireUploadAnalysis',wireUploadAnalysis);
  safeRun('wireSqlGenerator',  wireSqlGenerator);
  safeRun('wireAugmentation',  wireAugmentation);
  safeRun('profileModal',      wireProfileModal);
  safeRun('newChatBtn', ()=>{
    document.getElementById('newChatBtn').addEventListener('click', startNewChat);
  });
}

/* ============================================================
   AUTH-AWARE FETCH
============================================================ */
function authedFetch(path, opts={}){
  const url = path.startsWith('http') ? path : getApiBase() + path;
  opts.headers = Object.assign({}, opts.headers, authHeaders());
  return fetch(url, opts);
}

/* ============================================================
   API HEALTH CHECK
============================================================ */
async function pingApi(){
  const dot = document.getElementById('apiStatusDot');
  try{
    const res = await authedFetch('/recent-chat-titles', { signal: AbortSignal.timeout(4000) });
    dot.className = 'status-dot ' + (res.ok ? 'ok' : 'bad');
    if(res.ok) loadRecentChats();
  }catch(e){ dot.className = 'status-dot bad'; }
}

/* ============================================================
   HELPERS
============================================================ */
function fileToBase64(file){
  return new Promise((resolve, reject)=>{
    const r = new FileReader();
    r.onload  = ()=> resolve(r.result.split(',')[1]);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}
function fmtBytes(b){ return (b/1024).toFixed(2) + ' KB'; }
function escapeHtml(str){
  return String(str).replace(/[&<>"']/g,
    c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function statusBox(el, type, html){ el.innerHTML = `<div class="status-msg ${type}">${html}</div>`; }
function clearStatus(el){ el.innerHTML = ''; }

function wireFileDrops(){
  document.querySelectorAll('.file-drop').forEach(drop=>{
    const input = document.getElementById(drop.dataset.target);
    if(!input) return;
    drop.addEventListener('click', ()=> input.click());
    ['dragover','dragleave','drop'].forEach(evt=>{
      drop.addEventListener(evt, e=>{
        e.preventDefault();
        drop.classList.toggle('dragover', evt==='dragover');
        if(evt==='drop' && e.dataTransfer.files.length){
          input.files = e.dataTransfer.files;
          input.dispatchEvent(new Event('change'));
        }
      });
    });
  });
}

function renderTable(rows, maxRows=null){
  if(!rows || !rows.length) return '<div class="status-msg info">No data to display.</div>';
  const data = maxRows ? rows.slice(0, maxRows) : rows;
  const cols = Object.keys(data[0]);
  let html = '<div class="data-table-wrap"><table class="data-table"><thead><tr>';
  cols.forEach(c=> html += `<th>${escapeHtml(c)}</th>`);
  html += '</tr></thead><tbody>';
  data.forEach(row=>{
    html += '<tr>';
    cols.forEach(c=>{ html += `<td>${escapeHtml(row[c]==null ? '' : row[c])}</td>`; });
    html += '</tr>';
  });
  html += '</tbody></table></div>';
  return html;
}

/* ============================================================
   ⑧ SKELETON LOADERS
============================================================ */
function chatSkeletonHTML(){
  return `
    <div class="msg ai skel-msg">
      <div class="skel-avatar skel"></div>
      <div class="skel-bubble">
        <div class="skel-line skel" style="width:85%;"></div>
        <div class="skel-line skel" style="width:60%;"></div>
        <div class="skel-line skel" style="width:40%;"></div>
      </div>
    </div>`;
}

function analysisSkeletonHTML(){
  return `
    <div class="skel-panel">
      <div class="skel-title skel"></div>
      <div class="skel-row skel"></div>
      <div class="skel-row skel"></div>
      <div class="skel-row skel"></div>
    </div>
    <div class="skel-panel">
      <div class="skel-title skel" style="width:30%;"></div>
      <div class="skel-table skel"></div>
    </div>
    <div class="skel-panel">
      <div class="skel-title skel" style="width:35%;"></div>
      <div class="skel-chart skel"></div>
    </div>`;
}

function sqlSkeletonHTML(){
  return `
    <div class="skel-panel">
      <div class="skel-title skel" style="width:35%;"></div>
      <div class="skel-row skel" style="height:90px;border-radius:8px;"></div>
    </div>`;
}

function wireDarkMode(){
  const DARK_KEY = 'dab_dark';
  const btn      = document.getElementById('darkToggle');

  function applyTheme(dark){
    document.body.classList.toggle('dark', dark);
    localStorage.setItem(DARK_KEY, dark ? '1' : '0');
  }

  // Restore saved preference
  const saved = localStorage.getItem(DARK_KEY);
  if(saved === '1') applyTheme(true);
  else if(saved === null && window.matchMedia('(prefers-color-scheme: dark)').matches) applyTheme(true);

  btn.addEventListener('click', ()=>{
    applyTheme(!document.body.classList.contains('dark'));
  });
}

/* ============================================================
   ② EXPORT CHAT
============================================================ */
function wireExportChat(){
  const toggleBtn = document.getElementById('exportToggleBtn');
  const menu      = document.getElementById('exportMenu');

  // Toggle dropdown
  toggleBtn.addEventListener('click', e=>{
    e.stopPropagation();
    menu.classList.toggle('open');
  });
  document.addEventListener('click', ()=> menu.classList.remove('open'));

  // Download as Markdown
  document.getElementById('exportMdBtn').addEventListener('click', ()=>{
    menu.classList.remove('open');
    if(!state.chatHistory.length){
      alert('No chat history to export yet. Start a conversation first.'); return;
    }
    const auth = loadAuth();
    const user = (auth && auth.user) ? auth.user.name : 'User';
    const date = new Date().toLocaleDateString('en-US', { year:'numeric', month:'long', day:'numeric' });

    let md = `# DataAnalystBot — Chat Export\n\n`;
    md    += `**Session:** \`${state.sessionId}\`  \n`;
    md    += `**User:** ${escapeHtml(user)}  \n`;
    md    += `**Exported:** ${date}\n\n---\n\n`;

    state.chatHistory.forEach(msg=>{
      const role = msg.type === 'ai' ? '**🤖 DataAnalystBot**' : `**🧑 ${escapeHtml(user)}**`;
      md += `${role}\n\n${msg.content}\n\n---\n\n`;
    });

    downloadFile(md, `chat-${state.sessionId.slice(0,8)}.md`, 'text/markdown');
  });

  // Download as PDF (print-to-PDF via browser)
  document.getElementById('exportPdfBtn').addEventListener('click', ()=>{
    menu.classList.remove('open');
    if(!state.chatHistory.length){
      alert('No chat history to export yet. Start a conversation first.'); return;
    }
    const auth = loadAuth();
    const user = (auth && auth.user) ? auth.user.name : 'User';
    const date = new Date().toLocaleDateString('en-US', { year:'numeric', month:'long', day:'numeric' });

    const isDark = document.body.classList.contains('dark');
    const bg     = isDark ? '#0f1117' : '#f5f6fa';
    const fg     = isDark ? '#e2e6f3' : '#1b1f2a';
    const surf   = isDark ? '#181c27' : '#ffffff';
    const border = isDark ? '#272d3d' : '#e3e6ee';
    const accent = '#5b5ff0';

    let rows = '';
    state.chatHistory.forEach(msg=>{
      const isAi   = msg.type === 'ai';
      const sender = isAi ? '🤖 DataAnalystBot' : `🧑 ${escapeHtml(user)}`;
      const bg2    = isAi ? surf : accent;
      const fg2    = isAi ? fg   : '#fff';
      rows += `
        <div style="margin-bottom:16px;">
          <div style="font-size:11px;font-weight:700;margin-bottom:6px;color:${isAi ? accent : fg};opacity:0.7;">${sender}</div>
          <div style="background:${bg2};border:1px solid ${border};border-radius:10px;padding:12px 16px;font-size:13px;line-height:1.6;color:${fg2};white-space:pre-wrap;word-break:break-word;">${escapeHtml(msg.content)}</div>
        </div>`;
    });

    const html = `<!DOCTYPE html><html><head><meta charset="UTF-8">
      <title>Chat Export — DataAnalystBot</title>
      <style>
        body{font-family:Inter,system-ui,sans-serif;background:${bg};color:${fg};margin:0;padding:32px;}
        h1{font-size:20px;margin-bottom:4px;}
        .meta{font-size:12px;opacity:0.5;margin-bottom:24px;}
        hr{border:none;border-top:1px solid ${border};margin:20px 0;}
        @media print{body{padding:20px;}}
      </style></head><body>
      <h1>DataAnalystBot — Chat Export</h1>
      <div class="meta">Session: ${state.sessionId} &nbsp;·&nbsp; ${date}</div>
      <hr>${rows}
      </body></html>`;

    const win = window.open('', '_blank');
    win.document.write(html);
    win.document.close();
    setTimeout(()=>{ win.focus(); win.print(); }, 400);
  });
}

function downloadFile(content, filename, mime){
  const blob = new Blob([content], { type: mime });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ============================================================
   ③ IMAGE UPLOAD QUOTA UI
   Tracks uploads client-side in localStorage with timestamps.
   Resets automatically after 6 hours.
============================================================ */
function initImageQuota(){
  // Load persisted timestamps
  try{
    const raw = JSON.parse(localStorage.getItem(IMG_QUOTA_KEY) || '[]');
    state.imageUploads = raw.filter(t=> Date.now() - t < IMAGE_WINDOW_MS);
  }catch(e){ state.imageUploads = []; }
  renderImageQuota();
}

function recordImageUpload(){
  state.imageUploads.push(Date.now());
  // keep only within window
  state.imageUploads = state.imageUploads.filter(t=> Date.now() - t < IMAGE_WINDOW_MS);
  localStorage.setItem(IMG_QUOTA_KEY, JSON.stringify(state.imageUploads));
  renderImageQuota();
}

function renderImageQuota(){
  // Purge expired
  state.imageUploads = state.imageUploads.filter(t=> Date.now() - t < IMAGE_WINDOW_MS);
  const used  = state.imageUploads.length;
  const left  = IMAGE_QUOTA - used;

  // Dots
  for(let i=1; i<=IMAGE_QUOTA; i++){
    const dot = document.getElementById('iq' + i);
    if(dot) dot.className = 'iq-dot' + (i <= used ? ' used' : '');
  }
  // Count
  const countEl = document.getElementById('iqCount');
  if(countEl) countEl.textContent = `${used}/${IMAGE_QUOTA}`;

  // Inline label
  const inlineEl = document.getElementById('imgQuotaInline');
  if(inlineEl){
    inlineEl.textContent = `(${used}/${IMAGE_QUOTA} used)`;
    inlineEl.style.color = used >= IMAGE_QUOTA ? 'var(--danger)' : 'var(--text-soft)';
  }

  // Reset timer — time until oldest upload expires
  const resetEl = document.getElementById('iqReset');
  if(resetEl){
    if(used === 0){
      resetEl.textContent = 'no limit';
    } else {
      const oldest  = Math.min(...state.imageUploads);
      const resetIn = Math.max(0, (oldest + IMAGE_WINDOW_MS) - Date.now());
      const h       = Math.floor(resetIn / 3600000);
      const m       = Math.floor((resetIn % 3600000) / 60000);
      resetEl.textContent = `resets in ${h}h ${m}m`;
    }
  }

  // Disable image drop if quota full
  const imgInput = document.getElementById('chatImage');
  const imgDrop  = imgInput?.closest('.upload-slot')?.querySelector('.file-drop');
  if(imgInput) imgInput.disabled = left <= 0;
  if(imgDrop){
    imgDrop.style.opacity     = left <= 0 ? '0.45' : '1';
    imgDrop.style.cursor      = left <= 0 ? 'not-allowed' : 'pointer';
    imgDrop.title             = left <= 0 ? 'Image upload limit reached (3 per 6 hours)' : '';
    imgDrop.querySelector('span').textContent = left <= 0 ? 'Limit reached (resets in 6h)' : 'Drop image or browse';
  }

  // Refresh every minute so timer updates
  clearTimeout(renderImageQuota._timer);
  if(used > 0){
    renderImageQuota._timer = setTimeout(renderImageQuota, 60000);
  }
}

/* ============================================================
   SIDEBAR — sessions
============================================================ */
async function startNewChat(){
  if(state.chatHistory.length){
    try{
      await authedFetch('/save-chat', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ session_id:state.sessionId, chat_history:state.chatHistory })
      });
    }catch(_){}
  }
  state.sessionId   = crypto.randomUUID();
  state.chatHistory = [];
  document.getElementById('sessionIdLabel').textContent = state.sessionId.slice(0,8);
  renderChat();
  loadRecentChats();
}

async function loadRecentChats(){
  const list = document.getElementById('chatList');
  try{
    const res      = await authedFetch('/recent-chat-titles');
    const data     = await res.json();
    const sessions = (data.sessions || []).slice().reverse();
    if(!sessions.length){ list.innerHTML = '<div class="chat-empty">No recent sessions yet.</div>'; return; }
    list.innerHTML = '';
    sessions.forEach(s=>{
      const b = document.createElement('button');
      b.className   = 'chat-item' + (s.session_id===state.sessionId ? ' active' : '');
      b.textContent = (s.title || s.session_id).slice(0,40);
      b.addEventListener('click', ()=> loadSession(s.session_id));
      list.appendChild(b);
    });
  }catch(e){ list.innerHTML = '<div class="chat-empty">Could not load sessions.</div>'; }
}

async function loadSession(sessionId){
  try{
    const res  = await authedFetch('/recent-chats/' + sessionId);
    const data = await res.json();
    state.sessionId   = sessionId;
    state.chatHistory = data.chat_history || [];
    document.getElementById('sessionIdLabel').textContent = sessionId.slice(0,8);
    renderChat();
    loadRecentChats();
    document.querySelector('.nav-tab[data-tab="chat"]').click();
  }catch(e){ alert('Could not load chat history: ' + e.message); }
}

/* ============================================================
   TAB 1 — CHAT
============================================================ */
function wireChat(){
  ['chatImage','chatCsv','chatPdf'].forEach(id=>{
    document.getElementById(id).addEventListener('change', e=>{
      const f = e.target.files[0] || null;
      state.files[id] = f;
      const chip = document.getElementById(id + 'Chip');
      chip.innerHTML = f
        ? `<span class="file-chip">${escapeHtml(f.name)} (${fmtBytes(f.size)}) <button data-clear="${id}">✕</button></span>`
        : '';
      // Track image uploads for quota
      if(id === 'chatImage' && f) recordImageUpload();
    });
    document.getElementById(id + 'Chip').addEventListener('click', e=>{
      const clearId = e.target.dataset.clear;
      if(!clearId) return;
      state.files[clearId] = null;
      document.getElementById(clearId).value = '';
      document.getElementById(clearId + 'Chip').innerHTML = '';
    });
  });

  document.getElementById('chatSendBtn').addEventListener('click', sendChatMessage);
  document.getElementById('chatInput').addEventListener('keydown', e=>{ if(e.key==='Enter') sendChatMessage(); });
}

function renderChat(){
  const win = document.getElementById('chatWindow');
  if(!state.chatHistory.length){
    win.innerHTML = `<div class="chat-empty-state">
      <div class="empty-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9" y2="9.01"/><line x1="15" y1="9" x2="15" y2="9.01"/></svg></div>
      <h3>Ask your first question</h3>
      <p>Try: "Summarize the trends in this dataset" or attach a file above.</p>
    </div>`;
    return;
  }
  win.innerHTML = '';
  state.chatHistory.forEach(msg=>{
    const isAi = msg.type === 'ai';
    const div  = document.createElement('div');
    div.className = 'msg ' + (isAi ? 'ai' : 'user');
    let extra = '';
    if(!isAi && msg.image)
      extra = `<img src="data:${msg.image_type||'image/png'};base64,${msg.image}" alt="upload">`;
    div.innerHTML = `<div class="msg-avatar">${isAi ? 'AI' : 'You'}</div>
      <div class="msg-bubble">${escapeHtml(msg.content)}${extra}</div>`;
    win.appendChild(div);
  });
  win.scrollTop = win.scrollHeight;
}

async function sendChatMessage(){
  const input   = document.getElementById('chatInput');
  const sendBtn = document.getElementById('chatSendBtn');
  const text    = input.value.trim();
  if(!text) return;
  sendBtn.disabled = true;

  let image_b64=null, image_type=null, csv_b64=null, csv_filename=null, pdf_b64=null, pdf_filename=null;
  if(state.files.chatImage){ image_b64 = await fileToBase64(state.files.chatImage); image_type = state.files.chatImage.type; }
  if(state.files.chatCsv){   csv_b64   = await fileToBase64(state.files.chatCsv);   csv_filename = state.files.chatCsv.name; }
  if(state.files.chatPdf){   pdf_b64   = await fileToBase64(state.files.chatPdf);   pdf_filename = state.files.chatPdf.name; }

  state.chatHistory.push({ type:'human', content:text, image:image_b64, image_type });
  renderChat();
  input.value = '';

  // Show a skeleton "typing" placeholder while waiting for the response
  const win = document.getElementById('chatWindow');
  const skelEl = document.createElement('div');
  skelEl.innerHTML = chatSkeletonHTML();
  win.appendChild(skelEl.firstElementChild);
  win.scrollTop = win.scrollHeight;

  const payload = {
    question:     text,
    session_id:   state.sessionId,
    chat_history: state.chatHistory.map(m=>({ type:m.type, content:m.content }))
  };
  if(image_b64){ payload.image_base64 = image_b64; payload.image_type = image_type; }
  if(csv_b64){   payload.csv_base64   = csv_b64;   payload.csv_filename = csv_filename; }
  if(pdf_b64){   payload.pdf_base64   = pdf_b64;   payload.pdf_filename = pdf_filename; }

  try{
    const res = await authedFetch('/multi-upload', {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)
    });
    if(res.status === 429){ handleRateLimit('chat'); throw new Error('Rate limit reached. Please wait a moment.'); }
    if(!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    state.chatHistory.push({ type:'ai', content: data.response || '⚠️ No answer returned.' });
  }catch(e){
    state.chatHistory.push({ type:'ai', content: '❌ ' + e.message });
  }
  renderChat();
  sendBtn.disabled = false;
}

/* ============================================================
   TAB 2 — UPLOAD & ANALYSIS
============================================================ */
function wireUploadAnalysis(){
  document.getElementById('analysisCsv').addEventListener('change', e=>{
    const f   = e.target.files[0];
    state.files.analysisCsv = f || null;
    const box = document.getElementById('analysisFileDetails');
    const btn = document.getElementById('analyzeBtn');
    if(f){ box.innerHTML = renderTable([{ Filename:f.name, Size:fmtBytes(f.size), Type:f.type||'—' }]); btn.disabled = false; }
    else  { box.innerHTML = ''; btn.disabled = true; }
  });

  document.getElementById('analyzeBtn').addEventListener('click', async ()=>{
    const statusEl  = document.getElementById('analysisStatus');
    const resultsEl = document.getElementById('analysisResults');
    const btn       = document.getElementById('analyzeBtn');
    const f         = state.files.analysisCsv;
    if(!f) return;
    resultsEl.innerHTML = analysisSkeletonHTML();
    btn.disabled = true;
    statusBox(statusEl, 'info', '<span class="spinner"></span> Analyzing your data…');
    try{
      const b64     = await fileToBase64(f);
      const payload = { csv_base64:b64, csv_filename:f.name, session_id:state.sessionId };
      let res = await authedFetch('/analyze-data', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
      if(!res.ok) res = await authedFetch('/clean-data', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
      if(!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      clearStatus(statusEl);
      resultsEl.innerHTML = renderAnalysis(data);
    }catch(e){
      if(e.message.includes('429') || e.message.toLowerCase().includes('rate')) handleRateLimit('upload');
      statusBox(statusEl, 'error', '❌ Analysis failed: ' + e.message);
    }
    btn.disabled = false;
  });
}

function renderAnalysis(resp){
  let html = '';

  html += '<div class="panel"><h3>Data cleaning log</h3>';
  const log = resp.cleaning_log || [];
  if(log.length){ html += '<ol style="padding-left:20px;font-size:13px;line-height:1.8;">'; log.forEach(s=> html += `<li>${escapeHtml(s)}</li>`); html += '</ol>'; }
  else html += '<div class="status-msg info">No cleaning steps reported.</div>';
  html += '</div>';

  const stats = resp.statistical_summary;
  if(typeof stats === 'string' && stats.trim()){
    let sumText = stats, corrText = '';
    if(stats.includes('Correlation Matrix:'))[sumText, corrText] = stats.split('Correlation Matrix:');
    sumText = sumText.replace('Numeric Variable Summary:','').trim();
    html += `<div class="panel"><h3>Numeric variable summary</h3>${sumText ? `<pre>${escapeHtml(sumText)}</pre>` : '<div class="status-msg info">No numeric summary.</div>'}</div>`;
    if(corrText.trim()) html += `<div class="panel"><h3>Correlation matrix</h3><pre>${escapeHtml(corrText.trim())}</pre></div>`;
  } else if(stats && typeof stats === 'object'){
    html += `<div class="panel"><h3>Statistical summary</h3><pre>${escapeHtml(JSON.stringify(stats,null,2))}</pre></div>`;
  }

  const colInfo = resp.column_info || {};
  if(typeof colInfo === 'object'){
    html += `<div class="panel"><details class="expander"><summary>Column info</summary><div class="expander-body"><div class="grid-2">
      <div><label class="field-label">Original columns</label><pre>${(colInfo.original_columns||[]).map((c,i)=>`${i+1}. ${c}`).join('\n')}</pre>
        ${colInfo.data_types ? '<label class="field-label" style="margin-top:10px;display:block;">Data types</label>' + renderTable(Object.entries(colInfo.data_types).map(([k,v])=>({Column:k,Type:v}))) : ''}
      </div>
      <div><label class="field-label">Cleaned columns</label><pre>${(colInfo.cleaned_columns||[]).map((c,i)=>`${i+1}. ${c}`).join('\n')}</pre>
        ${colInfo.missing_values ? '<label class="field-label" style="margin-top:10px;display:block;">Missing values</label>' + renderTable(Object.entries(colInfo.missing_values).map(([k,v])=>({Column:k,Count:v}))) : ''}
      </div></div>
      ${colInfo.unique_counts ? '<label class="field-label" style="margin-top:14px;display:block;">Unique counts</label>' + renderTable(Object.entries(colInfo.unique_counts).map(([k,v])=>({Column:k,Unique:v}))) : ''}
    </div></details></div>`;
  }

  const cleaned = (resp.sample_data && resp.sample_data.cleaned) || [];
  html += `<div class="panel"><h3>Cleaned data preview</h3>${renderTable(cleaned)}</div>`;

  const plots    = resp.visualizations || {};
  const plotKeys = Object.keys(plots);
  if(plotKeys.length){
    html += '<div class="panel"><h3>Visualizations</h3>';
    plotKeys.forEach((name, idx)=>{
      const fig = plots[name];
      if(fig.error){ html += `<div class="status-msg warn">${escapeHtml(name)}: ${escapeHtml(fig.error)}</div>`; }
      else{
        const divId = 'plot_' + idx + '_' + Math.random().toString(36).slice(2);
        html += `<div style="margin-bottom:18px;"><label class="field-label">${escapeHtml(name.replace(/_/g,' '))}</label><div id="${divId}" style="background:var(--bg);border:1px solid var(--border);border-radius:8px;min-height:340px;"></div></div>`;
        queuePlotlyRender(divId, fig, name);
      }
    });
    html += '</div>';
  }
  return html;
}

const PLOTLY_CDN = 'https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js';
let _plotlyPromise = null;
function loadPlotly(){
  if(window.Plotly) return Promise.resolve();
  if(_plotlyPromise) return _plotlyPromise;
  _plotlyPromise = new Promise((res, rej)=>{ const s=document.createElement('script'); s.src=PLOTLY_CDN; s.onload=res; s.onerror=rej; document.head.appendChild(s); });
  return _plotlyPromise;
}
function queuePlotlyRender(divId, fig, name){
  loadPlotly().then(()=>{
    const el = document.getElementById(divId);
    if(!el) return;
    const isDark = document.body.classList.contains('dark');
    const layout = Object.assign({}, fig.layout, {
      title: name.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase()),
      paper_bgcolor:'transparent', plot_bgcolor:'transparent',
      font:{ color: isDark ? '#e2e6f3' : '#1b1f2a', family:'Inter' },
      margin:{ l:40, r:20, t:40, b:40 }
    });
    Plotly.newPlot(el, fig.data||[], layout, { responsive:true, displaylogo:false });
  }).catch(()=>{ const el=document.getElementById(divId); if(el) el.innerHTML='<div class="status-msg warn">Could not load chart library.</div>'; });
}

/* ============================================================
   TAB 3 — SQL GENERATOR
============================================================ */
function wireSqlGenerator(){
  document.getElementById('sqlSchemaCsv').addEventListener('change', async e=>{
    const f       = e.target.files[0];
    const preview = document.getElementById('sqlSchemaPreview');
    if(!f){ preview.innerHTML=''; return; }
    state.files.sqlSchemaCsv = f;
    preview.innerHTML = '<div class="status-msg info"><span class="spinner"></span> Reading file…</div>';
    try{
      if(!f.name.toLowerCase().endsWith('.csv')){ preview.innerHTML='<div class="status-msg warn">Convert to CSV for auto-detection.</div>'; return; }
      const rows = parseCSV(await f.text());
      if(!rows.length){ preview.innerHTML='<div class="status-msg warn">File appears empty.</div>'; return; }
      const cols   = Object.keys(rows[0]);
      const dtypes = inferDtypes(rows, cols);
      const schema = cols.map(c=>`${c} (${dtypes[c]})`).join(', ');
      let html = '<label class="field-label" style="margin-top:10px;display:block;">Preview (first 5 rows)</label>' + renderTable(rows, 5);
      html    += `<label class="field-label" style="margin-top:10px;display:block;">Auto-detected schema</label><pre>-- ${escapeHtml(schema)}</pre>`;
      html    += `<div class="status-msg info"><button class="btn" style="margin-left:4px;" id="useSchemaBtn">Insert into schema field</button></div>`;
      preview.innerHTML = html;
      document.getElementById('useSchemaBtn').addEventListener('click', ()=>{
        document.getElementById('sqlSchema').value = `-- Auto-detected columns:\n-- ${schema}`;
      });
    }catch(err){ preview.innerHTML=`<div class="status-msg error">Could not read file: ${escapeHtml(err.message)}</div>`; }
  });

  document.getElementById('sqlGenerateBtn').addEventListener('click', async ()=>{
    const statusEl  = document.getElementById('sqlStatus');
    const resultsEl = document.getElementById('sqlResults');
    const btn       = document.getElementById('sqlGenerateBtn');
    const desc      = document.getElementById('sqlDescription').value.trim();
    if(!desc){ statusBox(statusEl, 'warn', '⚠️ Please describe what you want to query.'); return; }
    resultsEl.innerHTML = sqlSkeletonHTML();
    btn.disabled = true;
    statusBox(statusEl, 'info', '<span class="spinner"></span> Generating SQL query…');
    const payload = {
      description: desc,
      db_schema:   document.getElementById('sqlSchema').value,
      db_type:     document.getElementById('sqlDbType').value,
      query_type:  document.getElementById('sqlQueryType').value,
      session_id:  state.sessionId
    };
    try{
      let res = await authedFetch('/generate-sql', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
      if(!res.ok) res = await authedFetch('/sql-query', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
      if(!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      clearStatus(statusEl); statusBox(statusEl, 'success', '✅ SQL query generated.');
      let html = `<div class="panel"><h3>Generated query</h3><pre><code>${escapeHtml(data.sql_query||'')}</code></pre>`;
      if(data.explanation) html += `<details class="expander"><summary>Explanation</summary><div class="expander-body">${escapeHtml(data.explanation)}</div></details>`;
      if(data.suggestions) html += `<details class="expander"><summary>Optimization suggestions</summary><div class="expander-body">${escapeHtml(data.suggestions)}</div></details>`;
      html += `<button class="btn primary block" id="downloadSqlBtn" style="margin-top:10px;">Download .sql file</button></div>`;
      resultsEl.innerHTML = html;
      document.getElementById('downloadSqlBtn').addEventListener('click', ()=>{
        downloadFile(data.sql_query||'', 'generated_query.sql', 'text/plain');
      });
    }catch(e){
      if(e.message.includes('429') || e.message.toLowerCase().includes('rate')) handleRateLimit('sql');
      statusBox(statusEl, 'error', '❌ SQL generation failed: ' + e.message);
    }
    btn.disabled = false;
  });
}

function parseCSV(text){
  const lines = text.split(/\r?\n/).filter(l=>l.length);
  if(!lines.length) return [];
  const headers = lines[0].split(',').map(h=>h.trim());
  return lines.slice(1).map(line=>{ const vals=line.split(','); const o={}; headers.forEach((h,i)=> o[h]=vals[i]!==undefined?vals[i].trim():''); return o; });
}
function inferDtypes(rows, cols){
  const d={};
  cols.forEach(c=>{
    let ai=true, af=true;
    for(const r of rows){ const v=r[c]; if(v===''||v===undefined) continue; if(!/^-?\d+$/.test(v)) ai=false; if(!/^-?\d+(\.\d+)?$/.test(v)) af=false; }
    d[c]=ai?'int64':af?'float64':'object';
  });
  return d;
}

/* ============================================================
   TAB 4 — DATA AUGMENTATION
============================================================ */
function wireAugmentation(){
  document.getElementById('augCsv').addEventListener('change', e=>{
    const f   = e.target.files[0];
    state.files.augCsv = f || null;
    const box = document.getElementById('augFileDetails');
    const btn = document.getElementById('diagnoseBtn');
    document.getElementById('augDiagnosisPanel').innerHTML = '';
    document.getElementById('augResultsPanel').innerHTML   = '';
    state.aug = { b64:null, filename:null, diagnosis:null, result:null };
    if(f){ box.innerHTML = renderTable([{ Filename:f.name, Size:fmtBytes(f.size), Type:f.type||'—' }]); btn.disabled=false; }
    else  { box.innerHTML=''; btn.disabled=true; }
  });

  document.getElementById('diagnoseBtn').addEventListener('click', async ()=>{
    const statusEl = document.getElementById('augDiagnoseStatus');
    const btn      = document.getElementById('diagnoseBtn');
    const f        = state.files.augCsv;
    if(!f) return;
    btn.disabled = true;
    statusBox(statusEl, 'info', '<span class="spinner"></span> Scanning your data for issues…');
    try{
      const b64 = await fileToBase64(f);
      state.aug.b64 = b64; state.aug.filename = f.name;
      const res = await authedFetch('/diagnose-data', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ csv_base64:b64, csv_filename:f.name, session_id:state.sessionId }) });
      if(!res.ok) throw new Error('HTTP ' + res.status);
      const resp = await res.json();
      if(!resp.success) throw new Error(resp.message || 'Diagnosis failed.');
      const diagnosis = resp.diagnosis;
      state.aug.diagnosis = diagnosis;
      clearStatus(statusEl);
      renderDiagnosis(diagnosis);
    }catch(e){ statusBox(statusEl, 'error', '❌ Diagnosis failed: ' + e.message); }
    btn.disabled = false;
  });
}

function renderDiagnosis(diagnosis){
  const panel = document.getElementById('augDiagnosisPanel');
  const hasIssues = diagnosis.has_issues || false;
  const issues    = diagnosis.issues        || [];
  const recs      = diagnosis.recommendations || [];

  let html = '<div class="panel"><h3>Diagnosis results</h3>';
  html += hasIssues
    ? `<div class="status-msg warn">⚠️ Found ${issues.length} issue(s) in your dataset.</div>`
    : '<div class="status-msg success">✅ No major issues detected.</div>';
  issues.forEach(i=> html += `<div style="font-size:13px;margin:4px 0;">• ${escapeHtml(String(i))}</div>`);

  if(recs.length){
    html += '<label class="field-label" style="margin-top:14px;display:block;">Augmentation plan</label>';
    html += renderTable(recs.map(r=>({ Type:(r.type||'').replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase()), Description:r.description||'', Severity:(r.severity||'—').toUpperCase() })));
  }

  html += `<hr style="border-color:var(--border);margin:18px 0;">
    <label class="field-label" style="display:block;margin-bottom:10px;">Select augmentation options</label>
    <div class="grid-2">
      <div>
        <div class="check-row"><input type="checkbox" id="optImpute" checked><div><span class="ck-label">Impute missing values</span></div></div>
        <div class="check-row"><input type="checkbox" id="optOutliers" checked><div><span class="ck-label">Treat outliers (winsorize)</span></div></div>
        <div class="check-row"><input type="checkbox" id="optDedup" checked><div><span class="ck-label">Remove duplicates</span></div></div>
      </div>
      <div>
        <div class="check-row"><input type="checkbox" id="optTransform"><div><span class="ck-label">Fix skewed distributions (log)</span></div></div>
        <div class="check-row"><input type="checkbox" id="optSynthetic"><div><span class="ck-label">Generate synthetic rows</span><span class="ck-sub">Adds Gaussian-noise rows to expand small datasets.</span></div></div>
      </div>
    </div>
    <button class="btn primary block" id="applyAugBtn" style="margin-top:14px;">Apply augmentation</button>
    <div id="augApplyStatus"></div></div>`;

  panel.innerHTML = html;
  document.getElementById('applyAugBtn').addEventListener('click', applyAugmentation);
}

async function applyAugmentation(){
  const statusEl = document.getElementById('augApplyStatus');
  const btn      = document.getElementById('applyAugBtn');
  btn.disabled   = true;
  statusBox(statusEl, 'info', '<span class="spinner"></span> Augmenting your data…');
  const payload = {
    csv_base64:              state.aug.b64,
    csv_filename:            state.aug.filename,
    session_id:              state.sessionId,
    apply_imputation:        document.getElementById('optImpute').checked,
    apply_outlier_treatment: document.getElementById('optOutliers').checked,
    apply_deduplication:     document.getElementById('optDedup').checked,
    apply_transformations:   document.getElementById('optTransform').checked,
    apply_synthetic_rows:    document.getElementById('optSynthetic').checked
  };
  try{
    const res = await authedFetch('/augment-data', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
    if(!res.ok) throw new Error('HTTP ' + res.status);
    const result = await res.json();
    state.aug.result = result;
    clearStatus(statusEl);
    renderAugResult(result);
  }catch(e){ statusBox(statusEl, 'error', '❌ Augmentation failed: ' + e.message); }
  btn.disabled = false;
}

function renderAugResult(result){
  const panel = document.getElementById('augResultsPanel');
  if(!result.success){ panel.innerHTML=`<div class="panel"><div class="status-msg error">❌ ${escapeHtml(result.message||'Augmentation failed.')}</div></div>`; return; }
  const [origRows,,] = result.original_shape;
  const [augRows, augCols] = result.augmented_shape;
  const delta = augRows - origRows;
  let html = `<div class="panel"><h3>Augmentation complete</h3>
    <div class="metrics-row">
      <div class="metric-card"><div class="m-label">Rows before</div><div class="m-value">${origRows}</div></div>
      <div class="metric-card"><div class="m-label">Rows after</div><div class="m-value">${augRows}</div><div class="m-delta">${delta>=0?'+':''}${delta}</div></div>
      <div class="metric-card"><div class="m-label">Columns</div><div class="m-value">${augCols}</div></div>
    </div>`;
  const log = result.change_log || [];
  if(log.length){ html+=`<details class="expander" open><summary>Change log</summary><div class="expander-body">`; log.forEach(e=> html+=`<div style="font-size:13px;margin:6px 0;"><strong>${escapeHtml(e.step||'')}</strong> — ${escapeHtml(e.detail||'')}</div>`); html+=`</div></details>`; }
  html += `<details class="expander"><summary>Data preview (before vs after)</summary><div class="expander-body"><div class="grid-2">
    <div><label class="field-label">Original</label>${renderTable(result.sample_original)}</div>
    <div><label class="field-label">Augmented</label>${renderTable(result.sample_augmented)}</div>
  </div></div></details>
  <button class="btn primary block" id="downloadAugBtn" style="margin-top:14px;">Download augmented CSV</button>
  <button class="btn block" id="runAugAnalysisBtn" style="margin-top:10px;">Run analysis on augmented data</button>
  <div id="augAnalysisStatus"></div><div id="augAnalysisResult"></div></div>`;
  panel.innerHTML = html;

  document.getElementById('downloadAugBtn').addEventListener('click', ()=>{
    const chars=atob(result.augmented_csv_base64); const bytes=new Uint8Array(chars.length);
    for(let i=0;i<chars.length;i++) bytes[i]=chars.charCodeAt(i);
    const blob=new Blob([bytes],{type:'text/csv'}); const a=document.createElement('a');
    a.href=URL.createObjectURL(blob); a.download=result.augmented_filename||'augmented.csv'; a.click();
  });

  document.getElementById('runAugAnalysisBtn').addEventListener('click', async ()=>{
    const statusEl=document.getElementById('augAnalysisStatus');
    statusBox(statusEl,'info','<span class="spinner"></span> Analyzing augmented data…');
    try{
      const res=await authedFetch('/analyze-data',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({csv_base64:result.augmented_csv_base64,csv_filename:result.augmented_filename,session_id:state.sessionId})});
      if(!res.ok) throw new Error('HTTP '+res.status);
      const analysis=await res.json(); clearStatus(statusEl);
      document.getElementById('augAnalysisResult').innerHTML=renderAnalysis(analysis);
    }catch(e){ statusBox(statusEl,'error','❌ '+e.message); }
  });
}

/* ============================================================
   PROFILE MODAL
============================================================ */
function wireProfileModal(){
  const modal    = document.getElementById('profileModal');
  const openBtn  = document.getElementById('profileBtn');
  const closeBtn = document.getElementById('profileModalClose');

  openBtn.addEventListener('click', ()=>{
    const auth = loadAuth(); const user=(auth&&auth.user)||{}; const name=user.name||'';
    document.getElementById('profileName').value  = name;
    document.getElementById('profileEmail').value = user.email||'';
    document.getElementById('modalName').textContent  = name||'—';
    document.getElementById('modalEmail').textContent = user.email||'—';
    document.getElementById('modalAvatar').textContent= (name.charAt(0)||'U').toUpperCase();
    document.getElementById('profileStatus').innerHTML='';
    document.getElementById('passwordStatus').innerHTML='';
    document.getElementById('profileCurrentPw').value='';
    document.getElementById('pwCurrentPw').value='';
    document.getElementById('pwNewPw').value='';
    document.getElementById('pwConfirmPw').value='';
    modal.classList.add('open');
  });
  closeBtn.addEventListener('click', ()=> modal.classList.remove('open'));
  modal.addEventListener('click', e=>{ if(e.target===modal) modal.classList.remove('open'); });

  document.querySelectorAll('.modal-tab').forEach(tab=>{
    tab.addEventListener('click', ()=>{
      document.querySelectorAll('.modal-tab').forEach(t=>t.classList.remove('active'));
      document.querySelectorAll('.modal-section').forEach(s=>s.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById(tab.dataset.section).classList.add('active');
    });
  });

  document.getElementById('saveProfileBtn').addEventListener('click', async ()=>{
    const statusEl=document.getElementById('profileStatus');
    const name=document.getElementById('profileName').value.trim();
    const email=document.getElementById('profileEmail').value.trim();
    const currentPw=document.getElementById('profileCurrentPw').value;
    if(!name){ statusBox(statusEl,'error','Name cannot be empty.'); return; }
    if(!email||!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)){ statusBox(statusEl,'error','Enter a valid email.'); return; }
    if(!currentPw){ statusBox(statusEl,'warn','Enter your current password to save.'); return; }
    statusBox(statusEl,'info','<span class="spinner"></span> Saving…');
    try{
      const res=await authedFetch('/auth/update-profile',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,email,current_password:currentPw})});
      if(!res.ok){ let d='Update failed.'; try{const e=await res.json();d=e.detail||d;}catch(_){} statusBox(statusEl,'error',escapeHtml(d)); return; }
      const data=await res.json(); const persist=!!localStorage.getItem('dab_auth');
      setAuth({token:data.token,user:data.user},persist);
      document.getElementById('userNameLabel').textContent=data.user.name;
      document.getElementById('userEmailLabel').textContent=data.user.email;
      document.getElementById('userAvatar').textContent=data.user.name.charAt(0).toUpperCase();
      document.getElementById('modalAvatar').textContent=data.user.name.charAt(0).toUpperCase();
      document.getElementById('modalName').textContent=data.user.name;
      document.getElementById('modalEmail').textContent=data.user.email;
      document.getElementById('profileCurrentPw').value='';
      statusBox(statusEl,'success','✅ Profile updated.');
    }catch(e){ statusBox(statusEl,'error','❌ '+e.message); }
  });

  document.getElementById('savePasswordBtn').addEventListener('click', async ()=>{
    const statusEl=document.getElementById('passwordStatus');
    const currentPw=document.getElementById('pwCurrentPw').value;
    const newPw=document.getElementById('pwNewPw').value;
    const confirmPw=document.getElementById('pwConfirmPw').value;
    if(!currentPw){ statusBox(statusEl,'error','Enter your current password.'); return; }
    if(newPw.length<8){ statusBox(statusEl,'error','New password must be at least 8 characters.'); return; }
    if(newPw!==confirmPw){ statusBox(statusEl,'error','New passwords do not match.'); return; }
    statusBox(statusEl,'info','<span class="spinner"></span> Updating password…');
    try{
      const auth=loadAuth(); const user=(auth&&auth.user)||{};
      const res=await authedFetch('/auth/update-profile',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:user.name||'',email:user.email||'',current_password:currentPw,new_password:newPw})});
      if(!res.ok){ let d='Failed.'; try{const e=await res.json();d=e.detail||d;}catch(_){} statusBox(statusEl,'error',escapeHtml(d)); return; }
      const data=await res.json(); const persist=!!localStorage.getItem('dab_auth');
      setAuth({token:data.token,user:data.user},persist);
      document.getElementById('pwCurrentPw').value='';
      document.getElementById('pwNewPw').value='';
      document.getElementById('pwConfirmPw').value='';
      statusBox(statusEl,'success','✅ Password updated.');
    }catch(e){ statusBox(statusEl,'error','❌ '+e.message); }
  });
}

/* ============================================================
   RATE LIMIT BANNER
============================================================ */
function handleRateLimit(tabId){
  const el = document.getElementById(tabId + 'RateWarn');
  if(el){ el.classList.add('visible'); setTimeout(()=> el.classList.remove('visible'), 60000); }
}