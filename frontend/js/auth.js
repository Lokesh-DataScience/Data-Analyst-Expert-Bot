/* ============================================================
   AUTH MODULE
   Endpoints expected from backend:
     POST /auth/login   { email, password }       -> { token, user:{id,name,email} }
     POST /auth/signup  { name, email, password } -> { token, user:{id,name,email} }
   Falls back to demo mode if backend returns 404 or is unreachable.
============================================================ */

const AUTH_STORAGE_KEY = 'dab_auth';

/* ── Storage helpers ── */
function getApiBase(){
  return (document.getElementById('apiBaseInput')?.value || 'http://localhost:8000')
    .trim().replace(/\/$/, '');
}
function setAuth(auth, persist){
  const json = JSON.stringify(auth);
  if(persist) localStorage.setItem(AUTH_STORAGE_KEY, json);
  else sessionStorage.setItem(AUTH_STORAGE_KEY, json);
}
function loadAuth(){
  try{
    return JSON.parse(localStorage.getItem(AUTH_STORAGE_KEY))
        || JSON.parse(sessionStorage.getItem(AUTH_STORAGE_KEY));
  }catch(e){ return null; }
}
function clearAuth(){
  localStorage.removeItem(AUTH_STORAGE_KEY);
  sessionStorage.removeItem(AUTH_STORAGE_KEY);
}
function authToken(){ const a = loadAuth(); return a ? a.token : null; }
function authHeaders(){
  const t = authToken();
  return t ? { 'Authorization': 'Bearer ' + t } : {};
}

/* ── HTML escape ── */
function escapeHtmlAuth(str){
  return String(str).replace(/[&<>"']/g,
    c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

/* ── Switch between login / signup ── */
document.querySelectorAll('[data-switch]').forEach(link=>{
  link.addEventListener('click', e=>{
    e.preventDefault();
    document.querySelectorAll('.auth-form').forEach(f=>f.classList.remove('active'));
    document.getElementById(link.dataset.switch + 'Form').classList.add('active');
  });
});

/* ── Show / hide password ── */
document.querySelectorAll('.pw-toggle').forEach(btn=>{
  btn.addEventListener('click', ()=>{
    const input = document.getElementById(btn.dataset.target);
    const show = input.type === 'password';
    input.type = show ? 'text' : 'password';
    btn.textContent = show ? 'hide' : 'show';
  });
});

document.getElementById('forgotLink')?.addEventListener('click', e=>{
  e.preventDefault();
  alert('Password reset will be available once the auth backend is connected.');
});

/* ============================================================
   LOGIN
============================================================ */
document.getElementById('loginForm').addEventListener('submit', async e=>{
  e.preventDefault();
  const statusEl = document.getElementById('loginStatus');
  const email    = document.getElementById('loginEmail').value.trim();
  const password = document.getElementById('loginPassword').value;
  const remember = document.getElementById('loginRemember').checked;
  const btn      = e.target.querySelector('button[type=submit]');

  btn.disabled = true;
  statusEl.innerHTML = '<div class="status-msg info"><span class="spinner"></span> Signing in...</div>';

  try{
    const res = await fetch(getApiBase() + '/auth/login', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ email, password })
    });

    if(res.ok){
      const data = await res.json();
      setAuth({ token: data.token, user: data.user || { name: email.split('@')[0], email } }, remember);
      enterApp();
      return;
    }

    if(res.status === 404){
      statusEl.innerHTML = '<div class="status-msg warn">Auth backend not found — continuing in demo mode.</div>';
      setAuth({ token:'demo-token', user:{ name: email.split('@')[0] || 'Demo User', email } }, remember);
      setTimeout(enterApp, 700);
      return;
    }

    // Try to parse error detail; res body already consumed above if !ok
    let detail = 'Invalid email or password.';
    try{ const err = await res.json(); detail = err.detail || err.message || detail; }catch(_){}
    statusEl.innerHTML = `<div class="status-msg error">${escapeHtmlAuth(detail)}</div>`;

  }catch(err){
    statusEl.innerHTML = '<div class="status-msg warn">Could not reach the API — continuing in demo mode.</div>';
    setAuth({ token:'demo-token', user:{ name: email.split('@')[0] || 'Demo User', email } }, remember);
    setTimeout(enterApp, 700);
  }
  btn.disabled = false;
});

/* ============================================================
   SIGNUP
============================================================ */
document.getElementById('signupForm').addEventListener('submit', async e=>{
  e.preventDefault();
  const statusEl  = document.getElementById('signupStatus');
  const name      = document.getElementById('signupName').value.trim();
  const email     = document.getElementById('signupEmail').value.trim();
  const password  = document.getElementById('signupPassword').value;
  const btn       = e.target.querySelector('button[type=submit]');

  // Basic client-side validation
  if(!name){
    statusEl.innerHTML = '<div class="status-msg error">Please enter your full name.</div>';
    return;
  }
  if(!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)){
    statusEl.innerHTML = '<div class="status-msg error">Please enter a valid email address.</div>';
    return;
  }
  if(password.length < 8){
    statusEl.innerHTML = '<div class="status-msg error">Password must be at least 8 characters.</div>';
    return;
  }

  btn.disabled = true;
  statusEl.innerHTML = '<div class="status-msg info"><span class="spinner"></span> Creating your account...</div>';

  try{
    const res = await fetch(getApiBase() + '/auth/signup', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ name, email, password })
    });

    if(res.ok){
      const data = await res.json();
      setAuth({ token: data.token, user: data.user || { name, email } }, true);
      enterApp();
      return;
    }

    if(res.status === 404){
      statusEl.innerHTML = '<div class="status-msg warn">Auth backend not found — continuing in demo mode.</div>';
      setAuth({ token:'demo-token', user:{ name, email } }, true);
      setTimeout(enterApp, 700);
      return;
    }

    let detail = 'Could not create account.';
    try{ const err = await res.json(); detail = err.detail || err.message || detail; }catch(_){}
    statusEl.innerHTML = `<div class="status-msg error">${escapeHtmlAuth(detail)}</div>`;

  }catch(err){
    statusEl.innerHTML = '<div class="status-msg warn">Could not reach the API — continuing in demo mode.</div>';
    setAuth({ token:'demo-token', user:{ name, email } }, true);
    setTimeout(enterApp, 700);
  }
  btn.disabled = false;
});

/* ============================================================
   DEMO LOGIN
============================================================ */
document.getElementById('demoLoginBtn').addEventListener('click', ()=>{
  setAuth({ token:'demo-token', user:{ name:'Demo User', email:'demo@dataanalystbot.ai' } }, false);
  enterApp();
});

/* ============================================================
   LOGOUT
============================================================ */
document.getElementById('logoutBtn').addEventListener('click', ()=>{
  if(!confirm('Sign out of DataAnalystBot?')) return;
  clearAuth();

  // ── CRITICAL: reset initialized so initApp() runs fresh on next login ──
  if(typeof state !== 'undefined') state.initialized = false;

  document.body.classList.remove('authenticated');

  // Reset auth forms
  document.querySelectorAll('.auth-form').forEach(f=>f.classList.remove('active'));
  document.getElementById('loginForm').classList.add('active');
  document.getElementById('loginEmail').value = '';
  document.getElementById('loginPassword').value = '';
  document.getElementById('loginStatus').innerHTML = '';
  document.getElementById('signupStatus').innerHTML = '';
});

/* ============================================================
   ENTER APP — called after any successful auth
============================================================ */
function enterApp(){
  const auth = loadAuth();
  if(!auth) return;

  const user = auth.user || {};
  const name = user.name || user.email || 'User';

  document.getElementById('userNameLabel').textContent  = name;
  document.getElementById('userEmailLabel').textContent = user.email || '';
  document.getElementById('userAvatar').textContent     = name.trim().charAt(0).toUpperCase() || 'U';

  document.body.classList.add('authenticated');

  // Delegate to app.js
  if(typeof onAppEnter === 'function') onAppEnter();
}

/* ── Auto-login if a session token is already stored ── */
(function(){
  const auth = loadAuth();
  if(auth) enterApp();
})();