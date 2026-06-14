/* ============================================================
   AUTH MODULE  —  auth.js
   Handles: login, signup, forgot password, reset password,
            demo login, logout, profile modal open/close.
   Profile save logic lives in app.js (needs authedFetch).
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
  else        sessionStorage.setItem(AUTH_STORAGE_KEY, json);
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
function authToken(){  const a = loadAuth(); return a ? a.token : null; }
function authHeaders(){
  const t = authToken();
  return t ? { 'Authorization': 'Bearer ' + t } : {};
}
function escapeHtmlAuth(str){
  return String(str).replace(/[&<>"']/g,
    c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

/* ── Form switcher (login / signup / forgot / reset) ── */
function showAuthForm(name){
  document.querySelectorAll('.auth-form').forEach(f=>f.classList.remove('active'));
  document.getElementById(name + 'Form').classList.add('active');
}

document.querySelectorAll('[data-switch]').forEach(link=>{
  link.addEventListener('click', e=>{
    e.preventDefault();
    showAuthForm(link.dataset.switch);
  });
});

/* ── Show/hide password toggles ── */
document.querySelectorAll('.pw-toggle').forEach(btn=>{
  btn.addEventListener('click', ()=>{
    const input = document.getElementById(btn.dataset.target);
    const show  = input.type === 'password';
    input.type  = show ? 'text' : 'password';
    btn.textContent = show ? 'hide' : 'show';
  });
});

/* ── Forgot password link ── */
document.getElementById('forgotLink').addEventListener('click', e=>{
  e.preventDefault();
  showAuthForm('forgot');
});

/* ── Check URL for ?reset_token= on page load ── */
(function checkResetToken(){
  const params = new URLSearchParams(window.location.search);
  const token  = params.get('reset_token');
  if(token){
    document.getElementById('resetForm').dataset.token = token;
    showAuthForm('reset');
  }
})();

/* ============================================================
   LOGIN
============================================================ */
document.getElementById('loginForm').addEventListener('submit', async e=>{
  e.preventDefault();
  const statusEl  = document.getElementById('loginStatus');
  const email     = document.getElementById('loginEmail').value.trim();
  const password  = document.getElementById('loginPassword').value;
  const remember  = document.getElementById('loginRemember').checked;
  const btn       = e.target.querySelector('button[type=submit]');

  if(!email || !password){
    statusEl.innerHTML = '<div class="status-msg error">Please enter your email and password.</div>';
    return;
  }
  btn.disabled = true;
  statusEl.innerHTML = '<div class="status-msg info"><span class="spinner"></span> Signing in…</div>';

  try{
    const res = await fetch(getApiBase() + '/auth/login', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ email, password })
    });
    if(res.ok){
      const data = await res.json();
      setAuth({ token: data.token, user: data.user || { name: email.split('@')[0], email } }, remember);
      enterApp(); return;
    }
    if(res.status === 404){
      statusEl.innerHTML = '<div class="status-msg warn">Auth backend not found — continuing in demo mode.</div>';
      setAuth({ token:'demo-token', user:{ name: email.split('@')[0] || 'Demo User', email } }, remember);
      setTimeout(enterApp, 700); return;
    }
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
  const statusEl = document.getElementById('signupStatus');
  const name     = document.getElementById('signupName').value.trim();
  const email    = document.getElementById('signupEmail').value.trim();
  const password = document.getElementById('signupPassword').value;
  const btn      = e.target.querySelector('button[type=submit]');

  if(!name){
    statusEl.innerHTML = '<div class="status-msg error">Please enter your full name.</div>'; return;
  }
  if(!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)){
    statusEl.innerHTML = '<div class="status-msg error">Please enter a valid email address.</div>'; return;
  }
  if(password.length < 8){
    statusEl.innerHTML = '<div class="status-msg error">Password must be at least 8 characters.</div>'; return;
  }

  btn.disabled = true;
  statusEl.innerHTML = '<div class="status-msg info"><span class="spinner"></span> Creating your account…</div>';

  try{
    const res = await fetch(getApiBase() + '/auth/signup', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ name, email, password })
    });
    if(res.ok){
      const data = await res.json();
      setAuth({ token: data.token, user: data.user || { name, email } }, true);
      enterApp(); return;
    }
    if(res.status === 404){
      statusEl.innerHTML = '<div class="status-msg warn">Auth backend not found — continuing in demo mode.</div>';
      setAuth({ token:'demo-token', user:{ name, email } }, true);
      setTimeout(enterApp, 700); return;
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
   FORGOT PASSWORD
============================================================ */
document.getElementById('forgotForm').addEventListener('submit', async e=>{
  e.preventDefault();
  const statusEl = document.getElementById('forgotStatus');
  const email    = document.getElementById('forgotEmail').value.trim();
  const btn      = e.target.querySelector('button[type=submit]');

  if(!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)){
    statusEl.innerHTML = '<div class="status-msg error">Please enter a valid email address.</div>'; return;
  }
  btn.disabled = true;
  statusEl.innerHTML = '<div class="status-msg info"><span class="spinner"></span> Sending reset link…</div>';

  try{
    const res = await fetch(getApiBase() + '/auth/forgot-password', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ email })
    });
    if(res.ok){
      statusEl.innerHTML = '<div class="status-msg success">✅ If that email is registered, a reset link has been sent. Check your inbox (and spam folder).</div>';
      // Dev mode: the link is printed to the uvicorn terminal if email isn't configured
    } else {
      statusEl.innerHTML = '<div class="status-msg error">Something went wrong. Please try again.</div>';
    }
  }catch(err){
    statusEl.innerHTML = '<div class="status-msg error">Could not reach the API.</div>';
  }
  btn.disabled = false;
});

/* ============================================================
   RESET PASSWORD
============================================================ */
document.getElementById('resetForm').addEventListener('submit', async e=>{
  e.preventDefault();
  const statusEl   = document.getElementById('resetStatus');
  const newPw      = document.getElementById('resetPassword').value;
  const token      = document.getElementById('resetForm').dataset.token || '';
  const btn        = e.target.querySelector('button[type=submit]');

  if(newPw.length < 8){
    statusEl.innerHTML = '<div class="status-msg error">Password must be at least 8 characters.</div>'; return;
  }
  if(!token){
    statusEl.innerHTML = '<div class="status-msg error">Missing reset token. Please use the link from your email.</div>'; return;
  }

  btn.disabled = true;
  statusEl.innerHTML = '<div class="status-msg info"><span class="spinner"></span> Updating password…</div>';

  try{
    const res = await fetch(getApiBase() + '/auth/reset-password', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ token, new_password: newPw })
    });
    if(res.ok){
      statusEl.innerHTML = '<div class="status-msg success">✅ Password updated! You can now sign in.</div>';
      // Clean token from URL
      window.history.replaceState({}, document.title, window.location.pathname);
      setTimeout(()=> showAuthForm('login'), 2000);
    } else {
      let detail = 'Reset failed.';
      try{ const err = await res.json(); detail = err.detail || detail; }catch(_){}
      statusEl.innerHTML = `<div class="status-msg error">${escapeHtmlAuth(detail)}</div>`;
    }
  }catch(err){
    statusEl.innerHTML = '<div class="status-msg error">Could not reach the API.</div>';
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
  if(typeof state !== 'undefined') state.initialized = false;
  document.body.classList.remove('authenticated');
  // Close profile modal if open
  document.getElementById('profileModal').classList.remove('open');
  // Reset forms
  document.querySelectorAll('.auth-form').forEach(f=>f.classList.remove('active'));
  document.getElementById('loginForm').classList.add('active');
  document.getElementById('loginEmail').value    = '';
  document.getElementById('loginPassword').value = '';
  document.getElementById('loginStatus').innerHTML  = '';
  document.getElementById('signupStatus').innerHTML = '';
  document.getElementById('forgotStatus').innerHTML = '';
});

/* ============================================================
   ENTER APP
============================================================ */
function enterApp(){
  const auth = loadAuth();
  if(!auth) return;
  const user = auth.user || {};
  const name = user.name || user.email || 'User';

  document.getElementById('userNameLabel').textContent  = name;
  document.getElementById('userEmailLabel').textContent = user.email || '';
  document.getElementById('userAvatar').textContent     = name.trim().charAt(0).toUpperCase() || 'U';

  // Populate profile modal fields
  document.getElementById('modalAvatar').textContent = name.trim().charAt(0).toUpperCase() || 'U';
  document.getElementById('modalName').textContent   = name;
  document.getElementById('modalEmail').textContent  = user.email || '';
  document.getElementById('profileName').value       = name;
  document.getElementById('profileEmail').value      = user.email || '';

  document.body.classList.add('authenticated');
  if(typeof onAppEnter === 'function') onAppEnter();
}

/* ── Auto-login if session token already exists ── */
(function(){
  const auth = loadAuth();
  if(auth) enterApp();
})();