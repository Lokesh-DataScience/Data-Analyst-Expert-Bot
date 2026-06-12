/* ============================================================
   AUTH MODULE
   Talks to (future) backend endpoints:
     POST /auth/login   { email, password }        -> { token, user:{name,email} }
     POST /auth/signup  { name, email, password }  -> { token, user:{name,email} }
   Falls back to a local "demo mode" if the backend auth
   endpoints aren't available yet, so the UI is usable
   while auth is being wired up server-side.
============================================================ */

const AUTH_STORAGE_KEY = 'dab_auth';

function getApiBase(){
  return (document.getElementById('apiBaseInput')?.value || 'http://localhost:8000').trim().replace(/\/$/, '');
}

function getAuth(){
  try{ return JSON.parse(localStorage.getItem(AUTH_STORAGE_KEY)); }
  catch(e){ return null; }
}
function setAuth(auth, persist){
  if(persist) localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(auth));
  else sessionStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(auth));
}
function clearAuth(){
  localStorage.removeItem(AUTH_STORAGE_KEY);
  sessionStorage.removeItem(AUTH_STORAGE_KEY);
}
function loadAuth(){
  try{
    return JSON.parse(localStorage.getItem(AUTH_STORAGE_KEY)) ||
           JSON.parse(sessionStorage.getItem(AUTH_STORAGE_KEY));
  }catch(e){ return null; }
}
function authToken(){
  const a = loadAuth();
  return a ? a.token : null;
}
function authHeaders(){
  const t = authToken();
  return t ? { 'Authorization': 'Bearer ' + t } : {};
}

/* ============================================================
   UI: switch between login / signup forms
============================================================ */
document.querySelectorAll('[data-switch]').forEach(link=>{
  link.addEventListener('click', e=>{
    e.preventDefault();
    const target = link.dataset.switch;
    document.querySelectorAll('.auth-form').forEach(f=>f.classList.remove('active'));
    document.getElementById(target+'Form').classList.add('active');
  });
});

/* show / hide password */
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
  alert('Password reset isn\'t wired up yet — this will be available once the auth backend is connected.');
});

/* ============================================================
   LOGIN
============================================================ */
document.getElementById('loginForm').addEventListener('submit', async e=>{
  e.preventDefault();
  const status = document.getElementById('loginStatus');
  const email = document.getElementById('loginEmail').value.trim();
  const password = document.getElementById('loginPassword').value;
  const remember = document.getElementById('loginRemember').checked;
  const btn = e.target.querySelector('button[type=submit]');

  btn.disabled = true;
  status.innerHTML = '<div class="status-msg info"><span class="spinner"></span> Signing in...</div>';

  try{
    const res = await fetch(getApiBase()+'/auth/login', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ email, password })
    });

    if(res.ok){
      const data = await res.json();
      setAuth({ token:data.token, user:data.user || {name:email.split('@')[0], email} }, remember);
      enterApp();
      return;
    }

    if(res.status === 404){
      // Backend auth not implemented yet -> demo fallback
      status.innerHTML = '<div class="status-msg warn">Auth backend not found — continuing in demo mode.</div>';
      setAuth({ token:'demo-token', user:{ name:email.split('@')[0] || 'Demo User', email } }, remember);
      setTimeout(enterApp, 700);
      return;
    }

    const err = await res.json().catch(()=>({}));
    status.innerHTML = `<div class="status-msg error">${escapeHtmlAuth(err.detail || err.message || 'Invalid email or password.')}</div>`;
  }catch(err){
    status.innerHTML = '<div class="status-msg warn">Could not reach the API — continuing in demo mode.</div>';
    setAuth({ token:'demo-token', user:{ name:email.split('@')[0] || 'Demo User', email } }, remember);
    setTimeout(enterApp, 700);
  }
  btn.disabled = false;
});

/* ============================================================
   SIGNUP
============================================================ */
document.getElementById('signupForm').addEventListener('submit', async e=>{
  e.preventDefault();
  const status = document.getElementById('signupStatus');
  const name = document.getElementById('signupName').value.trim();
  const email = document.getElementById('signupEmail').value.trim();
  const password = document.getElementById('signupPassword').value;
  const btn = e.target.querySelector('button[type=submit]');

  btn.disabled = true;
  status.innerHTML = '<div class="status-msg info"><span class="spinner"></span> Creating your account...</div>';

  try{
    const res = await fetch(getApiBase()+'/auth/signup', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ name, email, password })
    });

    if(res.ok){
      const data = await res.json();
      setAuth({ token:data.token, user:data.user || {name, email} }, true);
      enterApp();
      return;
    }

    if(res.status === 404){
      status.innerHTML = '<div class="status-msg warn">Auth backend not found — continuing in demo mode.</div>';
      setAuth({ token:'demo-token', user:{ name, email } }, true);
      setTimeout(enterApp, 700);
      return;
    }

    const err = await res.json().catch(()=>({}));
    status.innerHTML = `<div class="status-msg error">${escapeHtmlAuth(err.detail || err.message || 'Could not create account.')}</div>`;
  }catch(err){
    status.innerHTML = '<div class="status-msg warn">Could not reach the API — continuing in demo mode.</div>';
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
  document.body.classList.remove('authenticated');
  document.getElementById('loginForm').classList.add('active');
  document.getElementById('signupForm').classList.remove('active');
  document.getElementById('loginEmail').value='';
  document.getElementById('loginPassword').value='';
});

/* ============================================================
   ENTER APP
============================================================ */
function enterApp(){
  const auth = loadAuth();
  if(!auth) return;
  const user = auth.user || {};
  const name = user.name || user.email || 'User';
  document.getElementById('userNameLabel').textContent = name;
  document.getElementById('userEmailLabel').textContent = user.email || '';
  document.getElementById('userAvatar').textContent = name.trim().charAt(0).toUpperCase() || 'U';
  document.body.classList.add('authenticated');
  if(typeof onAppEnter === 'function') onAppEnter();
}

function escapeHtmlAuth(str){
  return String(str).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

/* Auto-login if a session already exists */
(function(){
  const auth = loadAuth();
  if(auth) enterApp();
})();
