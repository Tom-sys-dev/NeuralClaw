function switchTab(tab) {
  document.getElementById('form-login').style.display    = tab === 'login'    ? 'block' : 'none';
  document.getElementById('form-register').style.display = tab === 'register' ? 'block' : 'none';
  document.getElementById('tab-login').classList.toggle('active', tab === 'login');
  document.getElementById('tab-register').classList.toggle('active', tab === 'register');
  clearMessages();
}
function clearMessages() {
  ['login-error','login-success','register-error','register-success'].forEach(id => {
    const el = document.getElementById(id);
    el.className = el.className.replace(' visible', '');
  });
}
function showMsg(id, msg) {
  const el = document.getElementById(id);
  el.textContent = msg;
  el.classList.add('visible');
}

async function doLogin() {
  const btn      = document.getElementById('login-btn');
  const username = document.getElementById('login-username').value.trim();
  const password = document.getElementById('login-password').value.trim();
  clearMessages();
  if (!username || !password) return showMsg('login-error', 'Veuillez remplir tous les champs');
  btn.disabled = true; btn.textContent = 'Connexion…';
  try {
    const res  = await fetch('/api/auth/login', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    const data = await res.json();
    if (data.ok) {
      showMsg('login-success', 'Bienvenue, ' + data.username + ' !');
      setTimeout(() => window.location = '/', 800);
    } else {
      showMsg('login-error', data.error || 'Erreur inconnue');
      btn.disabled = false; btn.textContent = 'Se connecter →';
    }
  } catch {
    showMsg('login-error', 'Erreur réseau');
    btn.disabled = false; btn.textContent = 'Se connecter →';
  }
}

async function doRegister() {
  const btn      = document.getElementById('register-btn');
  const username = document.getElementById('reg-username').value.trim();
  const email    = document.getElementById('reg-email').value.trim();
  const password = document.getElementById('reg-password').value.trim();
  const confirm  = document.getElementById('reg-confirm').value.trim();
  clearMessages();
  if (!username || !password || !confirm) return showMsg('register-error', 'Veuillez remplir les champs obligatoires');
  if (password !== confirm) return showMsg('register-error', 'Les mots de passe ne correspondent pas');
  if (password.length < 6) return showMsg('register-error', 'Le mot de passe doit faire au moins 6 caractères');
  btn.disabled = true; btn.textContent = 'Création…';
  try {
    const res  = await fetch('/api/auth/register', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, email })
    });
    const data = await res.json();
    if (data.ok) {
      showMsg('register-success', 'Compte créé ! Redirection…');
      setTimeout(() => window.location = '/', 900);
    } else {
      showMsg('register-error', data.error || 'Erreur inconnue');
      btn.disabled = false; btn.textContent = 'Créer mon compte →';
    }
  } catch {
    showMsg('register-error', 'Erreur réseau');
    btn.disabled = false; btn.textContent = 'Créer mon compte →';
  }
}

document.addEventListener('keydown', e => {
  if (e.key !== 'Enter') return;
  if (document.getElementById('form-login').style.display !== 'none') doLogin();
  else doRegister();
});
