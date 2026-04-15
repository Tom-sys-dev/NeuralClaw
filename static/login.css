/* ===================================================
   GLOBALS
   =================================================== */
let hasContext    = false;
let skills        = [];
let attachedFiles = [];
let contextSaveTimeout;
let currentUser   = null;
let warRoomData   = null;  // stocke la dernière synthèse War Room
let currentLang   = 'fr';
let mermaidCounter = 0;

/* ===================================================
   AUTH ZONE
   =================================================== */
function renderAuthZone() {
  const zone = document.getElementById('auth-zone');
  if (currentUser) {
    const initial = currentUser.username.charAt(0).toUpperCase();
    zone.innerHTML = `
      <div class="user-menu-wrapper" id="user-menu-wrapper">
        <button class="user-avatar-btn" onclick="toggleUserMenu(event)">
          <div class="avatar-circle">${escHtml(initial)}</div>
          <span>${escHtml(currentUser.username)}</span>
          <span style="font-size:10px;color:var(--muted);margin-left:2px">▾</span>
        </button>
        <div class="user-dropdown" id="user-dropdown">
          <div class="user-dropdown-header">
            <div class="uname">${escHtml(currentUser.username)}</div>
            <div class="uemail">${escHtml(currentUser.email || 'Pas d\'email renseigné')}</div>
          </div>
          <button class="dropdown-item" onclick="exportChat()">⬇ Exporter la conversation</button>
          <button class="dropdown-item danger" onclick="doLogout()">↩ Se déconnecter</button>
        </div>
      </div>`;
  } else {
    zone.innerHTML = `
      <button class="login-btn-topbar" onclick="window.location='/login'">
        👤 Se connecter
      </button>`;
  }
}

function toggleUserMenu(e) {
  e.stopPropagation();
  document.getElementById('user-dropdown').classList.toggle('open');
}

document.addEventListener('click', () => {
  const dd = document.getElementById('user-dropdown');
  if (dd) dd.classList.remove('open');
});

async function doLogout() {
  await fetch('/api/auth/logout', { method: 'POST' });
  currentUser = null;
  renderAuthZone();
  showToast('Déconnecté', 'success');
}

async function loadUser() {
  try {
    const res  = await fetch('/api/auth/me');
    const data = await res.json();
    currentUser = data.logged_in
      ? { username: data.username, email: data.email, created_at: data.created_at }
      : null;
  } catch {
    currentUser = null;
  }
  renderAuthZone();
}

/* ===================================================
   TEXTAREA + ENTER
   =================================================== */
const msgInput = document.getElementById('message-input');
msgInput.addEventListener('input', () => {
  msgInput.style.height = 'auto';
  msgInput.style.height = Math.min(msgInput.scrollHeight, 160) + 'px';
});
msgInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

/* ===================================================
   CONTEXT AUTO-SAVE
   =================================================== */
function saveContext() {
  const prompt = document.getElementById('system-prompt').value.trim();
  const model  = document.getElementById('model-select').value;
  fetch('/api/context', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ system_prompt: prompt, model })
  }).then(() => {
    hasContext = !!prompt;
    document.getElementById('context-indicator').style.display = hasContext ? 'flex' : 'none';
    document.getElementById('stat-ctx').textContent = hasContext ? 'Oui' : 'Non';
    document.getElementById('model-badge').textContent = model;
  }).catch(console.error);
}
function debouncedSaveContext() {
  clearTimeout(contextSaveTimeout);
  contextSaveTimeout = setTimeout(saveContext, 1000);
}
document.getElementById('model-select').addEventListener('change', saveContext);

/* ===================================================
   LANGUAGE
   =================================================== */
function setLang(lang) {
  currentLang = lang;
  // Persist silently via chat route on next message, also call context endpoint
  fetch('/api/context', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ lang })
  }).catch(console.error);
  showToast('Langue : ' + document.getElementById('lang-select').options[document.getElementById('lang-select').selectedIndex].text, 'success');
}

/* ===================================================
   SKILLS
   =================================================== */
function openSkillModal() {
  document.getElementById('skill-modal').classList.add('active');
  document.getElementById('skill-input').value = '';
  setTimeout(() => document.getElementById('skill-input').focus(), 100);
}
function closeSkillModal() { document.getElementById('skill-modal').classList.remove('active'); }
async function addSkill() {
  const name = document.getElementById('skill-input').value.trim();
  if (!name) return showToast('Veuillez entrer un nom de skill', 'error');
  if (skills.includes(name)) return showToast('Ce skill existe déjà', 'error');
  skills.push(name); renderSkills(); closeSkillModal();
  await syncSkills(); showToast('Skill "' + name + '" ajouté', 'success');
}
async function removeSkill(name) {
  skills = skills.filter(s => s !== name); renderSkills();
  await syncSkills(); showToast('Skill supprimé', 'success');
}
async function syncSkills() {
  await fetch('/api/skills', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ skills }) });
}
function renderSkills() {
  const container = document.getElementById('skills-list');
  container.innerHTML = skills.map(s => `
    <div class="skill-tag">
      <span class="skill-name">✨ ${escHtml(s)}</span>
      <button class="skill-remove" onclick="removeSkill('${escHtml(s)}')" title="Supprimer">✕</button>
    </div>`).join('');
  const has = skills.length > 0;
  document.getElementById('skills-indicator').style.display = has ? 'flex' : 'none';
  document.getElementById('stat-skills').textContent = skills.length;
}

/* ===================================================
   FILE HANDLING
   =================================================== */
function handleFileSelect(event) {
  Array.from(event.target.files).forEach(f => {
    if (f.size > 5 * 1024 * 1024) return showToast(f.name + ' dépasse 5MB', 'error');
    if (attachedFiles.find(x => x.name === f.name)) return showToast(f.name + ' déjà ajouté', 'error');
    attachedFiles.push(f);
  });
  renderAttachedFiles(); event.target.value = '';
}
function renderAttachedFiles() {
  const c = document.getElementById('file-attach-area');
  c.innerHTML = attachedFiles.map(f => `
    <div class="file-chip">
      📄 ${escHtml(f.name)} <small>(${(f.size/1024).toFixed(1)}KB)</small>
      <button class="remove" onclick="removeAttached('${escHtml(f.name)}')">✕</button>
    </div>`).join('');
  c.className = attachedFiles.length ? 'visible' : '';
}
function removeAttached(name) {
  attachedFiles = attachedFiles.filter(f => f.name !== name);
  renderAttachedFiles();
}

/* ===================================================
   SEND MESSAGE
   =================================================== */
async function sendMessage() {
    const text = msgInput.value.trim();
    if (!text && attachedFiles.length === 0) return;
    msgInput.value = ''; msgInput.style.height = 'auto';
    document.getElementById('welcome')?.remove();

    let display = text || '[Fichier joint]';
    if (attachedFiles.length) display += '\n📎 ' + attachedFiles.map(f => f.name).join(', ');
    appendMessage('user', display);
    setTyping(true);

    try {
        const fd = new FormData();
        fd.append('message', text);
        fd.append('lang', currentLang);
        attachedFiles.forEach(f => fd.append('files', f));
        attachedFiles = []; renderAttachedFiles();

        const res = await fetch('/api/chat', { method: 'POST', body: fd });
        const data = await res.json();
        setTyping(false);

        if (data.error) return showToast(data.error, 'error');

        if (data.ignored_files && data.ignored_files.length) {
            data.ignored_files.forEach(fileReason => {
                showToast(`Fichier ignoré: ${fileReason}`, 'warn');
            });
        }

        appendMessage('bot', data.reply);
        updateStats(data.total_messages, data.estimated_tokens);
    } catch {
        setTyping(false);
        showToast('Erreur réseau', 'error');
    }
}

/* ===================================================
   RENDER MESSAGE
   =================================================== */
function appendMessage(role, text) {
  const container = document.getElementById('messages');
  const typing    = document.getElementById('typing');
  const wrapper   = document.createElement('div');
  wrapper.className = `msg-wrapper ${role}`;

  const avatar = document.createElement('div');
  avatar.className = `avatar ${role}`;
  avatar.textContent = role === 'user'
    ? (currentUser ? currentUser.username.charAt(0).toUpperCase() : '🧑')
    : '🤖';

  const inner  = document.createElement('div');
  const bubble = document.createElement('div');
  bubble.className = `bubble ${role}`;

  if (role === 'bot') {
    // Traiter les blocs Mermaid séparément du reste du markdown
    bubble.innerHTML = renderWithMermaid(text);
    bubble.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
    // Rendre les diagrammes Mermaid
    renderMermaidInElement(bubble);
  } else {
    bubble.innerHTML = escHtml(text);
  }

  const time = document.createElement('div');
  time.className   = 'msg-time';
  time.textContent = new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
  if (role === 'user') time.style.textAlign = 'right';

  inner.append(bubble, time);
  if (role === 'user') wrapper.append(inner, avatar);
  else                 wrapper.append(avatar, inner);

  container.insertBefore(wrapper, typing);
  container.scrollTop = container.scrollHeight;
}

function renderWithMermaid(text) {
  // Remplacer les blocs ```mermaid par des div mermaid
  return marked.parse(text.replace(/```mermaid\n?([\s\S]*?)```/g, (match, code) => {
    const id = `mermaid-${++mermaidCounter}`;
    return `<div class="mermaid-wrapper"><div class="mermaid" id="${id}">${escHtml(code.trim())}</div></div>`;
  }));
}

async function renderMermaidInElement(el) {
  if (typeof mermaid === 'undefined') return;  // mermaid non chargé
  const nodes = el.querySelectorAll('.mermaid');
  for (const node of nodes) {
    try {
      const code = node.textContent;
      const { svg } = await mermaid.render(`mermaid-render-${Date.now()}`, code);
      node.innerHTML = svg;
    } catch (e) {
      node.innerHTML = `<pre style="color:var(--danger);font-size:11px;">Erreur diagramme: ${e.message}</pre>`;
    }
  }
}

function setTyping(on) {
  document.getElementById('typing').className = on ? 'visible' : '';
  document.getElementById('send-btn').disabled = on;
  if (on) document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
}
function updateStats(msgs, tokens) {
  document.getElementById('stat-msgs').textContent   = msgs   ?? '–';
  document.getElementById('stat-tokens').textContent = tokens ?? '–';
}

/* ===================================================
   CLEAR & EXPORT
   =================================================== */
async function clearChat() {
  if (!confirm('Effacer toute la mémoire de conversation ?')) return;
  await fetch('/api/clear', { method: 'POST' });
  document.getElementById('messages').innerHTML = `
    <div id="welcome">
      <h2>Conversation effacée ✓</h2><p>La mémoire a été réinitialisée.</p>
    </div>
    <div id="typing">
      <div class="avatar bot">🤖</div>
      <div class="typing-dots"><span></span><span></span><span></span></div>
    </div>`;
  updateStats(0, 0);
  showToast('Mémoire effacée', 'success');
}

async function exportChat() {
  const res  = await fetch('/api/history');
  const data = await res.json();
  if (!data.messages?.length) return showToast('Aucun message à exporter', 'error');

  let md = '# Conversation — ' + new Date().toLocaleString('fr-FR') + '\n\n';
  if (currentUser) md += '> **Utilisateur :** ' + currentUser.username + '\n\n';
  if (data.system_prompt) md += '> **Contexte :** ' + data.system_prompt + '\n\n';
  if (data.skills?.length) md += '> **Skills :** ' + data.skills.join(', ') + '\n\n---\n\n';
  data.messages.forEach(m => {
    md += (m.role === 'user' ? '**Vous**' : '**IA**') + '\n\n' + m.content + '\n\n---\n\n';
  });
  const a = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(new Blob([md], { type: 'text/markdown' })),
    download: 'conversation_' + Date.now() + '.md'
  });
  a.click();
  showToast('Exporté', 'success');
}

/* ===================================================
   ⚔️ WAR ROOM
   =================================================== */
function openWarRoom() {
  document.getElementById('warroom-overlay').classList.add('active');
  // Pré-remplir avec le message en cours si disponible
  const msg = document.getElementById('message-input').value.trim();
  if (msg) document.getElementById('warroom-input').value = msg;
}
function closeWarRoom() {
  document.getElementById('warroom-overlay').classList.remove('active');
}

function addThought(text, highlight = false) {
  const lines = document.getElementById('thought-lines');
  const d = new Date();
  const t = d.getHours().toString().padStart(2,'0') + ':' + d.getMinutes().toString().padStart(2,'0') + ':' + d.getSeconds().toString().padStart(2,'0');
  const line = document.createElement('div');
  line.className = 'thought-line' + (highlight ? ' highlight' : '');
  line.innerHTML = `<span class="tl-time">${t}</span><span class="tl-text">${escHtml(text)}</span>`;
  lines.appendChild(line);
  lines.scrollTop = lines.scrollHeight;
}

function setProgressStep(step) {
  // step: 'pm' | 'analyse' | 'debat' | 'synthese' | 'code' | 'done'
  const steps = ['pm', 'analyse', 'debat', 'synthese', 'code'];
  const idx = steps.indexOf(step);
  steps.forEach((s, i) => {
    const el = document.getElementById('ps-' + s);
    if (!el) return;
    if (i < idx) el.className = 'progress-step done';
    else if (i === idx) el.className = 'progress-step active';
    else el.className = 'progress-step';
  });
}

function setExpertState(idx, state, expert) {
  const card = document.getElementById('expert-' + idx);
  if (!card) return;
  card.className = 'expert-card ' + state;
  if (expert) {
    card.querySelector('.expert-emoji').textContent = expert.emoji || '🤖';
    card.querySelector('.expert-role').textContent  = expert.role;
    card.querySelector('.expert-specialty').textContent = expert.specialty;
  }
  const status = card.querySelector('.expert-status');
  status.className = 'expert-status ' + state;
  status.textContent = state === 'thinking' ? '⚡ En réflexion…' :
                       state === 'done'     ? '✓ Terminé'        : '⏳ En attente';
}

/* ===================================================
   EXPERT SELECTOR
   =================================================== */
let selectedExperts = [];

function toggleExpert(btn) {
  const id = btn.dataset.id;
  if (btn.classList.contains('selected')) {
    btn.classList.remove('selected');
    selectedExperts = selectedExperts.filter(e => e !== id);
  } else {
    if (selectedExperts.length >= 3) {
      showToast('Maximum 3 experts sélectionnables', 'warn');
      return;
    }
    btn.classList.add('selected');
    selectedExperts.push(id);
  }
  const hint = document.getElementById('expert-selection-hint');
  if (selectedExperts.length === 0) {
    hint.textContent = '👆 Sélectionne 1, 2 ou 3 experts. Si < 3 sélectionnés, l\'IA complète automatiquement.';
    hint.style.color = 'var(--muted)';
  } else {
    hint.textContent = `✅ ${selectedExperts.length}/3 expert(s) sélectionné(s)${selectedExperts.length < 3 ? ' — l\'IA complétera les profils manquants' : ' — démarrage complet avec tes experts'}`;
    hint.style.color = 'var(--war)';
  }
}

async function launchWarRoom() {
  const query = document.getElementById('warroom-input').value.trim();
  if (!query) return showToast('Décris ton problème d\'abord', 'error');

  const model = document.getElementById('model-select').value;
  const btn   = document.getElementById('wr-launch-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Analyse…';

  // Reset UI
  document.getElementById('thought-lines').innerHTML = '';
  document.getElementById('proposals-section').style.display  = 'none';
  document.getElementById('synthesis-section').style.display  = 'none';
  document.getElementById('programmer-section').style.display = 'none';
  document.getElementById('pm-section').style.display         = 'none';
  [0,1,2].forEach(i => setExpertState(i, 'waiting', null));
  setProgressStep('pm');
  addThought('📋 Chef de Projet — élaboration du plan préliminaire…', true);

  try {
    addThought('📡 Envoi de la requête au serveur…');
    if (selectedExperts.length > 0) {
      addThought(`🎭 Experts sélectionnés : ${selectedExperts.join(', ')}`);
    } else {
      addThought('🤖 Mode auto — l\'IA choisira les meilleurs experts');
    }

    const res  = await fetch('/api/warroom', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, model, lang: currentLang, selected_experts: selectedExperts })
    });
    const data = await res.json();

    if (data.error) {
      showToast(data.error, 'error');
      btn.disabled = false; btn.textContent = '🚀 Lancer';
      return;
    }

    warRoomData = data;

    // ── Afficher le plan du Chef de Projet ──
    const pm = data.pm || {};
    if (pm.objective) {
      document.getElementById('pm-objective').textContent = '🎯 ' + pm.objective;
      const stepsEl = document.getElementById('pm-steps');
      stepsEl.innerHTML = (pm.action_plan || []).map(s => `
        <div class="pm-step">
          <span class="step-num">${s.step}</span>
          <span style="flex:1">${escHtml(s.action)}</span>
          <span class="step-prio ${escHtml(s.priority || 'moyenne')}">${escHtml(s.priority || 'moyenne')}</span>
          <span class="step-dur">${escHtml(s.duration || '')}</span>
        </div>`).join('');
      const risksEl = document.getElementById('pm-risks');
      risksEl.innerHTML = (pm.key_risks || []).map(r => `<span class="pm-risk-tag">⚠️ ${escHtml(r)}</span>`).join('');
      document.getElementById('pm-section').style.display = 'block';
      addThought('✅ Plan préliminaire prêt — ' + (pm.action_plan || []).length + ' étapes définies');
    }

    // ── Orchestrateur / Experts ──
    setProgressStep('analyse');
    addThought('🔍 Orchestrateur — identification des experts…');
    addThought('✅ Stratégie : ' + (data.strategy || ''));

    setProgressStep('debat');
    // Animer les cards pendant le traitement
    [0,1,2].forEach(i => setTimeout(() => {
      const card = document.getElementById('expert-' + i);
      if (card && card.className.includes('waiting')) {
        card.className = 'expert-card thinking';
        card.querySelector('.expert-status').className  = 'expert-status thinking';
        card.querySelector('.expert-status').textContent = '⚡ En réflexion…';
      }
    }, i * 300));
    addThought('⚡ 3 agents experts en cours de délibération en parallèle…');

    data.experts.forEach((exp, i) => {
      addThought(`${exp.emoji} ${exp.role} : ${exp.specialty}`);
      setExpertState(i, 'done', exp);
    });

    setProgressStep('synthese');
    addThought('🏆 Synthèse critique en cours…', true);

    // ── Propositions ──
    const tabsEl     = document.getElementById('proposal-tabs');
    const contentsEl = document.getElementById('proposal-contents');
    tabsEl.innerHTML     = '';
    contentsEl.innerHTML = '';

    data.proposals.forEach((p, i) => {
      const tab = document.createElement('button');
      tab.className   = 'ptab' + (i === 0 ? ' active' : '');
      tab.textContent = `${p.expert.emoji} ${p.expert.role}`;
      tab.onclick = () => {
        document.querySelectorAll('.ptab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.proposal-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById('pc-' + i).classList.add('active');
      };
      tabsEl.appendChild(tab);

      const content = document.createElement('div');
      content.className = 'proposal-content' + (i === 0 ? ' active' : '');
      content.id        = 'pc-' + i;
      content.innerHTML = marked.parse(p.text);
      content.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
      contentsEl.appendChild(content);
    });

    document.getElementById('proposals-section').style.display = 'block';

    // ── Synthèse ──
    const synthEl = document.getElementById('synthesis-content');
    synthEl.innerHTML = marked.parse(data.synthesis);
    synthEl.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
    document.getElementById('synthesis-section').style.display = 'block';

    // ── Programmeur ──
    setProgressStep('code');
    addThought('💻 Agent Programmeur — génération du code…', true);

    if (data.programmer_output) {
      const progEl = document.getElementById('programmer-content');
      progEl.innerHTML = marked.parse(data.programmer_output);
      progEl.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
      document.getElementById('programmer-section').style.display = 'block';
      addThought('✅ Code généré avec succès !', true);
    }

    ['ps-pm','ps-analyse','ps-debat','ps-synthese','ps-code'].forEach(id => {
      document.getElementById(id).className = 'progress-step done';
    });
    addThought('🎉 War Room terminée — Plan d\'Action Final + Code prêts !', true);
    showToast('War Room terminée !', 'success');

  } catch(e) {
    showToast('Erreur réseau : ' + e.message, 'error');
    addThought('❌ Erreur : ' + e.message);
  }

  btn.disabled = false; btn.textContent = '🚀 Relancer';
}

function sendSynthesisToChat() {
  if (!warRoomData) return;
  const text = '**⚔️ War Room — Plan d\'Action Final**\n\n' + warRoomData.synthesis;
  closeWarRoom();
  document.getElementById('welcome')?.remove();
  appendMessage('bot', text);
  showToast('Synthèse ajoutée au chat', 'success');
}

function sendProgrammerToChat() {
  if (!warRoomData?.programmer_output) return;
  const text = '**💻 War Room — Code Généré par l\'Agent Programmeur**\n\n' + warRoomData.programmer_output;
  closeWarRoom();
  document.getElementById('welcome')?.remove();
  appendMessage('bot', text);
  showToast('Code ajouté au chat', 'success');
}

function copyProgrammerCode() {
  if (!warRoomData?.programmer_output) return;
  navigator.clipboard.writeText(warRoomData.programmer_output)
    .then(() => showToast('Code copié !', 'success'))
    .catch(() => showToast('Erreur lors de la copie', 'error'));
}

/* ===================================================
   😈 AVOCAT DU DIABLE
   =================================================== */
function openAdvocate() {
  document.getElementById('advocate-overlay').classList.add('active');
  const msg = document.getElementById('message-input').value.trim();
  if (msg) document.getElementById('advocate-input').value = msg;
  setTimeout(() => document.getElementById('advocate-input').focus(), 100);
}
function closeAdvocate() {
  document.getElementById('advocate-overlay').classList.remove('active');
}

async function launchAdvocate() {
  const topic = document.getElementById('advocate-input').value.trim();
  if (!topic) return showToast('Entre un sujet à critiquer', 'error');

  const model = document.getElementById('model-select').value;
  const btn   = document.getElementById('adv-btn');
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span>';

  document.getElementById('advocate-placeholder').style.display = 'none';
  const result = document.getElementById('advocate-result');
  result.style.display = 'block';
  result.innerHTML = '<div style="text-align:center;padding:30px;color:var(--muted);"><span class="spinner"></span><br><br>L\'Avocat du Diable réfléchit…</div>';

  try {
    const res  = await fetch('/api/advocate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic, model })
    });
    const data = await res.json();
    if (data.error) {
      result.innerHTML = `<p style="color:var(--danger)">${escHtml(data.error)}</p>`;
    } else {
      result.innerHTML = marked.parse(data.critique);
      result.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
    }
  } catch(e) {
    result.innerHTML = `<p style="color:var(--danger)">Erreur réseau: ${escHtml(e.message)}</p>`;
  }

  btn.disabled = false; btn.textContent = '😈 Critiquer';
}



/* ===================================================
   UTILS
   =================================================== */
function prefillCurrentMessage() {
  const msg = document.getElementById('message-input').value.trim();
  if (!msg) return showToast('Aucun message à utiliser', 'warn');
  document.getElementById('warroom-input').value  = msg;
  document.getElementById('advocate-input').value = msg;
  showToast('Message copié dans les outils', 'success');
}

function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/'/g,"&#39;").replace(/"/g,'&quot;');
}
function showToast(msg, type = '') {
  const t = document.getElementById('toast');
  t.textContent = msg; t.className = 'show ' + type;
  clearTimeout(t._tid);
  t._tid = setTimeout(() => t.className = '', 3000);
}

// Fermer les overlays avec Escape
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    closeSkillModal();
    closeWarRoom();
    closeAdvocate();
  }
});

// Fermer en cliquant en dehors du modal
['warroom-overlay','advocate-overlay'].forEach(id => {
  document.getElementById(id).addEventListener('click', function(e) {
    if (e.target === this) {
      closeWarRoom(); closeAdvocate();
    }
  });
});

/* ===================================================
   INIT
   =================================================== */
(async () => {
  await loadUser();
  try {
    const res  = await fetch('/api/history');
    const data = await res.json();
    if (data.messages?.length) {
      document.getElementById('welcome')?.remove();
      data.messages.forEach(m => appendMessage(m.role, m.content));
      updateStats(data.messages.length, data.estimated_tokens);
    }
    if (data.system_prompt) {
      document.getElementById('system-prompt').value = data.system_prompt;
      hasContext = true;
      document.getElementById('context-indicator').style.display = 'flex';
      document.getElementById('stat-ctx').textContent = 'Oui';
    }
    if (data.skills?.length) { skills = data.skills; renderSkills(); }
    if (data.model) {
      document.getElementById('model-select').value      = data.model;
      document.getElementById('model-badge').textContent = data.model;
    }
    if (data.lang) {
      currentLang = data.lang;
      const sel = document.getElementById('lang-select');
      if (sel) sel.value = data.lang;
    }
  } catch (e) { console.error('Init error:', e); }
})();
