/* ===== Shark Answer — Frontend Logic ===== */

// ── State ──
let selectedFiles = [];
let currentResult = null;
let currentSubmissionId = null;
let _lastSubmissionId = null;
let _chatSidebarOpen = false;

// ── DOM refs ──
const dropZone         = document.getElementById('dropZone');
const fileInput        = document.getElementById('fileInput');
const filePreviews     = document.getElementById('filePreviews');
const submitBtn        = document.getElementById('submitBtn');
const uploadSection    = document.getElementById('uploadSection');
const progressSection  = document.getElementById('progressSection');
const resultsSection   = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');
const exportBtns       = document.getElementById('exportBtns');
const historyList      = document.getElementById('historyList');
const sidebar          = document.getElementById('sidebar');
const sidebarToggle    = document.getElementById('sidebarToggle');
const newSubmission    = document.getElementById('newSubmission');
const profileSelect    = document.getElementById('profileSelect');
const mainArea         = document.getElementById('mainArea');
const chatSidebar      = document.getElementById('chatSidebar');
const chatToggleBtn    = document.getElementById('chatToggleBtn');

// ── Accepted file types ──
const ACCEPTED_EXTENSIONS = new Set([
  '.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif',
  '.gif', '.bmp', '.tiff', '.tif', '.pdf', '.docx', '.doc',
]);
const ACCEPTED_MIME_TYPES = new Set([
  'image/jpeg', 'image/png', 'image/webp', 'image/gif',
  'image/heic', 'image/heif', 'image/bmp', 'image/tiff',
  'application/pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'application/msword',
]);

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
  loadHistory();
  loadProfiles();
});

// ── Left sidebar toggle (mobile) ──
sidebarToggle.addEventListener('click', () => {
  sidebar.classList.toggle('max-lg:-translate-x-full');
});

// ── New submission ──
newSubmission.addEventListener('click', showUploadSection);

// ── File handling ──
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  addFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', () => { addFiles(fileInput.files); fileInput.value = ''; });

function isAcceptedFile(f) {
  if (ACCEPTED_MIME_TYPES.has(f.type)) return true;
  const ext = '.' + f.name.split('.').pop().toLowerCase();
  return ACCEPTED_EXTENSIONS.has(ext);
}
function getFileTypeLabel(f) {
  const ext = f.name.split('.').pop().toLowerCase();
  if (ext === 'pdf') return 'pdf';
  if (['docx', 'doc'].includes(ext)) return 'docx';
  if (['heic', 'heif'].includes(ext)) return 'heic';
  return 'image';
}
function addFiles(fileList) {
  let rejected = 0;
  for (const f of fileList) {
    if (isAcceptedFile(f)) selectedFiles.push(f);
    else rejected++;
  }
  if (rejected > 0) showToast(`${rejected} file(s) skipped — use JPG, PNG, WEBP, HEIC, PDF, or DOCX.`, 'warn');
  renderPreviews();
  updateSubmitButton();
}
function removeFile(index) { selectedFiles.splice(index, 1); renderPreviews(); updateSubmitButton(); }
function renderPreviews() {
  if (selectedFiles.length === 0) { filePreviews.classList.add('hidden'); filePreviews.innerHTML = ''; return; }
  filePreviews.classList.remove('hidden');
  filePreviews.innerHTML = '';
  selectedFiles.forEach((f, i) => {
    const div = document.createElement('div');
    div.className = 'file-thumb';
    const type = getFileTypeLabel(f);
    if (type === 'image') {
      const img = document.createElement('img');
      img.src = URL.createObjectURL(f);
      img.alt = f.name;
      div.appendChild(img);
    } else {
      const icon = document.createElement('div');
      icon.className = 'file-thumb-icon';
      if (type === 'pdf') {
        icon.innerHTML = `<svg class="w-8 h-8 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"/></svg><span class="text-xs text-red-400 font-bold mt-1">PDF</span>`;
      } else {
        icon.innerHTML = `<svg class="w-8 h-8 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg><span class="text-xs text-blue-400 font-bold mt-1">DOCX</span>`;
      }
      const fname = document.createElement('span');
      fname.className = 'file-thumb-name';
      fname.textContent = f.name.length > 14 ? f.name.slice(0, 11) + '…' : f.name;
      div.appendChild(icon);
      div.appendChild(fname);
    }
    const btn = document.createElement('div');
    btn.className = 'remove-btn';
    btn.innerHTML = '✕';
    btn.onclick = () => removeFile(i);
    div.appendChild(btn);
    filePreviews.appendChild(div);
  });
}
function updateSubmitButton() { submitBtn.disabled = selectedFiles.length === 0; }

// ── Submit (SSE streaming) ──
submitBtn.addEventListener('click', submitPaper);
async function submitPaper() {
  if (selectedFiles.length === 0) return;
  const fd = new FormData();
  selectedFiles.forEach(f => fd.append('images', f));
  fd.append('subject', document.getElementById('subjectSelect').value);
  fd.append('language', document.getElementById('langSelect').value);
  fd.append('max_versions', document.getElementById('versionsSelect').value);
  fd.append('examiner_profile', profileSelect.value);

  showProgressSection();
  setStage('upload');

  try {
    const response = await fetch('/api/solve/stream', { method: 'POST', body: fd });
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finalData = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Parse complete SSE events (separated by \n\n)
      const parts = buffer.split('\n\n');
      buffer = parts.pop() ?? '';   // keep incomplete trailing chunk

      for (const eventStr of parts) {
        if (!eventStr.trim()) continue;
        let eventName = 'message';
        let dataStr = '';
        for (const line of eventStr.split('\n')) {
          if (line.startsWith('event: ')) eventName = line.slice(7).trim();
          else if (line.startsWith('data: ')) dataStr = line.slice(6).trim();
        }
        if (!dataStr) continue;
        let evData;
        try { evData = JSON.parse(dataStr); } catch { continue; }

        if (eventName === 'error') throw new Error(evData.message || 'Server error');
        if (eventName === 'result') { finalData = evData; continue; }
        handleSSEEvent(eventName, evData);
      }
    }

    if (finalData) {
      currentResult = finalData;
      currentSubmissionId = finalData.submission_id || null;
      completeAllStages();
      await sleep(400);
      showResultsSection(finalData);
      loadHistory();
    }
  } catch (err) {
    showError(err.message);
  }
}

function handleSSEEvent(eventName, data) {
  switch (eventName) {
    case 'stage':
      if (data.done) {
        markStageDone(data.id, data.label);
      } else {
        setStage(data.id, data.label);
      }
      break;
    case 'progress':
      document.getElementById('progressSubtext').textContent = data.label || '';
      break;
    case 'scoring':
      document.getElementById('progressSubtext').textContent = data.label || 'Scoring…';
      break;
  }
}

// ── Progress stages ──
function setStage(stageName, labelOverride) {
  const stages = document.querySelectorAll('#pipelineStages .stage');
  let passed = false;
  stages.forEach(s => {
    const name = s.dataset.stage;
    if (name === stageName) {
      s.classList.add('active'); s.classList.remove('done'); passed = true;
      const spanEl = s.querySelector('span');
      const label = labelOverride || (spanEl ? spanEl.textContent : stageName);
      document.getElementById('progressSubtext').textContent = label;
      if (labelOverride && spanEl) spanEl.textContent = labelOverride;
    } else if (!passed) {
      s.classList.remove('active'); s.classList.add('done');
    } else {
      s.classList.remove('active', 'done');
    }
  });
}
function markStageDone(stageName, labelOverride) {
  const stages = document.querySelectorAll('#pipelineStages .stage');
  stages.forEach(s => {
    if (s.dataset.stage === stageName) {
      s.classList.remove('active'); s.classList.add('done');
      if (labelOverride) { const sp = s.querySelector('span'); if (sp) sp.textContent = labelOverride; }
    }
  });
}
function completeAllStages() {
  document.querySelectorAll('#pipelineStages .stage').forEach(s => {
    s.classList.remove('active'); s.classList.add('done');
  });
  document.getElementById('progressSubtext').textContent = 'Complete!';
}

// ── Paper version style labels ──
const PAPER_STYLE_LABELS = [
  { short: 'V1 — Formal',   long: 'Formal Academic — precise, examiner-approved phrasing' },
  { short: 'V2 — Concise',  long: 'Concise — same key points, minimal words' },
  { short: 'V3 — Natural',  long: 'Natural Voice — how a top student would write it' },
  { short: 'V4 — Alt',      long: 'Alternative Method / Approach' },
  { short: 'V5 — Extended', long: 'Extended Working — every step shown explicitly' },
];

// ── Results rendering (paper-based) ──
function showResultsSection(data) {
  uploadSection.classList.add('hidden');
  progressSection.classList.add('hidden');
  resultsSection.classList.remove('hidden');
  exportBtns.classList.remove('hidden');
  exportBtns.classList.add('flex');
  chatToggleBtn.classList.remove('hidden');
  chatToggleBtn.classList.add('flex');

  resultsContainer.innerHTML = '';

  // Cost bar
  const cost = data.cost_summary || {};
  const costBar = el('div', 'cost-bar');
  costBar.innerHTML = `
    <div class="cost-item">Questions: <span>${data.total_questions}</span></div>
    <div class="cost-item">Total Cost: <span>$${(cost.total_cost_usd || 0).toFixed(4)}</span></div>
    <div class="cost-item">Input Tokens: <span>${(cost.total_input_tokens || 0).toLocaleString()}</span></div>
    <div class="cost-item">Output Tokens: <span>${(cost.total_output_tokens || 0).toLocaleString()}</span></div>
  `;
  resultsContainer.appendChild(costBar);

  if (!data.results || data.results.length === 0) {
    const noData = el('div', 'text-center py-16 text-zinc-500');
    noData.textContent = 'No questions were processed.';
    resultsContainer.appendChild(noData);
    return;
  }

  // Find max versions across all questions
  const maxVersions = Math.max(
    ...data.results.map(qr => (qr.versions ? qr.versions.length : 0)),
    1
  );

  // Paper-level tabs (Paper V1, V2, V3…)
  const paperTabRow = el('div', 'paper-tab-row');
  const paperPanels = [];

  for (let v = 1; v <= maxVersions; v++) {
    const styleInfo = PAPER_STYLE_LABELS[v - 1] || { short: `V${v}`, long: `Version ${v}` };
    const tab = el('button', 'paper-tab' + (v === 1 ? ' active' : ''));
    tab.textContent = styleInfo.short;
    tab.title = styleInfo.long;
    const vIdx = v - 1;
    tab.onclick = () => {
      paperTabRow.querySelectorAll('.paper-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      paperPanels.forEach((p, pi) => { p.style.display = pi === vIdx ? 'block' : 'none'; });
    };
    paperTabRow.appendChild(tab);
  }
  resultsContainer.appendChild(paperTabRow);

  // Create one panel per paper version
  for (let v = 1; v <= maxVersions; v++) {
    const styleInfo = PAPER_STYLE_LABELS[v - 1] || { short: `V${v}`, long: `Version ${v}` };
    const panel = el('div', 'paper-panel');
    if (v !== 1) panel.style.display = 'none';

    // Style badge for this paper
    const styleBadge = el('div', 'flex items-center gap-2 mb-5');
    styleBadge.innerHTML = `
      <span class="badge-model">${esc(styleInfo.short)}</span>
      <span class="text-xs text-zinc-500">${esc(styleInfo.long)}</span>
    `;
    panel.appendChild(styleBadge);

    // Each question for this version
    data.results.forEach(qr => {
      // Use Nth version, fallback to last available
      const version = qr.versions && qr.versions.length > 0
        ? (qr.versions[v - 1] || qr.versions[qr.versions.length - 1])
        : null;

      const block = el('div', 'question-block');

      // Question header — no model name shown (moved to tech details)
      const header = el('div', 'flex items-start justify-between mb-3 gap-4');
      const providerStr = version ? esc(version.provider) : '';
      const verifiedBadge = version && version.verified
        ? '<span class="badge-verified ml-1">✓ Verified</span>' : '';
      const qs = version && version.quality_score != null ? version.quality_score : null;
      const scoreBadge = qs
        ? `<span class="badge-score${(() => { const p = qs.split('/'); return p.length === 2 && parseInt(p[0]) === parseInt(p[1]); })() ? ' badge-score-full' : ''} ml-1">🎯 ${esc(qs)}</span>`
        : '';
      header.innerHTML = `
        <h3 class="text-base font-semibold shrink-0">Q${esc(qr.question_number)}</h3>
        <div class="flex flex-wrap items-center gap-1.5 justify-end">
          ${verifiedBadge}${scoreBadge}
        </div>
      `;
      block.appendChild(header);

      // Question text (subtle, collapsed style)
      const qText = el('div', 'text-xs text-zinc-500 mb-4 leading-relaxed border-l-2 border-zinc-700/60 pl-3 line-clamp-3');
      qText.textContent = qr.question_text;
      block.appendChild(qText);

      // Errors
      if (qr.errors && qr.errors.length > 0) {
        const errDiv = el('div', 'bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-sm text-red-400 mb-4');
        errDiv.textContent = qr.errors.join('; ');
        block.appendChild(errDiv);
      }

      if (!version) {
        const noVer = el('div', 'text-xs text-zinc-600 italic');
        noVer.textContent = 'No answer for this version';
        block.appendChild(noVer);
        panel.appendChild(block);
        return;
      }

      // Answer text
      const ansDiv = el('div', 'answer-text mt-1');
      ansDiv.innerHTML = formatAnswer(version.answer_text);
      block.appendChild(ansDiv);

      // Explanation toggle
      if (version.explanation_text) {
        const toggleBtn = el('button', 'mt-4 text-xs text-shark-400 hover:text-shark-300 flex items-center gap-1 transition');
        toggleBtn.innerHTML = `
          <svg class="w-3 h-3 transition-transform expl-arrow" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
          </svg>
          Show Explanation
        `;
        const explPanel = el('div', 'explanation-panel mt-2');
        const explContent = el('div', 'answer-text text-sm');
        explContent.innerHTML = formatAnswer(version.explanation_text);
        explPanel.appendChild(explContent);
        toggleBtn.onclick = () => {
          const open = explPanel.classList.toggle('open');
          toggleBtn.querySelector('.expl-arrow').style.transform = open ? 'rotate(180deg)' : '';
          toggleBtn.childNodes[toggleBtn.childNodes.length - 1].textContent =
            open ? ' Hide Explanation' : ' Show Explanation';
        };
        block.appendChild(toggleBtn);
        block.appendChild(explPanel);
      }

      // Technical details collapsible (hidden by default — no AI branding in main view)
      const techBtn = el('button', 'mt-3 text-xs text-zinc-600 hover:text-zinc-400 flex items-center gap-1 transition');
      techBtn.innerHTML = 'ℹ Technical Details';
      const techPanel = el('div', 'hidden mt-1 p-3 rounded-lg border border-zinc-800/60 text-xs text-zinc-600 space-y-0.5');
      techPanel.innerHTML = [
        `<div>Pipeline: <span class="text-zinc-500">${esc(qr.pipeline)}</span></div>`,
        providerStr ? `<div>Model: <span class="text-zinc-500">${providerStr}</span></div>` : '',
        version.approach_label ? `<div>Style: <span class="text-zinc-500">${esc(version.approach_label)}</span></div>` : '',
        qs ? `<div>Score: <span class="text-zinc-500">${esc(qs)}</span></div>` : '',
      ].filter(Boolean).join('');
      techBtn.onclick = () => techPanel.classList.toggle('hidden');
      block.appendChild(techBtn);
      block.appendChild(techPanel);

      panel.appendChild(block);
    });

    paperPanels.push(panel);
    resultsContainer.appendChild(panel);
  }

  // Render math
  renderMath();

  // Update cost display
  updateCostDisplay(cost.total_cost_usd || 0);

  // Populate chat sidebar question selector
  setupChatPanel(data);
}

// ── Math rendering ──
function renderMath() {
  if (typeof renderMathInElement !== 'undefined') {
    renderMathInElement(resultsContainer, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false },
        { left: '\\(', right: '\\)', display: false },
        { left: '\\[', right: '\\]', display: true },
      ],
      throwOnError: false,
    });
  } else {
    setTimeout(renderMath, 500);
  }
}

// ── Format answer text (markdown-lite) ──
function formatAnswer(text) {
  if (!text) return '';
  let html = esc(text);
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/^### (.+)$/gm, '<div class="text-base font-semibold text-zinc-200 mt-4 mb-1">$1</div>');
  html = html.replace(/^## (.+)$/gm, '<div class="text-lg font-semibold text-zinc-100 mt-5 mb-2">$1</div>');
  html = html.replace(/^[-•] (.+)$/gm, '<div class="ml-4 flex gap-2"><span class="text-shark-400">•</span><span>$1</span></div>');
  html = html.replace(/^(\d+)\. (.+)$/gm, '<div class="ml-4 flex gap-2"><span class="text-shark-400 font-mono text-xs mt-0.5">$1.</span><span>$2</span></div>');
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="bg-zinc-800/50 rounded-lg p-3 mt-2 mb-2 overflow-x-auto text-xs font-mono text-green-300">$2</pre>');
  html = html.replace(/`([^`]+)`/g, '<code class="bg-zinc-800 px-1.5 py-0.5 rounded text-xs text-orange-300">$1</code>');
  html = html.replace(/\n/g, '<br>');
  return html;
}

// ── Right chat sidebar ──
function toggleChatSidebar() {
  _chatSidebarOpen = !_chatSidebarOpen;
  if (_chatSidebarOpen) {
    chatSidebar.classList.remove('translate-x-full');
    mainArea.style.paddingRight = '384px'; // 96 * 4 = 384px (w-96)
    chatToggleBtn.classList.add('active-chat-btn');
  } else {
    chatSidebar.classList.add('translate-x-full');
    mainArea.style.paddingRight = '';
    chatToggleBtn.classList.remove('active-chat-btn');
  }
}

function setupChatPanel(data) {
  const select = document.getElementById('chatQuestionSelect');
  if (!data.results || data.results.length === 0) return;

  select.innerHTML = '';
  data.results.forEach(qr => {
    const opt = document.createElement('option');
    opt.value = qr.question_number;
    opt.textContent = `Q${qr.question_number} — ${qr.question_text.slice(0, 40)}${qr.question_text.length > 40 ? '…' : ''}`;
    select.appendChild(opt);
  });

  // Reset chat messages to empty state
  const msgs = document.getElementById('chatMessages');
  msgs.innerHTML = `<div id="chatEmptyState" class="text-center py-10"><p class="text-xs text-zinc-600 leading-relaxed">Select a question and ask Claude<br>to review or correct the answer</p></div>`;
}

async function sendChatMessage() {
  const sid = currentSubmissionId || _lastSubmissionId;
  if (!sid) { showToast('No submission loaded', 'warn'); return; }

  const select = document.getElementById('chatQuestionSelect');
  const input = document.getElementById('chatInput');
  const sendBtn = document.getElementById('chatSendBtn');
  const messages = document.getElementById('chatMessages');

  const message = input.value.trim();
  if (!message) return;
  const questionNumber = select.value;

  sendBtn.disabled = true;
  sendBtn.textContent = '...';
  input.disabled = true;

  // Remove empty state if present
  const emptyState = document.getElementById('chatEmptyState');
  if (emptyState) emptyState.remove();

  appendChatMessage(messages, 'user', message);
  input.value = '';

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ submission_id: sid, question_number: questionNumber, message }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    appendChatMessage(messages, 'assistant', data.reply);
  } catch (e) {
    appendChatMessage(messages, 'error', 'Error: ' + e.message);
  } finally {
    sendBtn.disabled = false;
    sendBtn.innerHTML = `<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/></svg> Send`;
    input.disabled = false;
    input.focus();
  }
}

// Enter to send (Shift+Enter = newline)
document.addEventListener('DOMContentLoaded', () => {
  const chatInput = document.getElementById('chatInput');
  if (chatInput) {
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChatMessage(); }
    });
  }
});

function appendChatMessage(container, role, content) {
  const wrap = document.createElement('div');
  if (role === 'user') {
    wrap.className = 'flex justify-end';
    const bubble = document.createElement('div');
    bubble.className = 'max-w-xs bg-shark-600/25 border border-shark-500/25 rounded-2xl rounded-tr-sm px-3 py-2 text-sm text-zinc-200';
    bubble.textContent = content;
    wrap.appendChild(bubble);
  } else if (role === 'assistant') {
    wrap.className = 'flex gap-2.5';
    const avatar = document.createElement('div');
    avatar.className = 'w-6 h-6 rounded-full bg-shark-600 flex items-center justify-center text-white text-xs font-bold shrink-0 mt-0.5';
    avatar.textContent = 'C';
    const bubble = document.createElement('div');
    bubble.className = 'flex-1 bg-zinc-800/60 border border-zinc-700/50 rounded-2xl rounded-tl-sm px-3 py-2 text-sm text-zinc-200 answer-text min-w-0';
    bubble.innerHTML = formatAnswer(content);
    wrap.appendChild(avatar);
    wrap.appendChild(bubble);
  } else {
    wrap.className = 'text-xs text-red-400 px-2';
    wrap.textContent = content;
  }
  container.appendChild(wrap);
  container.scrollTop = container.scrollHeight;

  if (typeof renderMathInElement !== 'undefined') {
    try {
      renderMathInElement(wrap, {
        delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }],
        throwOnError: false,
      });
    } catch (e) { /* ignore */ }
  }
}

// ── Export functions ──

// PDF export now opens a modal to choose which paper versions to download
function exportPDF() {
  if (!currentResult) return;
  const modal = document.getElementById('exportModal');
  const checksDiv = document.getElementById('exportVersionChecks');
  const maxVersions = Math.max(...(currentResult.results || []).map(qr => (qr.versions || []).length), 1);
  const styleLabels = ['Formal Academic', 'Concise', 'Natural Voice', 'Alternative Method', 'Extended Working'];
  checksDiv.innerHTML = '';
  for (let v = 1; v <= maxVersions; v++) {
    const label = styleLabels[v - 1] || `Version ${v}`;
    const row = el('label', 'flex items-center gap-2.5 cursor-pointer py-1');
    row.innerHTML = `
      <input type="checkbox" value="${v}" class="export-ver-check w-4 h-4 rounded accent-shark-500" checked>
      <span class="text-sm">Paper ${v} — <span class="text-zinc-400">${label}</span></span>
    `;
    checksDiv.appendChild(row);
  }
  modal.classList.remove('hidden');
}

async function doExportPDF() {
  const checks = Array.from(document.querySelectorAll('.export-ver-check:checked'));
  const versions = checks.map(c => parseInt(c.value));
  if (versions.length === 0) { showToast('Select at least one paper version', 'warn'); return; }
  closeExportModal();
  try {
    if (versions.length === 1) {
      const resp = await fetch('/api/export/pdf/paper', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: currentResult, version_number: versions[0] }),
      });
      if (!resp.ok) throw new Error('Export failed');
      downloadBlob(await resp.blob(), `paper_${versions[0]}.pdf`);
    } else {
      const resp = await fetch('/api/export/pdf/zip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: currentResult, version_numbers: versions }),
      });
      if (!resp.ok) throw new Error('Export failed');
      downloadBlob(await resp.blob(), 'shark_papers.zip');
    }
  } catch (e) { showToast('PDF export failed: ' + e.message, 'error'); }
}

function closeExportModal() {
  document.getElementById('exportModal').classList.add('hidden');
}
async function exportDocx() {
  if (!currentResult) return;
  try {
    const resp = await fetch('/api/export/docx', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ data: currentResult }) });
    if (!resp.ok) throw new Error('Export failed');
    downloadBlob(await resp.blob(), 'shark_answer_results.docx');
  } catch (e) { showToast('Word export failed: ' + e.message, 'error'); }
}
async function exportMarkdown() {
  if (!currentResult) return;
  try {
    const resp = await fetch('/api/export/md', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ data: currentResult }) });
    if (!resp.ok) throw new Error('Export failed');
    downloadBlob(await resp.blob(), 'shark_answer_results.md');
  } catch (e) { showToast('Markdown export failed: ' + e.message, 'error'); }
}
async function exportTxt() {
  if (!currentResult) return;
  try {
    const resp = await fetch('/api/export/txt', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ data: currentResult }) });
    if (!resp.ok) throw new Error('Export failed');
    downloadBlob(await resp.blob(), 'shark_answer_results.txt');
  } catch (e) { showToast('Text export failed: ' + e.message, 'error'); }
}
async function exportXlsx() {
  if (!currentResult) return;
  try {
    const resp = await fetch('/api/export/xlsx', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ data: currentResult }) });
    if (!resp.ok) throw new Error('Export failed');
    downloadBlob(await resp.blob(), 'shark_answer_results.xlsx');
  } catch (e) { showToast('Excel export failed: ' + e.message, 'error'); }
}
function openPreview() {
  if (!currentResult) return;
  const lastEntry = document.querySelector('#historyList .history-item');
  if (lastEntry) window.open(`/preview/${lastEntry.dataset.id}`, '_blank');
}
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

// ── History ──
async function loadHistory() {
  try {
    const resp = await fetch('/api/history');
    if (!resp.ok) return;
    const items = await resp.json();
    renderHistory(items);
    if (items.length > 0) {
      _lastSubmissionId = items[0].id;
      if (!currentSubmissionId) currentSubmissionId = _lastSubmissionId;
    }
  } catch (e) { /* silent */ }
}
function renderHistory(items) {
  if (!items || items.length === 0) {
    historyList.innerHTML = '<p class="px-2 py-8 text-xs text-zinc-500 text-center">No submissions yet</p>';
    return;
  }
  historyList.innerHTML = '';
  items.forEach(h => {
    const div = el('div', 'history-item');
    div.dataset.id = h.id;
    const nameStr = h.filenames ? h.filenames[0] : 'Paper';
    div.innerHTML = `
      <div class="text-sm font-medium truncate">${esc(nameStr)}</div>
      <div class="text-xs text-zinc-500 mt-0.5">${esc(h.subject.replace('_', ' '))} · ${h.total_questions}Q · ${esc(h.timestamp)}</div>
    `;
    div.onclick = () => loadHistoryEntry(h.id);
    historyList.appendChild(div);
  });
}
async function loadHistoryEntry(sid) {
  try {
    const resp = await fetch(`/api/history/${sid}`);
    if (!resp.ok) return;
    const data = await resp.json();
    currentResult = data;
    currentSubmissionId = sid;
    showResultsSection(data);
    historyList.querySelectorAll('.history-item').forEach(item => {
      item.classList.toggle('active', item.dataset.id === sid);
    });
  } catch (e) { /* silent */ }
}

// ── Examiner profiles ──
async function loadProfiles() {
  try {
    const resp = await fetch('/api/examiner/profiles');
    if (!resp.ok) return;
    const profiles = await resp.json();
    profiles.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.name;
      opt.textContent = `${p.name} (${p.region})`;
      profileSelect.appendChild(opt);
    });
  } catch (e) { /* silent */ }
}

// ── UI state helpers ──
function showUploadSection() {
  selectedFiles = [];
  currentResult = null;
  renderPreviews();
  updateSubmitButton();
  uploadSection.classList.remove('hidden');
  progressSection.classList.add('hidden');
  resultsSection.classList.add('hidden');
  exportBtns.classList.add('hidden');
  exportBtns.classList.remove('flex');
  chatToggleBtn.classList.add('hidden');
  chatToggleBtn.classList.remove('flex');
  // Close chat sidebar
  if (_chatSidebarOpen) toggleChatSidebar();
  document.querySelectorAll('#pipelineStages .stage').forEach(s => s.classList.remove('active', 'done'));
}
function showProgressSection() {
  uploadSection.classList.add('hidden');
  progressSection.classList.remove('hidden');
  resultsSection.classList.add('hidden');
  exportBtns.classList.add('hidden');
  exportBtns.classList.remove('flex');
}
function showError(message) {
  progressSection.classList.add('hidden');
  uploadSection.classList.remove('hidden');
  showToast(message, 'error');
}
function showToast(message, type = 'error') {
  const colors = { error: 'bg-red-600/90 text-white', warn: 'bg-amber-600/90 text-white', info: 'bg-zinc-700/90 text-zinc-100' };
  const toast = el('div', `fixed bottom-6 right-6 ${colors[type] || colors.info} px-5 py-3 rounded-xl shadow-2xl text-sm z-50 max-w-md`);
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 6000);
}
function updateCostDisplay(cost) {
  document.getElementById('totalCostDisplay').textContent = `Session: $${cost.toFixed(4)}`;
}

// ── Utilities ──
function el(tag, className) { const e = document.createElement(tag); if (className) e.className = className; return e; }
function esc(str) { if (!str) return ''; const div = document.createElement('div'); div.textContent = str; return div.innerHTML; }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
