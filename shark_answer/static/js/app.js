/* ===== Shark Answer — Frontend Logic ===== */

// State
let selectedFiles = [];
let currentResult = null;
let currentSubmissionId = null;

// DOM refs
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const filePreviews = document.getElementById('filePreviews');
const submitBtn = document.getElementById('submitBtn');
const uploadSection = document.getElementById('uploadSection');
const progressSection = document.getElementById('progressSection');
const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');
const exportBtns = document.getElementById('exportBtns');
const historyList = document.getElementById('historyList');
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
const newSubmission = document.getElementById('newSubmission');
const profileSelect = document.getElementById('profileSelect');

// ===== Init =====
document.addEventListener('DOMContentLoaded', () => {
  loadHistory();
  loadProfiles();
});

// ===== Sidebar toggle (mobile) =====
sidebarToggle.addEventListener('click', () => {
  sidebar.classList.toggle('max-lg:-translate-x-full');
});

// ===== New submission =====
newSubmission.addEventListener('click', () => {
  showUploadSection();
});

// ===== File handling =====
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  addFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', () => {
  addFiles(fileInput.files);
  fileInput.value = '';
});

function addFiles(fileList) {
  for (const f of fileList) {
    if (f.type.startsWith('image/')) {
      selectedFiles.push(f);
    }
  }
  renderPreviews();
  updateSubmitButton();
}

function removeFile(index) {
  selectedFiles.splice(index, 1);
  renderPreviews();
  updateSubmitButton();
}

function renderPreviews() {
  if (selectedFiles.length === 0) {
    filePreviews.classList.add('hidden');
    filePreviews.innerHTML = '';
    return;
  }
  filePreviews.classList.remove('hidden');
  filePreviews.innerHTML = '';
  selectedFiles.forEach((f, i) => {
    const div = document.createElement('div');
    div.className = 'file-thumb';
    const img = document.createElement('img');
    img.src = URL.createObjectURL(f);
    img.alt = f.name;
    const btn = document.createElement('div');
    btn.className = 'remove-btn';
    btn.innerHTML = '✕';
    btn.onclick = () => removeFile(i);
    div.appendChild(img);
    div.appendChild(btn);
    filePreviews.appendChild(div);
  });
}

function updateSubmitButton() {
  submitBtn.disabled = selectedFiles.length === 0;
}

// ===== Submit =====
submitBtn.addEventListener('click', submitPaper);

async function submitPaper() {
  if (selectedFiles.length === 0) return;

  const subject = document.getElementById('subjectSelect').value;
  const language = document.getElementById('langSelect').value;
  const maxVersions = document.getElementById('versionsSelect').value;
  const profile = profileSelect.value;

  // Build FormData
  const fd = new FormData();
  selectedFiles.forEach(f => fd.append('images', f));
  fd.append('subject', subject);
  fd.append('language', language);
  fd.append('max_versions', maxVersions);
  fd.append('examiner_profile', profile);

  // Show progress
  showProgressSection();
  setStage('upload');

  try {
    // Simulate upload stage
    await sleep(300);
    setStage('extract');

    const response = await fetch('/api/solve', { method: 'POST', body: fd });

    setStage('solve');
    await sleep(200);
    setStage('verify');
    await sleep(200);
    setStage('finalize');

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();
    currentResult = data;
    await sleep(300);
    completeAllStages();
    await sleep(500);

    showResultsSection(data);
    loadHistory();  // refresh sidebar

  } catch (err) {
    showError(err.message);
  }
}

// ===== Progress stages =====
function setStage(stageName) {
  const stages = document.querySelectorAll('#pipelineStages .stage');
  let passed = false;
  stages.forEach(s => {
    const name = s.dataset.stage;
    if (name === stageName) {
      s.classList.add('active');
      s.classList.remove('done');
      passed = true;
      document.getElementById('progressSubtext').textContent =
        s.querySelector('span').textContent + '...';
    } else if (!passed) {
      s.classList.remove('active');
      s.classList.add('done');
    } else {
      s.classList.remove('active', 'done');
    }
  });
}

function completeAllStages() {
  document.querySelectorAll('#pipelineStages .stage').forEach(s => {
    s.classList.remove('active');
    s.classList.add('done');
  });
  document.getElementById('progressSubtext').textContent = 'Complete!';
}

// ===== Results rendering =====
function showResultsSection(data) {
  uploadSection.classList.add('hidden');
  progressSection.classList.add('hidden');
  resultsSection.classList.remove('hidden');
  exportBtns.classList.remove('hidden');
  exportBtns.classList.add('flex');

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

  // Question blocks
  data.results.forEach(qr => {
    const block = el('div', 'question-block');

    // Question header
    const header = el('div', 'flex items-center justify-between mb-4');
    header.innerHTML = `
      <h3 class="text-lg font-semibold">Question ${esc(qr.question_number)}</h3>
      <span class="text-xs text-zinc-500">Pipeline ${esc(qr.pipeline)} · ${esc(qr.subject)}</span>
    `;
    block.appendChild(header);

    // Question text
    const qText = el('div', 'text-sm text-zinc-400 mb-5 leading-relaxed');
    qText.textContent = qr.question_text;
    block.appendChild(qText);

    if (qr.errors && qr.errors.length > 0) {
      const errDiv = el('div', 'bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-sm text-red-400 mb-4');
      errDiv.textContent = qr.errors.join('; ');
      block.appendChild(errDiv);
    }

    if (qr.versions && qr.versions.length > 0) {
      // Version tabs
      const tabRow = el('div', 'flex gap-1 overflow-x-auto');
      const panels = [];
      qr.versions.forEach((v, idx) => {
        const tab = el('button', 'version-tab' + (idx === 0 ? ' active' : ''));
        tab.textContent = `V${v.version_number}`;
        tab.title = v.approach_label || '';
        tab.onclick = () => {
          tabRow.querySelectorAll('.version-tab').forEach(t => t.classList.remove('active'));
          tab.classList.add('active');
          panels.forEach((p, pi) => {
            p.style.display = pi === idx ? 'block' : 'none';
          });
        };
        tabRow.appendChild(tab);
      });
      block.appendChild(tabRow);

      // Version panels
      qr.versions.forEach((v, idx) => {
        const panel = el('div', 'answer-card');
        if (idx !== 0) panel.style.display = 'none';

        // Badges
        const badgeRow = el('div', 'flex flex-wrap gap-2 mb-4');
        if (v.approach_label) {
          badgeRow.innerHTML += `<span class="badge-model">${esc(v.approach_label)}</span>`;
        }
        badgeRow.innerHTML += `<span class="badge-model">${esc(v.provider)}</span>`;
        if (v.verified) {
          badgeRow.innerHTML += `<span class="badge-verified">✓ Verified</span>`;
        }
        if (v.quality_score != null) {
          badgeRow.innerHTML += `<span class="badge-score">Score: ${v.quality_score}</span>`;
        }
        panel.appendChild(badgeRow);

        // Answer text
        const ansDiv = el('div', 'answer-text');
        ansDiv.innerHTML = formatAnswer(v.answer_text);
        panel.appendChild(ansDiv);

        // Explanation toggle
        if (v.explanation_text) {
          const toggleBtn = el('button', 'mt-4 text-xs text-shark-400 hover:text-shark-300 flex items-center gap-1 transition');
          toggleBtn.innerHTML = `
            <svg class="w-3.5 h-3.5 transition-transform expl-arrow" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
            Show Explanation
          `;
          const explPanel = el('div', 'explanation-panel mt-3');
          const explContent = el('div', 'answer-text text-sm');
          explContent.innerHTML = formatAnswer(v.explanation_text);
          explPanel.appendChild(explContent);

          toggleBtn.onclick = () => {
            const open = explPanel.classList.toggle('open');
            toggleBtn.querySelector('.expl-arrow').style.transform = open ? 'rotate(180deg)' : '';
            toggleBtn.childNodes[toggleBtn.childNodes.length - 1].textContent =
              open ? ' Hide Explanation' : ' Show Explanation';
          };

          panel.appendChild(toggleBtn);
          panel.appendChild(explPanel);
        }

        panels.push(panel);
        block.appendChild(panel);
      });
    }

    resultsContainer.appendChild(block);
  });

  // Render math
  renderMath();

  // Update total cost in sidebar
  updateCostDisplay(cost.total_cost_usd || 0);
}

// ===== Math rendering =====
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
    // KaTeX not loaded yet, retry
    setTimeout(renderMath, 500);
  }
}

// ===== Format answer text (markdown-lite) =====
function formatAnswer(text) {
  if (!text) return '';
  let html = esc(text);
  // Bold: **text**
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Headers: lines starting with ### or ##
  html = html.replace(/^### (.+)$/gm, '<div class="text-base font-semibold text-zinc-200 mt-4 mb-1">$1</div>');
  html = html.replace(/^## (.+)$/gm, '<div class="text-lg font-semibold text-zinc-100 mt-5 mb-2">$1</div>');
  // Bullet lists
  html = html.replace(/^[-•] (.+)$/gm, '<div class="ml-4 flex gap-2"><span class="text-shark-400">•</span><span>$1</span></div>');
  // Numbered lists
  html = html.replace(/^(\d+)\. (.+)$/gm, '<div class="ml-4 flex gap-2"><span class="text-shark-400 font-mono text-xs mt-0.5">$1.</span><span>$2</span></div>');
  // Code blocks
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="bg-zinc-800/50 rounded-lg p-3 mt-2 mb-2 overflow-x-auto text-xs font-mono text-green-300">$2</pre>');
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code class="bg-zinc-800 px-1.5 py-0.5 rounded text-xs text-orange-300">$1</code>');
  // Newlines
  html = html.replace(/\n/g, '<br>');
  return html;
}

// ===== Export functions =====
async function exportPDF() {
  if (!currentResult) return;
  try {
    const resp = await fetch('/api/export/pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: currentResult }),
    });
    if (!resp.ok) throw new Error('Export failed');
    const blob = await resp.blob();
    downloadBlob(blob, 'shark_answer_results.pdf');
  } catch (e) {
    alert('PDF export failed: ' + e.message);
  }
}

async function exportDocx() {
  if (!currentResult) return;
  try {
    const resp = await fetch('/api/export/docx', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: currentResult }),
    });
    if (!resp.ok) throw new Error('Export failed');
    const blob = await resp.blob();
    downloadBlob(blob, 'shark_answer_results.docx');
  } catch (e) {
    alert('DOCX export failed: ' + e.message);
  }
}

function openPreview() {
  if (!currentResult) return;
  // Find submission id from history
  const lastEntry = document.querySelector('#historyList .history-item');
  if (lastEntry) {
    const sid = lastEntry.dataset.id;
    window.open(`/preview/${sid}`, '_blank');
  }
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ===== History =====
async function loadHistory() {
  try {
    const resp = await fetch('/api/history');
    if (!resp.ok) return;
    const items = await resp.json();
    renderHistory(items);
  } catch (e) {
    // silent
  }
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
    const subjectLabel = h.subject.replace('_', ' ');
    const nameStr = h.filenames ? h.filenames[0] : 'Paper';
    div.innerHTML = `
      <div class="text-sm font-medium truncate">${esc(nameStr)}</div>
      <div class="text-xs text-zinc-500 mt-0.5">${esc(subjectLabel)} · ${h.total_questions}Q · ${esc(h.timestamp)}</div>
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
    // highlight
    historyList.querySelectorAll('.history-item').forEach(item => {
      item.classList.toggle('active', item.dataset.id === sid);
    });
  } catch (e) {
    // silent
  }
}

// ===== Examiner profiles =====
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
  } catch (e) {
    // silent
  }
}

// ===== UI state helpers =====
function showUploadSection() {
  selectedFiles = [];
  currentResult = null;
  renderPreviews();
  updateSubmitButton();
  uploadSection.classList.remove('hidden');
  progressSection.classList.add('hidden');
  resultsSection.classList.add('hidden');
  exportBtns.classList.add('hidden');
  // Reset stages
  document.querySelectorAll('#pipelineStages .stage').forEach(s => {
    s.classList.remove('active', 'done');
  });
}

function showProgressSection() {
  uploadSection.classList.add('hidden');
  progressSection.classList.remove('hidden');
  resultsSection.classList.add('hidden');
  exportBtns.classList.add('hidden');
}

function showError(message) {
  progressSection.classList.add('hidden');
  uploadSection.classList.remove('hidden');
  // Show error toast
  const toast = el('div', 'fixed bottom-6 right-6 bg-red-600/90 text-white px-5 py-3 rounded-xl shadow-2xl text-sm z-50 max-w-md');
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 6000);
}

function updateCostDisplay(cost) {
  document.getElementById('totalCostDisplay').textContent = `Session: $${cost.toFixed(4)}`;
}

// ===== Utilities =====
function el(tag, className) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  return e;
}

function esc(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}
