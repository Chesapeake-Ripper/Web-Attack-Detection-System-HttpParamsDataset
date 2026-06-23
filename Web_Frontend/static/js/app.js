/* WAD · app.js */
const MODEL_KEY = 'wad_last_model';

document.addEventListener('DOMContentLoaded', () => {

  // ── 模型选择持久化 ────────────────────────────────────────
  // 恢复上次选择的模型
  const saved = localStorage.getItem(MODEL_KEY);

  // 处理所有 radio[name="model"] 和 radio[name="model_sel"]
  ['model', 'model_sel'].forEach(name => {
    const radios = document.querySelectorAll(`input[type="radio"][name="${name}"]`);
    if (!radios.length) return;

    // 恢复上次选择
    if (saved) {
      radios.forEach(r => { r.checked = (r.value === saved); });
    }

    // 监听变化并保存
    radios.forEach(r => {
      r.addEventListener('change', () => {
        localStorage.setItem(MODEL_KEY, r.value);
        // 同步批量检测页面的隐藏字段
        ['hidModel1', 'hidModel2'].forEach(id => {
          const el = document.getElementById(id);
          if (el) el.value = r.value;
        });
      });
    });

    // 同步隐藏字段的初始值
    const checked = document.querySelector(`input[type="radio"][name="${name}"]:checked`);
    if (checked) {
      ['hidModel1', 'hidModel2'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = checked.value;
      });
    }
  });

  // ── 防重复提交 ────────────────────────────────────────────
  document.querySelectorAll('form').forEach(f => {
    f.addEventListener('submit', () => {
      const btn = f.querySelector('button[type="submit"]');
      if (btn && !btn.disabled) {
        btn.disabled = true;
        setTimeout(() => { btn.disabled = false; }, 15000);
      }
    });
  });

  // ── Ctrl+Enter 快速提交 ───────────────────────────────────
  document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      const f = document.getElementById('sForm') || document.querySelector('form');
      if (f) f.requestSubmit();
    }
  });

});
// ── AI 深度分析 ───────────────────────────────────────────
(function initAiAnalyze() {
  const btn = document.getElementById('btnAnalyze');
  if (!btn) return;

  // 极简 Markdown → HTML（只处理 ###/加粗/代码/列表/换行，无需引入第三方库）
  function mdToHtml(md) {
    return md
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/^### (.+)$/gm, '<h3>$1</h3>')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/^[-*] (.+)$/gm, '<li>$1</li>')
      .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')   // 包裹列表
      .replace(/\n{2,}/g, '</p><p>')
      .replace(/\n/g, '<br>')
      .replace(/^(?!<[hup])(.+)$/gm, '<p>$1</p>')  // 普通段落
      .replace(/<p><\/p>/g, '');
  }

  btn.addEventListener('click', async () => {
    const body   = document.getElementById('aiBody');
    const payload    = btn.dataset.payload;
    const label      = btn.dataset.label;
    const labelCn    = btn.dataset.labelCn;
    const confidence = btn.dataset.confidence;
    const model      = btn.dataset.model;

    // 显示加载动画
    btn.disabled = true;
    body.innerHTML = `
      <div class="ai-loading">
        <div class="ai-dots">
          <span></span><span></span><span></span>
        </div>
        AI 正在分析，请稍候…
      </div>`;

    try {
      const resp = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ payload, label, label_cn: labelCn,
                               confidence: parseFloat(confidence), model }),
      });
      const data = await resp.json();

      if (data.success) {
        const html = mdToHtml(data.data.analysis);
        const modelTag = data.data.model || 'AI';
        body.innerHTML = `
          <div class="ai-result">${html}</div>
          <div class="ai-model-tag">Powered by ⚡: ${modelTag}</div>`;
      } else {
        body.innerHTML = `<div class="ai-error">⚠ ${data.error || '分析失败'}</div>`;
      }
    } catch (e) {
      body.innerHTML = `<div class="ai-error">⚠ 请求失败：${e.message}</div>`;
    } finally {
      btn.disabled = false;
      btn.innerHTML = '<i class="bi bi-arrow-clockwise me-1"></i>重新分析';
    }
  });
})();