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