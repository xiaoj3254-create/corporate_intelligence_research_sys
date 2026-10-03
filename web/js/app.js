/* 企业情报调研系统 · Vue 3 SPA（Options API + 全局构建，免编译） */
/* global Vue, marked, DOMPurify */

const { createApp } = Vue;

// ---------------------------------------------------------------- API 封装

async function api(path, options) {
  const resp = await fetch(path, options);
  if (!resp.ok) {
    let msg = `HTTP ${resp.status}`;
    try { msg = (await resp.json()).detail || msg; } catch (_) { /* ignore */ }
    throw new Error(msg);
  }
  return resp.json();
}

const API = {
  createTask: (body) => api('/api/tasks', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }),
  listTasks: () => api('/api/tasks'),
  getTask: (id) => api(`/api/tasks/${id}`),
  getEvents: (id) => api(`/api/tasks/${id}/events`),
  getResult: (id) => api(`/api/tasks/${id}/result`),
  getApproval: (id) => api(`/api/tasks/${id}/approval`),
  resume: (id, decision) => api(`/api/tasks/${id}/resume`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ decision }),
  }),
  kbSearch: (q, k = 8) => api(`/api/kb/search?q=${encodeURIComponent(q)}&k=${k}`),
};

// ---------------------------------------------------------------- 工具

// 事件/节点的语义色分类: 颜色即信息
function eventClass(ev) {
  if (ev.status === 'error') return 't-error';
  if (ev.status === 'pending') return 't-pending';
  const key = `${ev.node || ''} ${ev.agent || ''}`;
  if (/risk|compliance/.test(key)) return 't-risk';
  if (/analysis|finance|market|synthesiz/.test(key)) return 't-analysis';
  if (/deep|researcher|planner|critic/.test(key)) return 't-deep';
  if (/collection|search_agent|registry|patent/.test(key)) return 't-collection';
  return '';
}

const STATUS_TEXT = {
  running: '执行中',
  awaiting_approval: '待人工审批',
  completed: '已完成',
  failed: '执行失败',
  pending: '待执行',
};

function renderMarkdown(md) {
  if (!md) return '';
  const raw = marked.parse(md, { breaks: true });
  return DOMPurify.sanitize(raw);
}

// ---------------------------------------------------------------- 组件

const StatusPill = {
  props: { status: String, large: Boolean },
  computed: {
    text() { return STATUS_TEXT[this.status] || this.status; },
  },
  template: `
    <span class="pill" :class="[status, {large}]">
      <span class="dot"></span>{{ text }}
    </span>`,
};

const TimelinePanel = {
  props: { events: Array, polling: Boolean },
  methods: { eventClass },
  template: `
    <div>
      <div class="tl-wrap">
        <div v-if="!events.length" class="tl-empty">暂无执行记录，等待 Agent 启动…</div>
        <div v-for="(ev, i) in events" :key="i" class="tl-card" :class="eventClass(ev)">
          <div class="tl-head">
            <span class="tl-node">{{ ev.node }}</span>
            <span class="tl-agent">{{ ev.agent }}</span>
            <span v-if="ev.status==='error'" style="color:var(--c-error)">异常</span>
            <span class="tl-time">{{ ev.ts }}</span>
          </div>
          <div class="tl-task">{{ ev.task_desc || ev.task }}</div>
          <div class="tl-detail" :class="{'tl-error-text': ev.status==='error'}">{{ ev.detail }}</div>
        </div>
      </div>
      <p v-if="polling" class="tl-polling"><span class="dot"></span>实时刷新中…</p>
    </div>`,
};

const ReportPanel = {
  props: { taskId: String, refreshKey: Number },
  data: () => ({ report: '', findings: [], loading: true, error: '' }),
  computed: {
    renderedHtml() { return renderMarkdown(this.report); },
  },
  watch: { refreshKey() { this.load(); } },
  created() { this.load(); },
  methods: {
    async load() {
      this.loading = true;
      this.error = '';
      try {
        const data = await API.getResult(this.taskId);
        this.report = data.report || '';
        this.findings = data.findings || [];
        if (!this.report) this.error = '该任务尚未生成报告（可能仍在执行或等待审批）。';
      } catch (e) {
        this.error = `加载失败: ${e.message}`;
      }
      this.loading = false;
    },
    download() {
      const blob = new Blob([this.report], { type: 'text/markdown;charset=utf-8' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `情报报告_${this.taskId}.md`;
      a.click();
      URL.revokeObjectURL(a.href);
    },
  },
  template: `
    <div>
      <div v-if="loading" class="tl-empty">加载中…</div>
      <template v-else>
        <div v-if="error" class="tl-empty">{{ error }}</div>
        <template v-else>
          <div class="report-actions">
            <button class="btn ghost" @click="download">下载报告 (Markdown)</button>
          </div>
          <div class="report-md" v-html="renderedHtml"></div>
          <h3 style="margin-top:28px">📎 情报明细（{{ findings.length }} 条）</h3>
          <div v-for="(f, i) in findings" :key="i" class="finding-item">
            <div class="finding-head">
              <span class="finding-team">[{{ f.team }}/{{ f.agent }}]</span>
              <span>{{ f.task_desc }}</span>
              <span class="finding-cred" :class="'cred-' + f.credibility">
                <span class="dot"></span>{{ f.credibility }}
              </span>
            </div>
            <p class="finding-content">{{ (f.content || '').slice(0, 600) }}</p>
            <div v-if="f.source_urls && f.source_urls.length" class="finding-urls">
              <a v-for="(u, j) in f.source_urls.slice(0, 3)" :key="j"
                 :href="u" target="_blank" rel="noopener noreferrer">来源 {{ j + 1 }} ↗</a>
            </div>
          </div>
        </template>
      </template>
    </div>`,
};

const KbPanel = {
  data: () => ({ q: '', results: null, loading: false, error: '' }),
  methods: {
    async search() {
      if (!this.q.trim()) return;
      this.loading = true;
      this.error = '';
      try {
        const data = await API.kbSearch(this.q.trim());
        this.results = data.results;
      } catch (e) {
        this.error = e.message;
        this.results = [];
      }
      this.loading = false;
    },
  },
  template: `
    <div>
      <form class="kb-search" @submit.prevent="search">
        <input v-model="q" placeholder="例如：专利布局 / 供应链风险">
        <button class="btn primary" :disabled="loading">检索</button>
      </form>
      <p v-if="error" class="form-error">{{ error }}</p>
      <div v-if="loading" class="tl-empty">检索中…</div>
      <div v-else-if="results === null" class="kb-empty">输入关键词检索历史调研沉淀的情报（Chroma 向量 + BM25 混合检索，RRF 融合）</div>
      <div v-else-if="!results.length" class="kb-empty">知识库为空或无相关内容——完成一次调研后情报会自动沉淀。</div>
      <div v-for="(r, i) in results" :key="i" class="kb-result">
        <div class="kb-meta">[{{ r.metadata.team || 'report' }}/{{ r.metadata.agent || '' }}] {{ r.metadata.company || '' }}</div>
        <div class="kb-content">{{ r.content.slice(0, 400) }}</div>
      </div>
    </div>`,
};

const ApprovalPanel = {
  props: { taskId: String },
  emits: ['decided'],
  data: () => ({ draft: '', risks: [], submitting: false, feedback: '', error: '' }),
  created() { this.load(); },
  methods: {
    async load() {
      try {
        const data = await API.getApproval(this.taskId);
        this.draft = data.draft;
        this.risks = (data.findings || []).filter(
          (f) => f.team === 'risk' && f.credibility !== 'low');
      } catch (e) {
        this.error = `加载审批快照失败: ${e.message}`;
      }
    },
    async decide(decision) {
      if (decision !== 'approve' && !this.feedback.trim()) {
        this.error = '请填写修改意见，或直接点击「通过」。';
        return;
      }
      this.submitting = true;
      this.error = '';
      try {
        await API.resume(this.taskId, decision === 'approve' ? 'approve' : this.feedback.trim());
        this.$emit('decided');
      } catch (e) {
        this.error = `提交失败: ${e.message}`;
      }
      this.submitting = false;
    },
  },
  template: `
    <div class="approval-panel">
      <div class="approval-banner">⏸ 报告草稿已生成，等待人工审批（状态已持久化，刷新页面不丢失）</div>
      <p v-if="error" class="form-error">{{ error }}</p>

      <div v-if="risks.length" class="risk-list">
        <h3>⚠️ 风控组风险信号（{{ risks.length }} 条）</h3>
        <div v-for="(f, i) in risks" :key="i" class="risk-item">
          <b>{{ f.agent }}</b> — {{ (f.content || '').slice(0, 300) }}
        </div>
      </div>

      <details open>
        <summary style="cursor:pointer;margin-bottom:8px">📄 报告草稿（点击展开/收起）</summary>
        <div class="report-md" v-html="rendered"></div>
      </details>

      <div class="approval-actions">
        <label>修改意见（不通过时填写）
          <textarea v-model="feedback" rows="3"
                    placeholder="例如：补充对供应链集中度风险的分析…"></textarea>
        </label>
        <div class="approval-buttons">
          <button class="btn primary" style="flex:1" :disabled="submitting"
                  @click="decide('approve')">✅ 通过，生成最终报告</button>
          <button class="btn ghost" style="flex:1" :disabled="submitting"
                  @click="decide('revise')">📝 按修改意见重新生成</button>
        </div>
      </div>
    </div>`,
  computed: {
    rendered() { return renderMarkdown(this.draft); },
  },
};

// ---------------------------------------------------------------- 根应用

createApp({
  components: { StatusPill, TimelinePanel, ReportPanel, KbPanel, ApprovalPanel },
  data: () => ({
    form: { company: '', objective: '', mode: 'standard', autoApprove: false },
    formError: '',
    submitting: false,
    tasks: [],
    current: null,        // 当前选中任务（DB 行）
    events: [],
    activeTab: 'timeline',
    tabs: [
      { key: 'timeline', label: '🛰️ Agent 时间线' },
      { key: 'report', label: '📄 调研报告' },
      { key: 'kb', label: '📚 情报知识库' },
    ],
    polling: false,
    pollTimer: null,
    resultVersion: 0,     // 触发报告面板重新加载
  }),
  created() {
    this.refreshTasks();
  },
  unmounted() {
    this.stopPolling();
  },
  methods: {
    async refreshTasks() {
      try {
        this.tasks = await API.listTasks();
        // 保持当前任务状态同步（后台线程会更新 DB 状态）
        if (this.current) {
          const fresh = this.tasks.find((t) => t.id === this.current.id);
          if (fresh) {
            const prev = this.current.status;
            this.current = fresh;
            if (prev !== fresh.status && (fresh.status === 'completed' || fresh.status === 'failed')) {
              this.resultVersion += 1;
            }
          }
        }
      } catch (_) { /* 静默，下次轮询重试 */ }
    },
    async submitTask() {
      if (!this.form.company) {
        this.formError = '请输入公司名称';
        return;
      }
      this.submitting = true;
      this.formError = '';
      try {
        const { task_id } = await API.createTask({
          company: this.form.company,
          objective: this.form.objective,
          mode: this.form.mode,
          auto_approve: this.form.autoApprove,
        });
        this.form.company = '';
        this.form.objective = '';
        await this.refreshTasks();
        const task = this.tasks.find((t) => t.id === task_id);
        if (task) this.selectTask(task);
        this.activeTab = 'timeline';
      } catch (e) {
        this.formError = `提交失败: ${e.message}`;
      }
      this.submitting = false;
    },
    async selectTask(t) {
      this.stopPolling();
      this.current = t;
      this.activeTab = t.status === 'awaiting_approval' ? 'approval' : 'timeline';
      this.events = [];
      await this.loadEvents();
      if (['running', 'awaiting_approval'].includes(t.status)) this.startPolling();
    },
    async loadEvents() {
      if (!this.current) return;
      try {
        this.events = await API.getEvents(this.current.id);
      } catch (_) { /* 下次轮询重试 */ }
    },
    startPolling() {
      this.polling = true;
      this.pollTimer = setInterval(async () => {
        await this.refreshTasks();
        await this.loadEvents();
        if (this.current && !['running', 'awaiting_approval'].includes(this.current.status)) {
          this.stopPolling();
          this.resultVersion += 1;
        }
      }, 2000);
    },
    stopPolling() {
      this.polling = false;
      if (this.pollTimer) { clearInterval(this.pollTimer); this.pollTimer = null; }
    },
    async onDecided() {
      // 审批提交后：切回时间线并开始轮询
      await this.refreshTasks();
      this.activeTab = 'timeline';
      if (this.current) {
        const fresh = this.tasks.find((t) => t.id === this.current.id);
        if (fresh) this.current = fresh;
      }
      this.startPolling();
    },
  },
}).mount('#app');
