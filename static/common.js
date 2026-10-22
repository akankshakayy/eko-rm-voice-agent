const toLogin = () => { location.href = "/login?next=" + encodeURIComponent(location.pathname); };
const api = async (path, opts = {}) => {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined });
  if (r.status === 401) { toLogin(); throw new Error("Not signed in"); }
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
};
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtTime = ts => ts ? new Date(ts * 1000).toLocaleString("en-IN", { timeZone: "Asia/Kolkata", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }) : "–";
const fmtDur = s => { if (s == null) return "–"; s = Math.round(s); return s >= 60 ? `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s` : `${s}s`; };
const ago = ts => { const s = Date.now() / 1000 - ts; if (s < 60) return `${Math.round(s)}s ago`; if (s < 3600) return `${Math.round(s / 60)}m ago`; if (s < 86400) return `${Math.round(s / 3600)}h ago`; return `${Math.round(s / 86400)}d ago`; };
const num = (v, d = 0) => v == null ? "–" : Number(v).toLocaleString("en-IN", { maximumFractionDigits: d });

const OUTCOME = {
  ticket_raised: ["Ticket raised", "b-accent"], escalated: ["Escalated to RM", "b-warn"],
  not_understood: ["Not understood", "b-bad"], resolved_by_agent: ["Solved by agent", "b-ok"],
  dropped: ["Dropped", "b-bad"], info_only: ["No issue", ""],
};
const STATUS = { open: ["Open", "b-warn"], in_progress: ["In progress", "b-info"], resolved: ["Resolved", "b-ok"], closed: ["Closed", "b-ok"] };
const PRIORITY = { critical: "b-bad", high: "b-warn", medium: "b-info", low: "" };
const badge = (map, k) => { const v = map[k]; return v ? `<span class="badge ${v[1]}">${v[0]}</span>` : `<span class="badge">${esc(k || "–")}</span>`; };
const prBadge = p => `<span class="badge ${PRIORITY[p] || ""}">${esc(p)}</span>`;

let CONFIG = null;
const loadConfig = async () => (CONFIG = CONFIG || await api("/api/config"));
const catLabel = k => (CONFIG?.categories?.[k]?.label) || (k === "unclassified" ? "Unclassified" : k === "unknown" ? "Unknown" : k || "–");

function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
const PALETTE = () => ["--c1", "--c2", "--c3", "--c4", "--c5", "--c6", "--c7", "--c8"].map(cssVar);

// One hue per issue category, so an issue is the same colour in chips, bars and charts on every page
const CAT_TONE = { dmt_failed: "--c1", aeps_issue: "--c5", wallet_topup: "--c6", commission: "--c3", kyc_onboarding: "--c10",
  app_device: "--c9", bbps_recharge: "--c7", refund_status: "--c2", fraud_report: "--c4", ticket_status: "--c2", general_query: "--c8" };
const catTone = k => CAT_TONE[k] || "--c8";
const catColor = k => cssVar(catTone(k));
const catChip = (k, text) => `<span class="cat" style="--tone:var(${catTone(k)})">${esc(text ?? catLabel(k))}</span>`;

let ME = null;
const loadMe = async () => (ME = ME || await api("/api/auth/me"));

async function signOut() {
  try { await fetch("/api/auth/logout", { method: "POST" }); } finally { location.href = "/login"; }
}

// Which roles can open each page (the server enforces the same rules)
const PAGES = [["/call", "Call console", ["admin", "ops"]], ["/rm", "RM dashboard", ["admin", "ops", "rm"]],
  ["/ops", "Agent & model performance", ["admin", "ops"]], ["/users", "Users", ["admin"]]];
const canSee = (u, path) => PAGES.find(p => p[0] === path)?.[2].includes(u.role);

// ---------------------------------------------------------------- theme (light / dark, remembered per browser)
const THEME_KEY = "eko-theme";
function savedTheme() { try { return localStorage.getItem(THEME_KEY); } catch (e) { return null; } }
function currentTheme() {
  return document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}
function toggleTheme() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem(THEME_KEY, next); } catch (e) {}
  // Charts read their colours when drawn, so redraw pages that have them
  if (window.Chart && Object.keys(Chart.instances || {}).length) location.reload();
  else document.querySelectorAll(".theme-btn").forEach(b => b.innerHTML = themeIcon());
}
const themeIcon = () => currentTheme() === "dark"
  ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4.5"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>`
  : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/></svg>`;
if (savedTheme()) document.documentElement.dataset.theme = savedTheme();

// ---------------------------------------------------------------- small notification
function toast(msg) {
  let t = document.getElementById("toast");
  if (!t) { t = document.createElement("div"); t.id = "toast"; t.className = "toast"; t.setAttribute("role", "status"); document.body.append(t); }
  t.textContent = msg; t.classList.add("show");
  clearTimeout(toast.timer); toast.timer = setTimeout(() => t.classList.remove("show"), 1800);
}

const initials = name => String(name || "?").trim().split(/\s+/).slice(0, 2).map(w => w[0]).join("").toUpperCase();

function navHtml(active) {
  const l = PAGES;
  queueMicrotask(() => loadMe().then(u => {
    document.querySelectorAll("header.top nav a").forEach(a => { if (!canSee(u, a.getAttribute("href"))) a.remove(); });
    const box = document.getElementById("userBox");
    const role = u.role === "rm" ? `RM, ${esc(u.rm_region || "")}` : esc(u.role_label);
    if (box) box.innerHTML = `<span class="who" title="${esc(u.email)}"><span class="avatar" aria-hidden="true">${esc(initials(u.name))}</span>
        <span class="who-text"><b>${esc(u.name)}</b><span>${role}</span></span></span>
      <button class="secondary icon-btn theme-btn" onclick="toggleTheme()" title="Switch light / dark" aria-label="Switch light or dark theme">${themeIcon()}</button>
      <button class="secondary small" onclick="signOut()">Sign out</button>`;
  }).catch(() => {}));
  return `<a class="brand" href="/"><img src="/static/icons/eko-logo.webp" alt="Eko" width="46" height="22"><span>Sahayak</span></a><nav>${l.map(([h, t]) =>
    `<a href="${h}" class="${active === h ? "active" : ""}">${t}</a>`).join("")}</nav><div class="userbox" id="userBox"></div>`;
}

function chartDefaults() {
  if (!window.Chart) return;
  Chart.defaults.color = cssVar("--muted");
  Chart.defaults.borderColor = cssVar("--border");
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  Chart.defaults.plugins.legend.labels.boxWidth = 8;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.cornerRadius = 8;
  Chart.defaults.plugins.tooltip.boxPadding = 4;
  Chart.defaults.elements.bar.borderRadius = 6;
  Chart.defaults.elements.line.tension = 0.35;
  Chart.defaults.elements.point.radius = 2.5;
  Chart.defaults.elements.arc.borderColor = cssVar("--panel");
  Chart.defaults.animation.duration = 600;
  Chart.defaults.scale.grid.color = cssVar("--panel-2");
  Chart.defaults.scale.border = { display: false };
}
// Vertical (or horizontal) fade from a colour to a lighter tint, for bar fills
function fade(color, horizontal = false) {
  return ctx => {
    const { chart } = ctx, a = chart.chartArea;
    if (!a) return color;
    const g = horizontal ? chart.ctx.createLinearGradient(a.left, 0, a.right, 0) : chart.ctx.createLinearGradient(0, a.bottom, 0, a.top);
    g.addColorStop(0, color + "88"); g.addColorStop(1, color);
    return g;
  };
}
