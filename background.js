import * as _transformersModule from "./chunks/transformers.web-C0dh_jD5.js";

// ---------------------------------------------------------------------------
// Vite module-preload polyfill (do not modify)
// ---------------------------------------------------------------------------
const MODULE_PRELOAD_REL = "modulepreload";
const resolvePath = function (t) { return "/" + t; };
const preloadedModules = {};
const vitePreload = function (loadFn, deps, baseUrl) {
  let readyPromise = Promise.resolve();
  if (deps && deps.length > 0) {
    let allSettled = function (promises) {
      return Promise.all(
        promises.map((p) =>
          Promise.resolve(p).then(
            (v) => ({ status: "fulfilled", value: v }),
            (e) => ({ status: "rejected", reason: e }),
          ),
        ),
      );
    };
    var settled = allSettled;
    document.getElementsByTagName("link");
    const nonceMeta = document.querySelector("meta[property=csp-nonce]");
    const nonce = nonceMeta?.nonce || nonceMeta?.getAttribute("nonce");
    readyPromise = allSettled(
      deps.map((dep) => {
        if (((dep = resolvePath(dep)), dep in preloadedModules)) return;
        preloadedModules[dep] = true;
        const isCSS = dep.endsWith(".css");
        const cssSelector = isCSS ? '[rel="stylesheet"]' : "";
        if (document.querySelector(`link[href="${dep}"]${cssSelector}`)) return;
        const link = document.createElement("link");
        if (
          ((link.rel = isCSS ? "stylesheet" : MODULE_PRELOAD_REL),
          isCSS || (link.as = "script"),
          (link.crossOrigin = ""),
          (link.href = dep),
          nonce && link.setAttribute("nonce", nonce),
          document.head.appendChild(link),
          isCSS)
        )
          return new Promise((resolve, reject) => {
            link.addEventListener("load", resolve);
            link.addEventListener("error", () =>
              reject(new Error(`Unable to preload CSS for ${dep}`)),
            );
          });
      }),
    );
  }
  function throwPreloadError(err) {
    const event = new Event("vite:preloadError", { cancelable: true });
    if (((event.payload = err), window.dispatchEvent(event), !event.defaultPrevented))
      throw err;
  }
  return readyPromise.then((results) => {
    for (const result of results || [])
      result.status === "rejected" && throwPreloadError(result.reason);
    return loadFn().catch(throwPreloadError);
  });
};

// ---------------------------------------------------------------------------
// Chrome storage helpers
// ---------------------------------------------------------------------------
async function storageGet(key) {
  return (await chrome.storage.local.get(key))[key];
}
async function storageSet(key, value) {
  await chrome.storage.local.set({ [key]: value });
}
async function storageRemove(key) {
  await chrome.storage.local.remove(key);
}

// ---------------------------------------------------------------------------
// Embedding cache
// ---------------------------------------------------------------------------
const EMBEDDING_CACHE_KEY = "intent-lock:embedding-cache";
const EMBEDDING_CACHE_MAX_SIZE = 250;

async function getCachedEmbedding(textHash) {
  return ((await storageGet(EMBEDDING_CACHE_KEY)) ?? {})[textHash]?.embedding ?? null;
}

async function setCachedEmbedding(textHash, embedding) {
  const cache = (await storageGet(EMBEDDING_CACHE_KEY)) ?? {};
  cache[textHash] = { textHash, embedding, timestamp: Date.now() };
  const trimmed = Object.fromEntries(
    Object.values(cache)
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, EMBEDDING_CACHE_MAX_SIZE)
      .map((entry) => [entry.textHash, entry]),
  );
  await storageSet(EMBEDDING_CACHE_KEY, trimmed);
}

async function hashText(text) {
  const encoded = new TextEncoder().encode(text);
  const hashBuffer = await crypto.subtle.digest("SHA-256", encoded);
  return [...new Uint8Array(hashBuffer)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

// ---------------------------------------------------------------------------
// ML model (Xenova/all-MiniLM-L6-v2 via Transformers.js)
// ---------------------------------------------------------------------------
let pipelinePromise = null;

function polyfillWindowForServiceWorker() {
  const global = globalThis;
  global.window ??= globalThis;
}

async function getEmbeddingPipeline() {
  if (!pipelinePromise) {
    polyfillWindowForServiceWorker();
    pipelinePromise = Promise.resolve(_transformersModule).then(
      async ({ env, pipeline }) => {
        env.allowLocalModels = false;
        env.allowRemoteModels = true;
        return await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
      },
    );
  }
  return pipelinePromise;
}

function tensorToNumberArray(tensor) {
  return Array.isArray(tensor)
    ? tensor.map(Number)
    : Array.from(tensor.data, Number);
}

async function embedText(text) {
  const normalized = text.trim().replace(/\s+/g, " ");
  if (!normalized) return [];
  const hash = await hashText(normalized);
  const cached = await getCachedEmbedding(hash);
  if (cached) return cached;
  const result = await (await getEmbeddingPipeline())(normalized, {
    pooling: "mean",
    normalize: true,
  });
  const vector = tensorToNumberArray(result);
  await setCachedEmbedding(hash, vector);
  return vector;
}

async function warmUpModel() {
  await getEmbeddingPipeline();
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------
function cosineSimilarity(vecA, vecB) {
  if (vecA.length === 0 || vecB.length === 0 || vecA.length !== vecB.length)
    return 0;
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < vecA.length; i += 1) {
    dot  += vecA[i] * vecB[i];
    magA += vecA[i] * vecA[i];
    magB += vecB[i] * vecB[i];
  }
  return magA === 0 || magB === 0 ? 0 : dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

function similarityToAlignmentScore(goalVec, pageVec) {
  const raw = cosineSimilarity(goalVec, pageVec);
  const LOW_CLIP  = 0.18;
  const HIGH_CLIP = 0.62;
  const normalized = Math.min(1, Math.max(0, (raw - LOW_CLIP) / (HIGH_CLIP - LOW_CLIP)));
  // S-curve: compress extremes, expand the middle
  const curved = normalized < 0.5
    ? 0.5 * Math.pow(normalized / 0.5, 1.35)
    : 1 - 0.5 * Math.pow((1 - normalized) / 0.5, 1.35);
  return Math.round(Math.min(100, Math.max(0, curved * 100)));
}

// ---------------------------------------------------------------------------
// Analytics (Supabase — disabled until credentials are provided)
// ---------------------------------------------------------------------------
const ANALYTICS_QUEUE_KEY = "intent-lock:analytics-queue";
const SUPABASE_URL = void 0;  // set to your Supabase project URL to enable
const SUPABASE_ANON_KEY = void 0;

function isAnalyticsEnabled() {
  return !!SUPABASE_URL;
}

async function enqueueAnalyticsEvent(event) {
  const queue = (await storageGet(ANALYTICS_QUEUE_KEY)) ?? [];
  await storageSet(ANALYTICS_QUEUE_KEY, [...queue, event].slice(-500));
}

async function postToSupabase(table, payload) {
  if (!isAnalyticsEnabled()) return false;
  const res = await fetch(`${SUPABASE_URL}/rest/v1/${table}`, {
    method: "POST",
    headers: {
      apikey: SUPABASE_ANON_KEY,
      Authorization: `Bearer ${SUPABASE_ANON_KEY}`,
      "Content-Type": "application/json",
      Prefer: "return=minimal",
    },
    body: JSON.stringify(payload),
  });
  return res.ok;
}

async function logAnalyticsEvent(eventType, table, payload) {
  try {
    (await postToSupabase(table, payload)) ||
      (await enqueueAnalyticsEvent({ type: eventType, payload, createdAt: Date.now() }));
  } catch {
    await enqueueAnalyticsEvent({ type: eventType, payload, createdAt: Date.now() });
  }
}

async function logSessionStart(session) {
  await logAnalyticsEvent("session", "sessions", {
    id: session.sessionId,
    goal: session.goal,
    start_time: new Date(session.startTime).toISOString(),
    avg_score: session.currentAlignmentScore,
  });
}

async function logSessionEnd(session) {
  const avg = session.recentScores.length
    ? session.recentScores.reduce((sum, s) => sum + s, 0) / session.recentScores.length
    : session.currentAlignmentScore;
  await logAnalyticsEvent("session", "sessions", {
    id: session.sessionId,
    goal: session.goal,
    start_time: new Date(session.startTime).toISOString(),
    end_time: new Date().toISOString(),
    avg_score: avg,
  });
}

async function logPageEvent(event) {
  await logAnalyticsEvent("event", "events", {
    id: event.id,
    session_id: event.sessionId,
    timestamp: new Date(event.timestamp).toISOString(),
    url: event.url,
    domain: event.domain,
    score: event.score,
    drift_state: event.driftState,
  });
}

async function logDriftEvent(sessionId, triggerReason) {
  await logAnalyticsEvent("drift_event", "drift_events", {
    id: crypto.randomUUID(),
    session_id: sessionId,
    timestamp: new Date().toISOString(),
    trigger_reason: triggerReason,
  });
}

async function flushAnalyticsQueue() {
  const queue = (await storageGet(ANALYTICS_QUEUE_KEY)) ?? [];
  if (!queue.length || !isAnalyticsEnabled()) return;
  const failed = [];
  for (const item of queue) {
    const table = item.type === "drift_event" ? "drift_events" : `${item.type}s`;
    try {
      (await postToSupabase(table, item.payload)) || failed.push(item);
    } catch {
      failed.push(item);
    }
  }
  await storageSet(ANALYTICS_QUEUE_KEY, failed);
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function extractDomain(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "unknown";
  }
}

function isBrowserInternalUrl(url) {
  return url ? /^(chrome|edge|brave|about|file|view-source):/i.test(url) : true;
}

// ---------------------------------------------------------------------------
// Drift detection thresholds
// ---------------------------------------------------------------------------
const DRIFT_SCORE_THRESHOLD       = 55;   // below this = low alignment
const FOCUS_SCORE_THRESHOLD       = 72;   // above this = confirmed focus
const ROLLING_LOW_PAGES_REQUIRED  = 2;    // min consecutive low-score pages
const ROLLING_LOW_DURATION_MS     = 45 * 1000; // must sustain for 45s

function takeWhileFromEnd(arr, predicate) {
  const result = [];
  for (const item of arr) {
    if (!predicate(item)) break;
    result.push(item);
  }
  return result;
}

function appendPageEvent(session, pageEvent) {
  return {
    ...session,
    currentAlignmentScore: pageEvent.score,
    recentScores: [...session.recentScores, pageEvent.score].slice(-20),
    recentEvents: [...session.recentEvents, pageEvent].slice(-20),
  };
}

function evaluateDriftState(session, now = Date.now()) {
  const scores = session.recentScores;
  const events = session.recentEvents;
  const avgScore = scores.length
    ? scores.reduce((sum, s) => sum + s, 0) / scores.length
    : 100;

  // consecutive low-score pages (from most recent backwards)
  const recentLowPages = takeWhileFromEnd(
    [...events].reverse(),
    (e) => e.score < DRIFT_SCORE_THRESHOLD,
  );
  const oldestLowPage = recentLowPages.at(-1);
  const lowStreakDurationMs = oldestLowPage ? now - oldestLowPage.timestamp : 0;

  const latestScore = scores.at(-1) ?? 100;
  const isHardDrift    = latestScore < DRIFT_SCORE_THRESHOLD;
  const isRollingDrift =
    avgScore < DRIFT_SCORE_THRESHOLD &&
    recentLowPages.length >= ROLLING_LOW_PAGES_REQUIRED &&
    lowStreakDurationMs > ROLLING_LOW_DURATION_MS;

  if (isHardDrift || isRollingDrift) {
    return {
      driftState: "DRIFT_CONFIRMED",
      shouldNotify:
        now >= session.notificationCooldownUntil &&
        session.driftState !== "DRIFT_CONFIRMED",
      triggerReason: isHardDrift
        ? `Immediate hard drift: latest score ${latestScore}`
        : `Rolling average ${avgScore.toFixed(1)} with ${recentLowPages.length} low-score pages for ${Math.round(lowStreakDurationMs / 1000)} seconds`,
    };
  }
  if (avgScore < DRIFT_SCORE_THRESHOLD || recentLowPages.length >= 2)
    return { driftState: "POSSIBLE_DRIFT", shouldNotify: false };
  if (avgScore >= FOCUS_SCORE_THRESHOLD)
    return { driftState: "ACTIVE_FOCUS", shouldNotify: false };
  return { driftState: session.driftState, shouldNotify: false };
}

// ---------------------------------------------------------------------------
// Session storage
// ---------------------------------------------------------------------------
const ACTIVE_SESSION_KEY = "intent-lock:active-session";
const DRIFT_NOTIFICATION_ID = "intent-lock-drift";
const NOTIFICATION_COOLDOWN_MS = 120 * 1000;
const BLOCK_SCORE_THRESHOLD = 58;

async function loadSession() {
  return (await storageGet(ACTIVE_SESSION_KEY)) ?? null;
}
async function saveSession(session) {
  await storageSet(ACTIVE_SESSION_KEY, session);
}

function computeAnalytics(session) {
  if (!session)
    return { avgScore: 0, totalEvents: 0, driftEvents: 0, elapsedMs: 0 };
  return {
    avgScore: session.recentScores.length
      ? Math.round(
          session.recentScores.reduce((sum, s) => sum + s, 0) /
            session.recentScores.length,
        )
      : session.currentAlignmentScore,
    totalEvents: session.recentEvents.length,
    driftEvents: session.recentEvents.filter(
      (e) => e.driftState === "DRIFT_CONFIRMED",
    ).length,
    elapsedMs: Date.now() - session.startTime,
  };
}

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------
async function startSession(goal) {
  const existing = await loadSession();
  if (existing) await logSessionEnd(existing);
  await warmUpModel();
  const goalEmbedding = await embedText(goal);
  if (!goalEmbedding.length)
    return { ok: false, error: "Could not generate an embedding for that goal." };
  const session = {
    sessionId: crypto.randomUUID(),
    goal,
    goalEmbedding,
    startTime: Date.now(),
    currentAlignmentScore: 100,
    driftState: "ACTIVE_FOCUS",
    recentScores: [],
    recentEvents: [],
    notificationCooldownUntil: 0,
  };
  await saveSession(session);
  logSessionStart(session);
  evaluateActiveTab();
  return { ok: true, session, analytics: computeAnalytics(session) };
}

async function stopSession() {
  const session = await loadSession();
  if (session) {
    await logSessionEnd(session);
    await storageRemove(ACTIVE_SESSION_KEY);
  }
  return { ok: true, session: null, analytics: computeAnalytics(null) };
}

// ---------------------------------------------------------------------------
// Notifications
// ---------------------------------------------------------------------------
async function sendDriftNotification(session) {
  await chrome.notifications.clear(DRIFT_NOTIFICATION_ID);
  await chrome.notifications.create(DRIFT_NOTIFICATION_ID, {
    type: "basic",
    iconUrl: chrome.runtime.getURL("icons/icon128.svg"),
    title: "Intent Lock",
    message:
      "This page looks misaligned with your goal. Intent Lock is blocking low-alignment browsing unless you choose to continue intentionally.",
    buttons: [{ title: "Continue intentionally" }, { title: "5-minute detour" }],
    priority: 1,
  });
  await saveSession({
    ...session,
    notificationCooldownUntil: Date.now() + NOTIFICATION_COOLDOWN_MS,
  });
}

// ---------------------------------------------------------------------------
// Page blocking
// ---------------------------------------------------------------------------
async function sendBlockMessage(tabId, session, score) {
  const message = {
    type: "BLOCK_PAGE",
    goal: session.goal,
    score,
    threshold: BLOCK_SCORE_THRESHOLD,
  };
  try {
    await chrome.tabs.sendMessage(tabId, message);
  } catch {
    // Content script may not be injected yet — inject and retry
    try {
      await chrome.scripting.executeScript({
        target: { tabId },
        files: ["contentScript.js"],
      });
      await chrome.tabs.sendMessage(tabId, message);
    } catch { /* tab may have closed */ }
  }
}

// ---------------------------------------------------------------------------
// Page evaluation
// ---------------------------------------------------------------------------
async function evaluatePage(pageSummary, tabId) {
  const session = await loadSession();
  if (!session || isBrowserInternalUrl(pageSummary.url)) return;
  try {
    const pageEmbedding = await embedText(pageSummary.text);
    if (!pageEmbedding.length) return;

    const score = similarityToAlignmentScore(session.goalEmbedding, pageEmbedding);
    const pageEvent = {
      id: crypto.randomUUID(),
      sessionId: session.sessionId,
      timestamp: Date.now(),
      url: pageSummary.url,
      domain: extractDomain(pageSummary.url),
      score,
      driftState: session.driftState,
    };

    let updatedSession = appendPageEvent(session, pageEvent);
    const driftResult = evaluateDriftState(updatedSession);

    pageEvent.driftState = driftResult.driftState;
    updatedSession = {
      ...updatedSession,
      driftState: driftResult.driftState,
      recentEvents: [...updatedSession.recentEvents.slice(0, -1), pageEvent],
    };

    await saveSession(updatedSession);
    logPageEvent(pageEvent);
    flushAnalyticsQueue();

    if (driftResult.shouldNotify) {
      logDriftEvent(
        session.sessionId,
        driftResult.triggerReason ?? "Sustained low-alignment browsing",
      );
      await sendDriftNotification(updatedSession);
    }

    if (tabId && score < BLOCK_SCORE_THRESHOLD) {
      await sendBlockMessage(tabId, updatedSession, score);
    }
  } catch (err) {
    console.warn("Intent Lock: skipped page summary after processing error", err);
  }
}

async function extractPageSummary(tabId) {
  try {
    const response = await chrome.tabs.sendMessage(tabId, { type: "EXTRACT_PAGE" });
    if (response?.ok && response.summary) return response.summary;
  } catch {
    // Inject content script and retry
    try {
      await chrome.scripting.executeScript({
        target: { tabId },
        files: ["contentScript.js"],
      });
      const response = await chrome.tabs.sendMessage(tabId, { type: "EXTRACT_PAGE" });
      if (response?.ok && response.summary) return response.summary;
    } catch {
      return null;
    }
  }
  return null;
}

async function evaluateActiveTab() {
  if (!(await loadSession())) return;
  const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!activeTab?.id || isBrowserInternalUrl(activeTab.url)) return;
  const summary = await extractPageSummary(activeTab.id);
  if (summary) await evaluatePage(summary, activeTab.id);
}

// ---------------------------------------------------------------------------
// Message handler
// ---------------------------------------------------------------------------
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  (async () => {
    if (message.type === "START_SESSION") return startSession(message.goal);
    if (message.type === "STOP_SESSION")  return stopSession();
    if (message.type === "GET_STATE") {
      const session = await loadSession();
      return { ok: true, session, analytics: computeAnalytics(session) };
    }
    if (message.type === "PAGE_SUMMARY") {
      await evaluatePage(message.summary);
      const session = await loadSession();
      return { ok: true, session, analytics: computeAnalytics(session) };
    }
    return { ok: false, error: "Unknown message type." };
  })()
    .then(sendResponse)
    .catch((err) =>
      sendResponse({
        ok: false,
        error: err instanceof Error ? err.message : "Unexpected error",
      }),
    );
  return true; // keep message channel open for async response
});

// ---------------------------------------------------------------------------
// Tab event listeners
// ---------------------------------------------------------------------------
chrome.tabs.onActivated.addListener(() => evaluateActiveTab());

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && tab.active) evaluateActiveTab();
});

// ---------------------------------------------------------------------------
// Notification button handlers
// ---------------------------------------------------------------------------
chrome.notifications.onButtonClicked.addListener((notifId, buttonIndex) => {
  if (notifId !== DRIFT_NOTIFICATION_ID) return;
  (async () => {
    const session = await loadSession();
    if (!session) return;
    if (buttonIndex === 0) {
      // "Continue intentionally" — reset drift, apply cooldown
      await saveSession({
        ...session,
        driftState: "ACTIVE_FOCUS",
        notificationCooldownUntil: Date.now() + NOTIFICATION_COOLDOWN_MS,
      });
    } else if (buttonIndex === 1) {
      // "5-minute detour" — extend cooldown only
      await saveSession({
        ...session,
        notificationCooldownUntil: Date.now() + 300 * 1000,
      });
    }
    await chrome.notifications.clear(DRIFT_NOTIFICATION_ID);
  })();
});

chrome.notifications.onClicked.addListener((notifId) => {
  if (notifId === DRIFT_NOTIFICATION_ID) chrome.action.openPopup();
});