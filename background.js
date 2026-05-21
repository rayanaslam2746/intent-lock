import { f as getGoalPreset } from "./chunks/goalPresets-2bmHbkX7.js";
import { env as transformersEnv, pipeline as createPipeline } from "./chunks/transformers.web-C0dh_jD5.js";
async function storageGet(key) {
  return (await chrome.storage.local.get(key))[key];
}
async function storageSet(key, value) {
  await chrome.storage.local.set({ [key]: value });
}
async function storageRemove(key) {
  await chrome.storage.local.remove(key);
}
const EMBEDDING_CACHE_KEY = "intent-lock:embedding-cache-v2",
  EMBEDDING_CACHE_MAX = 250;
async function getCachedEmbedding(textHash) {
  return ((await storageGet(EMBEDDING_CACHE_KEY)) ?? {})[textHash]?.embedding ?? null;
}
async function setCachedEmbedding(textHash, embedding) {
  const cache = (await storageGet(EMBEDDING_CACHE_KEY)) ?? {};
  cache[textHash] = { textHash: textHash, embedding: embedding, timestamp: Date.now() };
  const sorted = Object.values(cache).sort((a, b) => b.timestamp - a.timestamp),
    trimmed = Object.fromEntries(sorted.slice(0, EMBEDDING_CACHE_MAX).map((entry) => [entry.textHash, entry]));
  await storageSet(EMBEDDING_CACHE_KEY, trimmed);
}
async function hashText(text) {
  const encoded = new TextEncoder().encode(text),
    hashBuffer = await crypto.subtle.digest("SHA-256", encoded);
  return [...new Uint8Array(hashBuffer)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}
let transformersPipeline = null;
async function getEmbeddingPipeline() {
  if (!transformersPipeline) {
    transformersEnv.allowLocalModels = !1;
    transformersEnv.allowRemoteModels = !0;
    transformersPipeline = await createPipeline("feature-extraction", "Xenova/bge-small-en-v1.5");
  }
  return transformersPipeline;
}
function extractEmbeddingArray(output) {
  return Array.isArray(output) ? output.map(Number) : Array.from(output.data, Number);
}
async function getTextEmbedding(text) {
  const normalized = text.trim().replace(/\s+/g, " ");
  if (!normalized) return [];
  const textHash = await hashText(normalized),
    cached = await getCachedEmbedding(textHash);
  if (cached) return cached;
  const pipelineOutput = await (await getEmbeddingPipeline())(normalized, { pooling: "mean", normalize: !0 }),
    embeddingArray = extractEmbeddingArray(pipelineOutput);
  return (await setCachedEmbedding(textHash, embeddingArray), embeddingArray);
}
async function warmUpEmbeddingPipeline() {
  await getEmbeddingPipeline();
}
function cosineSimilarity(vecA, vecB) {
  if (vecA.length === 0 || vecB.length === 0 || vecA.length !== vecB.length) return 0;
  let dotProduct = 0,
    magA = 0,
    magB = 0;
  for (let idx = 0; idx < vecA.length; idx += 1)
    ((dotProduct += vecA[idx] * vecB[idx]), (magA += vecA[idx] * vecA[idx]), (magB += vecB[idx] * vecB[idx]));
  return magA === 0 || magB === 0 ? 0 : dotProduct / (Math.sqrt(magA) * Math.sqrt(magB));
}
function computeAlignmentScore(goalEmbedding, pageEmbedding) {
  const similarity = cosineSimilarity(goalEmbedding, pageEmbedding),
    minThreshold = 0.35,
    normalized = Math.min(1, Math.max(0, (similarity - minThreshold) / (0.85 - minThreshold))),
    curved =
      normalized < 0.5
        ? 0.5 * Math.pow(normalized / 0.5, 1.35)
        : 1 - 0.5 * Math.pow((1 - normalized) / 0.5, 1.35);
  return Math.round(Math.min(100, Math.max(0, curved * 100)));
}
const ANALYTICS_QUEUE_KEY = "intent-lock:analytics-queue",
  SUPABASE_URL = void 0,
  SUPABASE_KEY = void 0;
function isAnalyticsEnabled() {
  return !!SUPABASE_URL;
}
async function enqueueAnalyticsEvent(event) {
  const queue = (await storageGet(ANALYTICS_QUEUE_KEY)) ?? [];
  await storageSet(ANALYTICS_QUEUE_KEY, [...queue, event].slice(-500));
}
async function sendToSupabase(table, payload) {
  return isAnalyticsEnabled()
    ? (
        await fetch(`${SUPABASE_URL}/rest/v1/${table}`, {
          method: "POST",
          headers: {
            apikey: SUPABASE_KEY,
            Authorization: `Bearer ${SUPABASE_KEY}`,
            "Content-Type": "application/json",
            Prefer: "return=minimal",
          },
          body: JSON.stringify(payload),
        })
      ).ok
    : !1;
}
async function trackAnalyticsEvent(eventType, table, payload) {
  try {
    (await sendToSupabase(table, payload)) ||
      (await enqueueAnalyticsEvent({ type: eventType, payload: payload, createdAt: Date.now() }));
  } catch {
    await enqueueAnalyticsEvent({ type: eventType, payload: payload, createdAt: Date.now() });
  }
}
async function trackSessionStart(session) {
  await trackAnalyticsEvent("session", "sessions", {
    id: session.sessionId,
    goal: session.goal,
    start_time: new Date(session.startTime).toISOString(),
    avg_score: session.currentAlignmentScore,
  });
}
async function trackSessionEnd(session) {
  const avgScore = session.recentScores.length
    ? session.recentScores.reduce((sum, score) => sum + score, 0) / session.recentScores.length
    : session.currentAlignmentScore;
  await trackAnalyticsEvent("session", "sessions", {
    id: session.sessionId,
    goal: session.goal,
    start_time: new Date(session.startTime).toISOString(),
    end_time: new Date().toISOString(),
    avg_score: avgScore,
  });
}
async function trackPageEvent(event) {
  await trackAnalyticsEvent("event", "events", {
    id: event.id,
    session_id: event.sessionId,
    timestamp: new Date(event.timestamp).toISOString(),
    url: event.url,
    domain: event.domain,
    score: event.score,
    drift_state: event.driftState,
    decision: event.decision,
  });
}
async function trackDriftEvent(sessionId, triggerReason) {
  await trackAnalyticsEvent("drift_event", "drift_events", {
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
      (await sendToSupabase(table, item.payload)) || failed.push(item);
    } catch {
      failed.push(item);
    }
  }
  await storageSet(ANALYTICS_QUEUE_KEY, failed);
}
function extractDomain(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "unknown";
  }
}
function isInternalUrl(url) {
  return url ? /^(chrome|edge|brave|about|file|view-source):/i.test(url) : !0;
}
function normalizeDomain(url) {
  return url
    .trim()
    .toLowerCase()
    .replace(/^https?:\/\//, "")
    .replace(/^www\./, "")
    .split("/")[0];
}
function domainMatchesList(domain, list) {
  const normalizedDomain = normalizeDomain(domain);
  return list.some((entry) => {
    const normalizedEntry = normalizeDomain(entry);
    return normalizedDomain === normalizedEntry || normalizedDomain.endsWith(`.${normalizedEntry}`);
  });
}
function deduplicateDomains(domains) {
  return [...new Set(domains.map(normalizeDomain).filter(Boolean))];
}
const DRIFT_SCORE_THRESHOLD = 55,
  FOCUS_SCORE_THRESHOLD = 72,
  MIN_LOW_SCORE_EVENTS = 2,
  LOW_SCORE_DURATION_MS = 45 * 1e3;
function takeWhile(arr, predicate) {
  const result = [];
  for (const item of arr) {
    if (!predicate(item)) break;
    result.push(item);
  }
  return result;
}
function appendSessionEvent(session, event) {
  return {
    ...session,
    currentAlignmentScore: event.score,
    recentScores: [...session.recentScores, event.score].slice(-20),
    recentEvents: [...session.recentEvents, event].slice(-20),
  };
}
function evaluateDriftState(session, now = Date.now()) {
  const scores = session.recentScores,
    events = session.recentEvents,
    avgScore = scores.length ? scores.reduce((sum, score) => sum + score, 0) / scores.length : 100,
    consecutiveLowScoreEvents = takeWhile([...events].reverse(), (event) => event.score < DRIFT_SCORE_THRESHOLD),
    oldestLowScoreEvent = consecutiveLowScoreEvents.at(-1),
    lowScoreDurationMs = oldestLowScoreEvent ? now - oldestLowScoreEvent.timestamp : 0,
    latestScore = scores.at(-1) ?? 100,
    isImmediateDrift = latestScore < DRIFT_SCORE_THRESHOLD,
    isSustainedDrift = avgScore < DRIFT_SCORE_THRESHOLD && consecutiveLowScoreEvents.length >= MIN_LOW_SCORE_EVENTS && lowScoreDurationMs > LOW_SCORE_DURATION_MS;
  return isImmediateDrift || isSustainedDrift
    ? {
        driftState: "DRIFT_CONFIRMED",
        shouldNotify:
          now >= session.notificationCooldownUntil &&
          session.driftState !== "DRIFT_CONFIRMED",
        triggerReason: isImmediateDrift
          ? `Immediate hard drift: latest score ${latestScore}`
          : `Rolling average ${avgScore.toFixed(1)} with ${consecutiveLowScoreEvents.length} low-score pages for ${Math.round(lowScoreDurationMs / 1e3)} seconds`,
      }
    : avgScore < DRIFT_SCORE_THRESHOLD || consecutiveLowScoreEvents.length >= 2
      ? { driftState: "POSSIBLE_DRIFT", shouldNotify: !1 }
      : avgScore >= FOCUS_SCORE_THRESHOLD
        ? { driftState: "ACTIVE_FOCUS", shouldNotify: !1 }
        : { driftState: session.driftState, shouldNotify: !1 };
}
const ACTIVE_SESSION_KEY = "intent-lock:active-session",
  DRIFT_NOTIFICATION_ID = "intent-lock-drift",
  NOTIFICATION_COOLDOWN_MS = 120 * 1e3,
  BLOCK_SCORE_THRESHOLD = 58;
async function getActiveSession() {
  const raw = (await storageGet(ACTIVE_SESSION_KEY)) ?? null;
  return !raw?.sessionId || !raw.goal || !raw.goalEmbedding
    ? null
    : {
        sessionId: raw.sessionId,
        goal: raw.goal,
        presetId: raw.presetId,
        mode: raw.mode ?? "custom",
        allowDomains: raw.allowDomains ?? [],
        blockDomains: raw.blockDomains ?? [],
        goalEmbedding: raw.goalEmbedding,
        startTime: raw.startTime ?? Date.now(),
        currentAlignmentScore: raw.currentAlignmentScore ?? 100,
        driftState: raw.driftState ?? "ACTIVE_FOCUS",
        recentScores: raw.recentScores ?? [],
        recentEvents: raw.recentEvents ?? [],
        notificationCooldownUntil: raw.notificationCooldownUntil ?? 0,
      };
}
async function saveActiveSession(session) {
  await storageSet(ACTIVE_SESSION_KEY, session);
}
function buildAnalyticsSummary(session) {
  if (!session) return { avgScore: 0, totalEvents: 0, driftEvents: 0, elapsedMs: 0 };
  const totalEvents = session.recentEvents.length;
  return {
    avgScore: session.recentScores.length
      ? Math.round(
          session.recentScores.reduce((sum, score) => sum + score, 0) / session.recentScores.length,
        )
      : session.currentAlignmentScore,
    totalEvents: totalEvents,
    driftEvents: session.recentEvents.filter(
      (event) => event.driftState === "DRIFT_CONFIRMED",
    ).length,
    elapsedMs: Date.now() - session.startTime,
  };
}
async function startSession(goalText, presetId, extraAllowDomains = [], extraBlockDomains = []) {
  const existingSession = await getActiveSession();
  existingSession && (await trackSessionEnd(existingSession));
  const preset = getGoalPreset(presetId),
    resolvedGoal = preset?.label ?? goalText,
    allowDomains = deduplicateDomains([...(preset?.allowDomains ?? []), ...extraAllowDomains]),
    blockDomains = deduplicateDomains([...(preset?.blockDomains ?? []), ...extraBlockDomains]).filter((domain) => !domainMatchesList(domain, allowDomains));
  await warmUpEmbeddingPipeline();
  const goalEmbedding = await getTextEmbedding(resolvedGoal);
  if (!goalEmbedding.length)
    return { ok: !1, error: "Could not generate an embedding for that goal." };
  const newSession = {
    sessionId: crypto.randomUUID(),
    goal: resolvedGoal,
    presetId: preset?.id,
    mode: preset ? "preset" : "custom",
    allowDomains: allowDomains,
    blockDomains: blockDomains,
    goalEmbedding: goalEmbedding,
    startTime: Date.now(),
    currentAlignmentScore: 100,
    driftState: "ACTIVE_FOCUS",
    recentScores: [],
    recentEvents: [],
    notificationCooldownUntil: 0,
  };
  return (await saveActiveSession(newSession), trackSessionStart(newSession), evaluateActiveTab(), { ok: !0, session: newSession, analytics: buildAnalyticsSummary(newSession) });
}
async function getActiveTabInfo() {
  const [activeTab] = await chrome.tabs.query({ active: !0, currentWindow: !0 });
  return !activeTab?.url || isInternalUrl(activeTab.url)
    ? { ok: !0, activeTab: null }
    : { ok: !0, activeTab: { url: activeTab.url, domain: extractDomain(activeTab.url) } };
}
async function addSiteRule(domain, rule) {
  const session = await getActiveSession();
  if (!session)
    return { ok: !1, error: "Start a session before changing site rules." };
  const normalizedDomain = deduplicateDomains([domain])[0];
  if (!normalizedDomain) return { ok: !1, error: "Could not read a valid domain." };
  const updatedSession =
    rule === "allow"
      ? {
          ...session,
          allowDomains: deduplicateDomains([...session.allowDomains, normalizedDomain]),
          blockDomains: session.blockDomains.filter((entry) => !domainMatchesList(normalizedDomain, [entry])),
        }
      : {
          ...session,
          blockDomains: deduplicateDomains([...session.blockDomains, normalizedDomain]),
          allowDomains: session.allowDomains.filter((entry) => !domainMatchesList(normalizedDomain, [entry])),
        };
  return (await saveActiveSession(updatedSession), evaluateActiveTab(), { ok: !0, session: updatedSession, analytics: buildAnalyticsSummary(updatedSession) });
}
async function stopSession() {
  const session = await getActiveSession();
  return (
    session && (await trackSessionEnd(session), await storageRemove(ACTIVE_SESSION_KEY)),
    { ok: !0, session: null, analytics: buildAnalyticsSummary(null) }
  );
}
async function showDriftNotification(session) {
  (await chrome.notifications.clear(DRIFT_NOTIFICATION_ID),
    await chrome.notifications.create(DRIFT_NOTIFICATION_ID, {
      type: "basic",
      iconUrl: chrome.runtime.getURL("icons/icon128.svg"),
      title: "Intent Lock",
      message:
        "This page looks misaligned with your goal. Intent Lock is blocking low-alignment browsing unless you choose to continue intentionally.",
      buttons: [
        { title: "Continue intentionally" },
        { title: "5-minute detour" },
      ],
      priority: 1,
    }),
    await saveActiveSession({ ...session, notificationCooldownUntil: Date.now() + NOTIFICATION_COOLDOWN_MS }));
}
async function injectBlockPage(tabId, session, score) {
  try {
    await chrome.tabs.sendMessage(tabId, {
      type: "BLOCK_PAGE",
      goal: session.goal,
      score: score,
      threshold: BLOCK_SCORE_THRESHOLD,
    });
  } catch {
    try {
      (await chrome.scripting.executeScript({
        target: { tabId: tabId },
        files: ["contentScript.js"],
      }),
        await chrome.tabs.sendMessage(tabId, {
          type: "BLOCK_PAGE",
          goal: session.goal,
          score: score,
          threshold: BLOCK_SCORE_THRESHOLD,
        }));
    } catch {}
  }
}
async function processPageSummary(pageSummary, tabId) {
  const session = await getActiveSession();
  if (!(!session || isInternalUrl(pageSummary.url)))
    try {
      const domain = extractDomain(pageSummary.url),
        isAllowed = domainMatchesList(domain, session.allowDomains),
        isBlocked = !isAllowed && domainMatchesList(domain, session.blockDomains),
        decision = isAllowed ? "allowed" : isBlocked ? "blocked" : "semantic";
      let score = 0;
      if (isAllowed) score = 100;
      else if (isBlocked) score = 0;
      else {
        const pageEmbedding = await getTextEmbedding(pageSummary.text);
        if (!pageEmbedding.length) return;
        score = computeAlignmentScore(session.goalEmbedding, pageEmbedding);
      }
      const pageEvent = {
        id: crypto.randomUUID(),
        sessionId: session.sessionId,
        timestamp: Date.now(),
        url: pageSummary.url,
        domain: domain,
        score: score,
        driftState: session.driftState,
        decision: decision,
      };
      let updatedSession = appendSessionEvent(session, pageEvent);
      const driftResult = evaluateDriftState(updatedSession);
      ((pageEvent.driftState = driftResult.driftState),
        (updatedSession = {
          ...updatedSession,
          driftState: driftResult.driftState,
          recentEvents: [...updatedSession.recentEvents.slice(0, -1), pageEvent],
        }),
        await saveActiveSession(updatedSession),
        trackPageEvent(pageEvent),
        flushAnalyticsQueue(),
        driftResult.shouldNotify &&
          (trackDriftEvent(
            session.sessionId,
            isBlocked
              ? `Visited blocked domain for ${session.goal}: ${pageEvent.domain}`
              : (driftResult.triggerReason ?? "Sustained low-alignment browsing"),
          ),
          await showDriftNotification(updatedSession)),
        tabId && score < BLOCK_SCORE_THRESHOLD && (await injectBlockPage(tabId, updatedSession, score)));
    } catch (err) {
      console.warn(
        "Intent Lock: skipped page summary after processing error",
        err,
      );
    }
}
async function extractPageContent(tabId) {
  try {
    const response = await chrome.tabs.sendMessage(tabId, { type: "EXTRACT_PAGE" });
    if (response?.ok && response.summary) return response.summary;
  } catch {
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tabId },
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
  if (!(await getActiveSession())) return;
  const [activeTab] = await chrome.tabs.query({ active: !0, currentWindow: !0 });
  if (!activeTab?.id || isInternalUrl(activeTab.url)) return;
  const pageSummary = await extractPageContent(activeTab.id);
  pageSummary && (await processPageSummary(pageSummary, activeTab.id));
}
chrome.runtime.onMessage.addListener(
  (message, sender, sendResponse) => (
    (async () => {
      if (message.type === "START_SESSION")
        return startSession(message.goal, message.presetId, message.allowDomains, message.blockDomains);
      if (message.type === "STOP_SESSION") return stopSession();
      if (message.type === "GET_ACTIVE_TAB") return getActiveTabInfo();
      if (message.type === "ADD_SITE_RULE") return addSiteRule(message.domain, message.rule);
      if (message.type === "GET_STATE") {
        const session = await getActiveSession();
        return { ok: !0, session: session, analytics: buildAnalyticsSummary(session) };
      }
      if (message.type === "PAGE_SUMMARY") {
        await processPageSummary(message.summary);
        const session = await getActiveSession();
        return { ok: !0, session: session, analytics: buildAnalyticsSummary(session) };
      }
      return { ok: !1, error: "Unknown message type." };
    })()
      .then(sendResponse)
      .catch((err) =>
        sendResponse({
          ok: !1,
          error: err instanceof Error ? err.message : "Unexpected error",
        }),
      ),
    !0
  ),
);
chrome.tabs.onActivated.addListener(() => {
  evaluateActiveTab();
});
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  changeInfo.status === "complete" && tab.active && evaluateActiveTab();
});
chrome.notifications.onButtonClicked.addListener((notificationId, buttonIndex) => {
  notificationId === DRIFT_NOTIFICATION_ID &&
    (async () => {
      const session = await getActiveSession();
      session &&
        (buttonIndex === 0 &&
          (await saveActiveSession({
            ...session,
            driftState: "ACTIVE_FOCUS",
            notificationCooldownUntil: Date.now() + NOTIFICATION_COOLDOWN_MS,
          })),
        buttonIndex === 1 &&
          (await saveActiveSession({
            ...session,
            notificationCooldownUntil: Date.now() + 300 * 1e3,
          })),
        await chrome.notifications.clear(DRIFT_NOTIFICATION_ID));
    })();
});
chrome.notifications.onClicked.addListener((notificationId) => {
  notificationId === DRIFT_NOTIFICATION_ID && chrome.action.openPopup();
});