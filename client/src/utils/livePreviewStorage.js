const PREFIX = "rentprompts:live-preview:v1";

/** Data URLs can be large; try to keep full image up to ~2MB string, then strip on quota errors. */
const MAX_STORED_URL_CHARS = 2_000_000;

export function livePreviewStorageKey(sessionId, messageId) {
  if (!sessionId || !messageId) return null;
  return `${PREFIX}:${sessionId}:${messageId}`;
}

function slimPreviewResult(preview) {
  if (!preview || typeof preview !== "object") return null;
  const next = { ...preview };
  const u = next.url;
  if (typeof u === "string" && u.startsWith("data:") && u.length > MAX_STORED_URL_CHARS) {
    next.url = null;
    next._imageDroppedForStorage = true;
  }
  return next;
}

function slimTestImage(b64) {
  if (typeof b64 !== "string" || !b64.startsWith("data:")) return b64 || null;
  if (b64.length > MAX_STORED_URL_CHARS) return null;
  return b64;
}

export function packLivePreviewState({ isPreviewMode, testInputs, previewResult, testImage }) {
  return {
    v: 1,
    isPreviewMode: Boolean(isPreviewMode),
    testInputs: testInputs && typeof testInputs === "object" ? testInputs : {},
    previewResult: slimPreviewResult(previewResult),
    testImage: slimTestImage(testImage)
  };
}

export function saveLivePreviewToStorage(key, state) {
  if (!key) return;
  try {
    const packed = packLivePreviewState(state);
    const json = JSON.stringify(packed);
    localStorage.setItem(key, json);
  } catch (e) {
    if (e?.name === "QuotaExceededError" || e?.code === 22) {
      try {
        const packed = packLivePreviewState(state);
        if (packed.previewResult && typeof packed.previewResult === "object") {
          const pr = { ...packed.previewResult };
          if (typeof pr.url === "string" && pr.url.startsWith("data:")) {
            pr.url = null;
            pr._imageDroppedForStorage = true;
          }
          packed.previewResult = pr;
        }
        packed.testImage = null;
        localStorage.setItem(key, JSON.stringify(packed));
      } catch (e2) {
        console.warn("[live-preview] storage save failed", e2);
      }
    } else {
      console.warn("[live-preview] storage save failed", e);
    }
  }
}

export function loadLivePreviewFromStorage(key) {
  if (!key) return null;
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || parsed.v !== 1) return null;
    return parsed;
  } catch {
    return null;
  }
}
