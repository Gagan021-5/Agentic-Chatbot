/**
 * Turn structured outbound user payloads into readable chat text for Redis + UI reload.
 * Raw `content` stays canonical for routing; `displayContent` is optional human view.
 */
export function formatUserPayloadForHistory(message) {
  const t = String(message || "").trim();
  const lower = t.toLowerCase();

  if (lower.startsWith("multi_select_form::")) {
    const jsonPart = t.slice("multi_select_form::".length);
    try {
      const payload = JSON.parse(jsonPart);
      const features = Array.isArray(payload.selectedOptions)
        ? payload.selectedOptions.filter(Boolean)
        : [];
      const vars = Array.isArray(payload.variables) ? payload.variables : [];
      const lines = ["Confirmed settings ✓"];
      if (features.length) {
        lines.push("");
        lines.push(`Features: ${features.join(", ")}`);
      }
      const varLines = vars.map((v) => {
        const name = (v && v.name) || "Field";
        const hasVal = v && v.value !== undefined && v.value !== null && String(v.value).trim() !== "";
        const val = hasVal
          ? String(v.value).trim()
          : v && v.placeholder
            ? String(v.placeholder).trim()
            : "—";
        return `${name}: ${val}`;
      });
      if (varLines.length) {
        lines.push("");
        lines.push(...varLines);
      }
      return lines.join("\n");
    } catch {
      return "Confirmed settings ✓";
    }
  }

  if (lower.startsWith("confirm seo::")) {
    const jsonPart = t.slice(t.indexOf("::") + 2);
    try {
      const payload = JSON.parse(jsonPart);
      const lines = ["Confirmed SEO metadata ✓", ""];
      for (const [k, v] of Object.entries(payload)) {
        lines.push(`${k}: ${typeof v === "object" ? JSON.stringify(v) : String(v)}`);
      }
      return lines.join("\n");
    } catch {
      return "Confirmed SEO metadata ✓";
    }
  }

  if (lower.startsWith("edit prompt::")) {
    const body = t.slice(t.indexOf("::") + 2).trim();
    return body ? `Updated prompt template:\n\n${body}` : "Updated prompt template";
  }

  return null;
}
