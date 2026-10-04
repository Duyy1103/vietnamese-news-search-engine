import type { DiagnosticsResponse, SearchFilters, SearchResponse, SuggestResponse } from "../types";

// Dung duong dan tuong doi /api de Next.js proxy toi backend (tranh CORS)
export const API_BASE_URL = "/api";

export const INITIAL_SEARCH_RESPONSE: SearchResponse = {
  query: "",
  mode: "keyword",
  total: 0,
  page: 1,
  page_size: 10,
  latency_ms: 0,
  results: [],
  facets: {
    category: [],
    author: [],
    source: [],
  },
  explain: null,
};

export async function fetchSearch(
  query: string,
  filters: SearchFilters,
  page: number,
  signal?: AbortSignal,
): Promise<SearchResponse> {
  // Dung URLSearchParams thay vi new URL() de tuong thich voi ca duong dan
  // tuong doi (/api) lan URL tuyet doi (http://...)
  const params = new URLSearchParams();
  params.set("q", query);
  params.set("mode", filters.mode);
  if (filters.rerank) params.set("rerank", "true");
  params.set("page", String(page));
  params.set("page_size", "10");
  params.set("sort", filters.sort);
  if (filters.category) params.set("category", filters.category);
  if (filters.author) params.set("author", filters.author);
  if (filters.source) params.set("source", filters.source);
  if (filters.fromDate) params.set("from_date", filters.fromDate);
  if (filters.toDate) params.set("to_date", filters.toDate);

  try {
    const response = await fetch(`${API_BASE_URL}/search?${params}`, { cache: "no-store", signal });
    if (!response.ok) {
      throw new Error(await readErrorMessage(response));
    }
    return response.json();
  } catch (error) {
    if (isAbortError(error)) {
      throw error;
    }
    throw error;
  }
}

async function readErrorMessage(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    if (typeof payload?.detail === "string") {
      return payload.detail;
    }
  } catch {
    // Fall back to the generic HTTP status message below.
  }
  return `Search request failed with ${response.status}`;
}

export async function fetchSuggestions(query: string, signal?: AbortSignal): Promise<string[]> {
  const params = new URLSearchParams({ q: query });
  try {
    const response = await fetch(`${API_BASE_URL}/suggest?${params}`, { cache: "no-store", signal });
    if (!response.ok) return [];
    const data = (await response.json()) as SuggestResponse;
    return data.suggestions.map((item) => item.text);
  } catch (error) {
    if (isAbortError(error)) {
      return [];
    }
    throw error;
  }
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

export async function fetchDiagnostics(signal?: AbortSignal): Promise<DiagnosticsResponse | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/diagnostics`, { cache: "no-store", signal });
    if (!response.ok) return null;
    return response.json();
  } catch {
    return null;
  }
}

export async function fetchSemanticStatus(): Promise<{ ready: boolean; loading: boolean }> {
  try {
    const response = await fetch(`${API_BASE_URL}/semantic/status`, { cache: "no-store" });
    if (!response.ok) return { ready: false, loading: false };
    return response.json();
  } catch {
    return { ready: false, loading: false };
  }
}

export async function triggerSemanticWarmup(): Promise<void> {
  try {
    await fetch(`${API_BASE_URL}/semantic/warmup`, { method: "POST", cache: "no-store" });
  } catch {
    // Ignore — warmup la best-effort
  }
}

export function trackClick(articleId: string, query: string, position: number) {
  const payload = new Blob(
    [
      JSON.stringify({
        article_id: articleId,
        query,
        position,
      }),
    ],
    { type: "application/json" },
  );
  navigator.sendBeacon(`${API_BASE_URL}/events/click`, payload);
}
