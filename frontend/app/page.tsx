"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { FilterPanel } from "./components/FilterPanel";
import { ResultList } from "./components/ResultList";
import { SearchBar } from "./components/SearchBar";
import { fetchSearch, fetchSuggestions, INITIAL_SEARCH_RESPONSE, isAbortError, trackClick, triggerSemanticWarmup, fetchSemanticStatus } from "./lib/api";
import type { SearchFilters, SearchResponse } from "./types";

const QUICK_QUERIES = ["giá vàng", "nga ukraine", "địa phương điểm thi", "việt nam", "công an huế"];

const INITIAL_FILTERS: SearchFilters = {
  mode: "keyword",
  rerank: false,
  category: "",
  author: "",
  source: "",
  fromDate: "",
  toDate: "",
  sort: "relevance",
};

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [filters, setFilters] = useState<SearchFilters>(INITIAL_FILTERS);
  const [page, setPage] = useState(1);
  const [searchNonce, setSearchNonce] = useState(0);
  const [response, setResponse] = useState<SearchResponse>(INITIAL_SEARCH_RESPONSE);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [semanticLoading, setSemanticLoading] = useState(false);
  const semanticRetryRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const searchRequestId = useRef(0);

  const totalPages = useMemo(
    () => Math.max(1, Math.ceil(response.total / response.page_size)),
    [response.total, response.page_size],
  );

  useEffect(() => {
    const trimmed = query.trim();
    const controller = new AbortController();
    const timeout = setTimeout(async () => {
      if (trimmed.length < 2) {
        setSuggestions([]);
        return;
      }
      setSuggestions(await fetchSuggestions(trimmed, controller.signal));
    }, 260);

    return () => {
      controller.abort();
      clearTimeout(timeout);
    };
  }, [query]);

  // Khi nhan duoc response co semantic_loading=true:
  // hien thi thong bao va tu dong retry sau 3 giay
  useEffect(() => {
    if (response.semantic_loading) {
      setSemanticLoading(true);
      if (semanticRetryRef.current) clearTimeout(semanticRetryRef.current);
      semanticRetryRef.current = setTimeout(() => {
        setSearchNonce((value) => value + 1);
      }, 3000);
    } else {
      setSemanticLoading(false);
      if (semanticRetryRef.current) {
        clearTimeout(semanticRetryRef.current);
        semanticRetryRef.current = null;
      }
    }
    return () => {
      if (semanticRetryRef.current) clearTimeout(semanticRetryRef.current);
    };
  }, [response.semantic_loading]);

  useEffect(() => {
    const controller = new AbortController();
    const requestId = searchRequestId.current + 1;
    searchRequestId.current = requestId;

    void runSearch(page, submittedQuery, filters, controller.signal, requestId);

    return () => {
      controller.abort();
    };
  }, [page, filters, submittedQuery, searchNonce]);

  async function runSearch(
    nextPage: number,
    nextQuery: string,
    nextFilters: SearchFilters,
    signal: AbortSignal,
    requestId: number,
  ) {
    setLoading(true);
    setError("");
    try {
      const data = await fetchSearch(nextQuery, nextFilters, nextPage, signal);
      if (searchRequestId.current !== requestId) return;
      setResponse(data);
    } catch (err) {
      if (isAbortError(err) || searchRequestId.current !== requestId) return;
      setResponse(INITIAL_SEARCH_RESPONSE);
      setError(err instanceof Error ? err.message : "Không thể kết nối dịch vụ tìm kiếm.");
    } finally {
      if (searchRequestId.current !== requestId) return;
      setLoading(false);
    }
  }

  function submitSearch(nextQuery: string) {
    const trimmed = nextQuery.trim();
    setQuery(trimmed);
    setSubmittedQuery(trimmed);
    setResponse(INITIAL_SEARCH_RESPONSE);
    setPage(1);
    setSearchNonce((value) => value + 1);
  }

  function updateFilters(nextFilters: SearchFilters) {
    setFilters(nextFilters);
    setPage(1);
    // Khi chuyen sang semantic/hybrid, kich hoat warmup ngay
    // de model bat dau load truoc khi co query
    if (nextFilters.mode === "semantic" || nextFilters.mode === "hybrid") {
      void triggerSemanticWarmup();
    }
  }

  function resetFilters() {
    setFilters(INITIAL_FILTERS);
    setPage(1);
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div>
            <span className="micro-label">Vietnamese News Search</span>
            <h1>Tìm kiếm báo tiếng Việt</h1>
            <p>Tra cứu nhanh tin tức theo từ khóa, nguồn báo và chuyên mục.</p>
          </div>
        </div>
      </header>

      <section className="search-layout">
        <FilterPanel filters={filters} response={response} onChange={updateFilters} onReset={resetFilters} />
        <div className="center-column">
          <SearchBar
            query={query}
            mode={filters.mode}
            suggestions={suggestions}
            quickQueries={QUICK_QUERIES}
            loading={loading}
            onQueryChange={setQuery}
            onModeChange={(mode) => updateFilters({ ...filters, mode })}
            onSubmit={submitSearch}
            semanticLoading={semanticLoading}
          />
          <ResultList
            response={response}
            submittedQuery={submittedQuery}
            loading={loading}
            error={error}
            page={page}
            totalPages={totalPages}
            onPageChange={setPage}
            onOpen={(articleId, position) => trackClick(articleId, submittedQuery, position)}
            semanticLoading={semanticLoading}
          />
        </div>
      </section>
    </main>
  );
}
