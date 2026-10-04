export type FacetItem = {
  value: string;
  count: number;
};

export type HighlightBlock = {
  title?: string[];
  summary?: string[];
  content?: string[];
};

export type SearchExplain = {
  query: string;
  folded_query: string;
  rerank?: boolean;
  reranker_model?: string | null;
  raw_fields: string[];
  folded_fields: string[];
  filters: Record<string, string | null>;
  sort: "relevance" | "newest" | string;
};

export type SearchResult = {
  id: string;
  doc_id: string;
  chunk_id: string;
  score: number;
  title: string;
  summary: string;
  url: string;
  category: string;
  author: string;
  source: string;
  published_at: string;
  highlight?: HighlightBlock;
  metadata?: Record<string, unknown>;
};

export type SearchMode = "keyword" | "semantic" | "hybrid";

export type SearchResponse = {
  query: string;
  mode: SearchMode;
  total: number;
  page: number;
  page_size: number;
  latency_ms: number;
  results: SearchResult[];
  facets: {
    category: FacetItem[];
    author: FacetItem[];
    source: FacetItem[];
  };
  explain?: SearchExplain | null;
  /** True khi E5 model dang duoc load lan dau — frontend nen hien thi loading va tu retry. */
  semantic_loading?: boolean;
};

export type SuggestResponse = {
  query: string;
  latency_ms: number;
  suggestions: Array<{
    text: string;
    type: string;
    weight: number;
  }>;
};

export type DiagnosticsResponse = {
  app_name: string;
  environment: string;
  mode: string;
  backend: {
    name: string;
    source: string;
    reachable: boolean;
    error?: string | null;
  };
  indexes: {
    article_data_path: string;
    suggestion_data_path: string;
    article_count: number;
    suggestion_count: number;
    vocabulary_size: number;
    postings_count: number;
    article_index_ready: boolean;
    suggestion_index_ready: boolean;
    keyword?: Record<string, unknown>;
    semantic?: Record<string, unknown>;
    hybrid?: Record<string, unknown>;
  };
  reports: {
    preprocessing_report_available: boolean;
    preprocessing_summary?: string | null;
    evaluation_metrics_available: boolean;
    evaluation_metrics: Array<Record<string, string>>;
  };
};

export type SearchFilters = {
  mode: SearchMode;
  rerank: boolean;
  category: string;
  author: string;
  source: string;
  fromDate: string;
  toDate: string;
  sort: "relevance" | "newest";
};

export type PipelineStep = {
  id: string;
  name: string;
  artifact: string;
  status: "done" | "active" | "planned";
  risk: string;
};
