import type { SearchResult } from "../types";

type ResultCardProps = {
  result: SearchResult;
  rank: number;
  onOpen: (articleId: string, position: number) => void;
};

export function ResultCard({ result, rank, onOpen }: ResultCardProps) {
  const scoreInfo = getScoreDisplay(result);

  return (
    <article className="result-item">
      <div className="rank-badge" aria-label={`Kết quả số ${rank}`}>
        {String(rank).padStart(2, "0")}
      </div>
      <div className="result-body">
        <div className="result-meta">
          <span>{result.source || "Nguồn chưa rõ"}</span>
          <span>{result.category || "Chưa phân loại"}</span>
          <span>{formatDate(result.published_at)}</span>
        </div>
        <a
          href={result.url}
          target="_blank"
          rel="noreferrer"
          className="result-heading"
          onClick={() => onOpen(result.id, rank)}
          dangerouslySetInnerHTML={renderHighlightedText(result.highlight?.title?.[0] || result.title)}
        />
        <p
          className={result.highlight?.summary?.[0] ? "result-highlight" : "result-summary"}
          dangerouslySetInnerHTML={renderHighlightedText(result.highlight?.summary?.[0] || result.summary)}
        />
        <div className="result-actions">
          <div className="score-tag-wrapper">
            <span className="score-label">{scoreInfo.label}:</span>
            <span className="score-value">{scoreInfo.val}</span>
            {scoreInfo.details && (
              <span className="score-details">{scoreInfo.details}</span>
            )}
          </div>
          <a href={result.url} target="_blank" rel="noreferrer" onClick={() => onOpen(result.id, rank)}>
            Mở bài viết
          </a>
        </div>
      </div>
    </article>
  );
}

function getScoreDisplay(result: SearchResult) {
  const score = result.score;
  const meta = result.metadata || {};

  if (score === undefined || score === null || score === 0) {
    return {
      label: "Độ phù hợp",
      val: "N/A",
      details: null
    };
  }

  // 1. RERANKED
  if (meta.reranked) {
    const retScore = typeof meta.retrieval_score === "number" ? meta.retrieval_score.toFixed(3) : null;
    return {
      label: "Reranked",
      val: score.toFixed(4),
      details: retScore ? `(Gốc: ${retScore})` : null
    };
  }

  // 2. HYBRID (RRF)
  if (meta.matched_modes && Array.isArray(meta.matched_modes)) {
    const modes = meta.matched_modes.map((m) => m === "keyword" ? "Từ khóa" : "Semantic").join(" + ");
    return {
      label: "Hybrid (RRF)",
      val: score.toFixed(4),
      details: `(${modes})`
    };
  }

  // 3. SEMANTIC or KEYWORD (General)
  // Cosine similarity is usually in the range [0, 1]. BM25 is usually >= 1.0 (up to 30+).
  const isPossiblySemantic = score > 0 && score <= 1.0;
  return {
    label: isPossiblySemantic ? "Semantic" : "Từ khóa (BM25)",
    val: score.toFixed(3),
    details: null
  };
}

function formatDate(value: string) {
  if (!value) {
    return "Không rõ ngày";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Không rõ ngày";
  }

  return new Intl.DateTimeFormat("vi-VN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function escapeHtml(value: string) {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function sanitizeHighlight(value: string) {
  return escapeHtml(value)
    .replaceAll("&lt;em&gt;", "<em>")
    .replaceAll("&lt;/em&gt;", "</em>");
}

function renderHighlightedText(value: string) {
  return { __html: sanitizeHighlight(value) };
}
