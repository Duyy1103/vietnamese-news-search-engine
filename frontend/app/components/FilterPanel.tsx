import type { ReactNode } from "react";
import type { SearchFilters, SearchResponse } from "../types";

type FilterPanelProps = {
  filters: SearchFilters;
  response: SearchResponse;
  onChange: (filters: SearchFilters) => void;
  onReset: () => void;
};

export function FilterPanel({ filters, response, onChange, onReset }: FilterPanelProps) {
  return (
    <aside className="filter-rail" aria-label="Bộ lọc tìm kiếm">
      <div className="rail-header">
        <div>
          <span className="micro-label">Bộ lọc</span>
          <h2>Thu hẹp kết quả</h2>
        </div>
        <button type="button" className="ghost-action" onClick={onReset}>
          Xóa
        </button>
      </div>

      <Field label="Chuyên mục">
        <select value={filters.category} onChange={(event) => onChange({ ...filters, category: event.target.value })}>
          <option value="">Tất cả chuyên mục</option>
          {response.facets.category.map((item) => (
            <option key={item.value} value={item.value}>
              {item.value} ({item.count})
            </option>
          ))}
        </select>
      </Field>

      <Field label="Nguồn báo">
        <select value={filters.source} onChange={(event) => onChange({ ...filters, source: event.target.value })}>
          <option value="">Tất cả nguồn</option>
          {response.facets.source.map((item) => (
            <option key={item.value} value={item.value}>
              {item.value} ({item.count})
            </option>
          ))}
        </select>
      </Field>

      <Field label="Tác giả">
        <select value={filters.author} onChange={(event) => onChange({ ...filters, author: event.target.value })}>
          <option value="">Tất cả tác giả</option>
          {response.facets.author.map((item) => (
            <option key={item.value} value={item.value}>
              {item.value} ({item.count})
            </option>
          ))}
        </select>
      </Field>

      <Field label="Sắp xếp">
        <select value={filters.sort} onChange={(event) => onChange({ ...filters, sort: event.target.value as SearchFilters["sort"] })}>
          <option value="relevance">Liên quan nhất</option>
          <option value="newest">Mới nhất</option>
        </select>
      </Field>

      <label className="toggle-field">
        <input
          type="checkbox"
          checked={filters.rerank}
          onChange={(event) => onChange({ ...filters, rerank: event.target.checked })}
        />
        <span>Dùng reranker</span>
      </label>

      <div className="date-pair">
        <Field label="Từ ngày">
          <input type="date" value={filters.fromDate} onChange={(event) => onChange({ ...filters, fromDate: event.target.value })} />
        </Field>
        <Field label="Đến ngày">
          <input type="date" value={filters.toDate} onChange={(event) => onChange({ ...filters, toDate: event.target.value })} />
        </Field>
      </div>
    </aside>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="field-block">
      <span>{label}</span>
      {children}
    </label>
  );
}
