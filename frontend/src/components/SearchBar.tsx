"use client";

import React from "react";

interface SearchBarProps {
  query: string;
  gender: "men" | "women";
  onQueryChange: (value: string) => void;
  onSearch: () => void;
  onClear: () => void;
}

export default function SearchBar({
  query,
  gender,
  onQueryChange,
  onSearch,
  onClear,
}: SearchBarProps) {
  const btnClass =
    gender === "men"
      ? "search-bar__btn search-bar__btn--boys"
      : "search-bar__btn search-bar__btn--girls";

  return (
    <div className="search-bar">
      {/* Search icon */}
      <span
        className="pointer-events-none flex shrink-0"
        style={{ color: "#8a9a8c" }}
        aria-hidden="true"
      >
        <svg
          className="h-5 w-5"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.75"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
          />
        </svg>
      </span>

      {/* Input */}
      <input
        id="search-input"
        type="text"
        autoComplete="off"
        placeholder="Search by mood, note, season, or perfume name..."
        className="search-bar__input"
        aria-label="Search perfumes"
        value={query}
        onChange={(e) => onQueryChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            onSearch();
          }
        }}
      />

      {/* Clear button */}
      {query.length > 0 && (
        <button
          type="button"
          className="search-bar__clear"
          onClick={onClear}
          aria-label="Clear search"
        >
          ×
        </button>
      )}

      {/* Search button */}
      <button
        type="button"
        id="search-btn"
        className={btnClass}
        onClick={onSearch}
      >
        Search
      </button>
    </div>
  );
}
