"use client";

import React from "react";

interface SearchBarProps {
  query: string;
  onQueryChange: (value: string) => void;
  onSearch: () => void;
}

export default function SearchBar({
  query,
  onQueryChange,
  onSearch,
}: SearchBarProps) {
  return (
    <div className="search-capsule glass-reflection rounded-full">
      <div className="relative flex items-center py-1 pl-5 pr-2">
        <span
          className="pointer-events-none flex shrink-0 text-stone-400"
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
        <input
          id="search-input"
          type="search"
          autoComplete="off"
          placeholder="Search by vibe or perfume name…"
          className="min-w-0 flex-1 border-0 bg-transparent py-3.5 pl-3 pr-3 text-base text-stone-800 placeholder:text-stone-400 outline-none ring-0"
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
        <button
          type="button"
          id="search-btn"
          className="search-theme-btn shrink-0 rounded-full px-5 py-2.5 text-sm font-semibold transition cursor-pointer"
          onClick={onSearch}
        >
          Search
        </button>
      </div>
    </div>
  );
}
