"use client";

import React, { useState, useCallback, useRef, useEffect } from "react";
import SearchBar from "@/components/SearchBar";
import GenderToggle from "@/components/GenderToggle";
import ResultsList from "@/components/ResultsList";
import { searchPerfumes, type PerfumeResult } from "@/lib/api";

const DEBOUNCE_MS = 320;

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [gender, setGender] = useState<"men" | "women">("men");
  const [results, setResults] = useState<PerfumeResult[]>([]);
  const [limit, setLimit] = useState(5);
  const [responseLimit, setResponseLimit] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const doSearch = useCallback(
    async (q: string, g: "men" | "women", l: number) => {
      if (!q.trim()) {
        setResults([]);
        setError(null);
        setResponseLimit(0);
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const genderParam = g === "men" ? "Man" : "Women";
        const data = await searchPerfumes({
          query: q,
          gender: genderParam,
          limit: l,
        });
        setResults(data.results);
        setResponseLimit(data.limit);
      } catch (e) {
        setError(
          e instanceof Error ? e.message : "Something went wrong."
        );
        setResults([]);
        setResponseLimit(0);
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const handleSearch = useCallback(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    doSearch(query, gender, limit);
  }, [query, gender, limit, doSearch]);

  const handleQueryChange = useCallback(
    (value: string) => {
      setQuery(value);
      setLimit(5);
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        doSearch(value, gender, 5);
      }, DEBOUNCE_MS);
    },
    [gender, doSearch]
  );

  const handleGenderChange = useCallback(
    (g: "men" | "women") => {
      setGender(g);
      setLimit(5);
      doSearch(query, g, 5);
    },
    [query, doSearch]
  );

  const handleLoadMore = useCallback(() => {
    const newLimit = limit + 5;
    setLimit(newLimit);
    doSearch(query, gender, newLimit);
  }, [query, gender, limit, doSearch]);

  // Clean up debounce on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const canLoadMore =
    results.length > 0 &&
    results.length === responseLimit &&
    responseLimit < 100;

  return (
    <div className="relative z-50 mx-auto max-w-3xl px-4 pb-24 pt-12 sm:px-6 sm:pt-16">
      <div className="hero-ambient-glow" aria-hidden="true" />

      {/* Header */}
      <header className="relative z-10 mb-10 text-center sm:mb-12">
        <h1
          className="logo-sillage text-8xl sm:text-9xl font-[family-name:var(--font-great-vibes)]"
          aria-label="Sillage"
        >
          <span className="logo-sillage-cap">S</span>illage
        </h1>
        <p className="mx-auto mb-1 mt-4 max-w-md text-sm leading-relaxed text-stone-500 sm:mt-5 sm:text-base">
          Describe your mood, an occasion, or a favorite note. I&apos;ll find
          your perfect scent trail.
        </p>
        <p className="mx-auto mt-2 max-w-md text-xs italic tracking-wide text-stone-400/70">
          The scent trail continues.
        </p>
      </header>

      {/* Search + Toggle */}
      <div className="relative z-10 mx-auto max-w-2xl space-y-5 sm:space-y-6">
        <SearchBar
          query={query}
          onQueryChange={handleQueryChange}
          onSearch={handleSearch}
        />
        <GenderToggle gender={gender} onGenderChange={handleGenderChange} />
      </div>

      {/* Status */}
      {(loading || error) && (
        <div
          className={`mt-10 text-center text-sm ${
            error ? "text-red-600/90" : "text-stone-500"
          }`}
          role="status"
          aria-live="polite"
        >
          {loading ? "Searching…" : error}
        </div>
      )}

      {/* Results */}
      <div className="mt-10" aria-live="polite">
        <ResultsList results={results} query={query} />
      </div>

      {/* Load More */}
      {canLoadMore && (
        <div className="mt-12 flex justify-center">
          <button
            type="button"
            id="load-more-btn"
            className="rounded-full border border-white/60 bg-white/40 px-10 py-3 text-sm font-semibold text-stone-700 shadow-[0_8px_30px_rgb(0,0,0,0.04)] backdrop-blur-2xl transition hover:-translate-y-0.5 hover:bg-white/55 hover:shadow-[0_12px_36px_rgb(0,0,0,0.06)] cursor-pointer"
            onClick={handleLoadMore}
          >
            Load 5 More
          </button>
        </div>
      )}
    </div>
  );
}
