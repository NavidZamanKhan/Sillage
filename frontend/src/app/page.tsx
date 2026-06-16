"use client";

import React, { useState, useCallback, useRef, useEffect } from "react";
import SearchBar from "@/components/SearchBar";
import GenderToggle from "@/components/GenderToggle";
import SuggestionChips from "@/components/SuggestionChips";
import ResultsList from "@/components/ResultsList";
import LoadMoreButton from "@/components/LoadMoreButton";
import { searchPerfumes, type PerfumeResult } from "@/lib/api";

const DEBOUNCE_MS = 320;

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [gender, setGender] = useState<"men" | "women">("men");
  const [results, setResults] = useState<PerfumeResult[]>([]);
  const [limit, setLimit] = useState(4);
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
      setLimit(4);
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        doSearch(value, gender, 4);
      }, DEBOUNCE_MS);
    },
    [gender, doSearch]
  );

  const handleClear = useCallback(() => {
    setQuery("");
    setResults([]);
    setError(null);
    setResponseLimit(0);
    setLimit(4);
  }, []);

  const handleGenderChange = useCallback(
    (g: "men" | "women") => {
      setGender(g);
      setLimit(4);
      doSearch(query, g, 4);
    },
    [query, doSearch]
  );

  const handleSuggestionSelect = useCallback(
    (text: string) => {
      setQuery(text);
      setLimit(4);
      if (debounceRef.current) clearTimeout(debounceRef.current);
      doSearch(text, gender, 4);
    },
    [gender, doSearch]
  );

  const handleLoadMore = useCallback(() => {
    const newLimit = limit + 4;
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

  const showSuggestions = query.trim().length === 0;

  return (
    <div className="mx-auto max-w-4xl px-4 pb-24 pt-10 sm:px-6 sm:pt-14">
      {/* Hero Section */}
      <header className="mb-8 text-center sm:mb-10">
        <h1
          className="logo-sillage text-6xl sm:text-7xl md:text-8xl"
          aria-label="Sillage"
        >
          Sillage
        </h1>
        <p
          className="mt-2 text-sm italic tracking-wide sm:text-base"
          style={{
            color: "#5a8a5a",
            fontFamily: "var(--font-playfair), Georgia, serif",
          }}
        >
          The scent trail continues.
        </p>
        <p
          className="mx-auto mt-4 max-w-lg text-sm leading-relaxed sm:text-base"
          style={{ color: "#5a6b5c" }}
        >
          Describe a mood, a note, a season, or a vibe, and let Sillage
          hand-pick fragrances written just for you.
        </p>
      </header>

      {/* Search + Toggle + Suggestions */}
      <div className="mx-auto max-w-2xl space-y-4 sm:space-y-5">
        <SearchBar
          query={query}
          gender={gender}
          onQueryChange={handleQueryChange}
          onSearch={handleSearch}
          onClear={handleClear}
        />
        <GenderToggle gender={gender} onGenderChange={handleGenderChange} />

        {/* Suggestion chips (visible only when input is empty) */}
        {showSuggestions && (
          <SuggestionChips
            gender={gender}
            onSelect={handleSuggestionSelect}
          />
        )}
      </div>

      {/* Results */}
      <div className="mt-10" aria-live="polite">
        <ResultsList
          results={results}
          query={query}
          error={error}
          loading={loading}
        />
      </div>

      {/* Load More */}
      {canLoadMore && (
        <LoadMoreButton
          remaining={4}
          onClick={handleLoadMore}
        />
      )}
    </div>
  );
}
