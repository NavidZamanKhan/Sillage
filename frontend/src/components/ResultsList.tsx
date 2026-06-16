"use client";

import React from "react";
import type { PerfumeResult } from "@/lib/api";
import { getCardColor } from "@/lib/scentUtils";
import ResultCard from "./ResultCard";

interface ResultsListProps {
  results: PerfumeResult[];
  query: string;
  error: string | null;
  loading: boolean;
}

/**
 * Determine the card variant and bento CSS class for a given index.
 * Cards are grouped in sets of 4 with alternating layout patterns.
 */
function getBentoLayout(globalIndex: number): {
  variant: "large" | "wide" | "small";
  className: string;
} {
  const groupIndex = globalIndex % 8; // 2 groups of 4 = 8 pattern cycle

  const LAYOUT: Record<number, { variant: "large" | "wide" | "small"; className: string }> = {
    // Group A (first 4)
    0: { variant: "large", className: "bento-item-0" },
    1: { variant: "wide", className: "bento-item-1" },
    2: { variant: "small", className: "bento-item-2" },
    3: { variant: "small", className: "bento-item-3" },
    // Group B (next 4, mirrored)
    4: { variant: "wide", className: "bento-item-4" },
    5: { variant: "large", className: "bento-item-5" },
    6: { variant: "small", className: "bento-item-6" },
    7: { variant: "small", className: "bento-item-7" },
  };

  return LAYOUT[groupIndex];
}

export default function ResultsList({
  results,
  query,
  error,
  loading,
}: ResultsListProps) {
  // Error state
  if (error) {
    return (
      <p className="error-state">
        Could not reach the scent trail. Check the backend connection.
      </p>
    );
  }

  // Loading state
  if (loading) {
    return (
      <p className="empty-state">Searching…</p>
    );
  }

  // No query yet
  if (!query.trim()) {
    return null;
  }

  // No results
  if (results.length === 0) {
    return (
      <p className="empty-state">
        No matches yet — try another mood, note, or name.
      </p>
    );
  }

  return (
    <>
      {/* Section heading */}
      <div className="mb-8 mt-2">
        <h2 className="section-heading">Curated for you</h2>
        <p className="section-subtitle">
          Scent stories selected to match your mood and notes.
        </p>
      </div>

      {/* Bento grid */}
      <div className="bento-grid" aria-live="polite">
        {results.map((r, idx) => {
          const layout = getBentoLayout(idx);
          return (
            <div key={`${r.perfume_name}-${r.brand}-${idx}`} className={layout.className}>
              <ResultCard
                result={r}
                variant={layout.variant}
                color={getCardColor(idx)}
                animationDelay={idx * 80}
              />
            </div>
          );
        })}
      </div>
    </>
  );
}
