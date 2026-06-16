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
 * Cards are grouped in sets of 8 with a structured, repeating bento pattern.
 */
function getBentoLayout(globalIndex: number): {
  variant: "large" | "wide" | "standard" | "tall";
  className: string;
  style?: React.CSSProperties;
} {
  const cycle = Math.floor(globalIndex / 8);
  const localIndex = globalIndex % 8;

  const VARIANTS: Record<number, "large" | "wide" | "standard" | "tall"> = {
    0: "large",
    1: "standard",
    2: "standard",
    3: "standard",
    4: "wide",
    5: "standard",
    6: "standard",
    7: "tall",
  };

  const variant = VARIANTS[localIndex];
  const groupClass = cycle === 0 ? `bento-item-g0-${localIndex}` : `bento-item-g1-${localIndex}`;
  const sizeClass = `bento-item-${variant}`;

  const res: {
    variant: "large" | "wide" | "standard" | "tall";
    className: string;
    style?: React.CSSProperties;
  } = {
    variant,
    className: `${groupClass} ${sizeClass}`,
  };

  if (cycle >= 1) {
    res.style = {
      "--base-row": 5 + 4 * (cycle - 1),
    } as React.CSSProperties;
  }

  return res;
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
            <div
              key={`${r.perfume_name}-${r.brand}-${idx}`}
              className={layout.className}
              style={layout.style}
            >
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
