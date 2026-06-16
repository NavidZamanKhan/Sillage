"use client";

import React from "react";
import type { PerfumeResult } from "@/lib/api";
import type { CardColor } from "@/lib/scentUtils";
import {
  generateScentSummary,
  generateVibeChips,
  formatMatchScore,
} from "@/lib/scentUtils";

interface ResultCardProps {
  result: PerfumeResult;
  variant: "large" | "wide" | "small";
  color: CardColor;
  animationDelay?: number;
}

export default function ResultCard({
  result,
  variant,
  color,
  animationDelay = 0,
}: ResultCardProps) {
  const brand =
    result.brand && String(result.brand) !== "nan" ? result.brand : "";
  const score = formatMatchScore(result);
  const summary = generateScentSummary(result);
  const chips = generateVibeChips(result);

  const sizeClass = `rec-card--${variant}`;
  const colorClass = `card-color-${color}`;

  return (
    <article
      className={`rec-card ${sizeClass} ${colorClass} fade-in-up`}
      style={{ animationDelay: `${animationDelay}ms` }}
    >
      {/* Top row: brand + badge */}
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          {brand && (
            <p className="rec-card__brand">{brand}</p>
          )}
          <h3 className="rec-card__name">
            {result.perfume_name || "Unknown"}
          </h3>
        </div>
        <div className="match-badge">
          <span className="match-badge__score">{score}%</span>
          <span className="match-badge__label">Match</span>
        </div>
      </div>

      {/* Summary */}
      <p className="rec-card__summary">{summary}</p>

      {/* Vibe chips */}
      <div className="vibe-chips">
        {chips.map((chip) => (
          <span key={chip} className="vibe-chip">
            {chip}
          </span>
        ))}
      </div>
    </article>
  );
}
