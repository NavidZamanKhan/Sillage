"use client";

import React, { useEffect, useRef } from "react";
import gsap from "gsap";
import type { PerfumeResult } from "@/lib/api";
import ResultCard from "./ResultCard";

interface ResultsListProps {
  results: PerfumeResult[];
  query: string;
}

export default function ResultsList({ results, query }: ResultsListProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const prevLengthRef = useRef(0);

  useEffect(() => {
    if (!containerRef.current) return;
    const cards = containerRef.current.querySelectorAll(".boutique-card");
    if (cards.length > 0 && cards.length !== prevLengthRef.current) {
      gsap.from(cards, {
        y: 34,
        opacity: 0,
        duration: 0.52,
        stagger: 0.07,
        ease: "power3.out",
      });
    }
    prevLengthRef.current = cards.length;
  }, [results]);

  if (!query.trim()) {
    return null;
  }

  if (results.length === 0) {
    return (
      <p className="text-center text-stone-500 py-14">
        No matches yet — try another vibe or name.
      </p>
    );
  }

  return (
    <div ref={containerRef} className="space-y-4">
      {results.map((r, idx) => (
        <ResultCard
          key={`${r.perfume_name}-${r.brand}-${idx}`}
          result={r}
          isFeatured={idx === 0}
        />
      ))}
    </div>
  );
}
