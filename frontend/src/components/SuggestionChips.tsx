"use client";

import React from "react";

const BOYS_SUGGESTIONS = [
  "fresh citrus",
  "dark oud winter",
  "clean office scent",
  "woody evening",
];

const GIRLS_SUGGESTIONS = [
  "soft vanilla",
  "rose date night",
  "clean floral",
  "sweet powdery musk",
];

interface SuggestionChipsProps {
  gender: "men" | "women";
  onSelect: (text: string) => void;
}

export default function SuggestionChips({
  gender,
  onSelect,
}: SuggestionChipsProps) {
  const suggestions = gender === "men" ? BOYS_SUGGESTIONS : GIRLS_SUGGESTIONS;

  return (
    <div className="suggestion-chips" role="group" aria-label="Suggested searches">
      {suggestions.map((text) => (
        <button
          key={text}
          type="button"
          className="suggestion-chip"
          onClick={() => onSelect(text)}
        >
          {text}
        </button>
      ))}
    </div>
  );
}
