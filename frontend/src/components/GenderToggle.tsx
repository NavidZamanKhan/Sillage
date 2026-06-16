"use client";

import React from "react";

interface GenderToggleProps {
  gender: "men" | "women";
  onGenderChange: (gender: "men" | "women") => void;
}

export default function GenderToggle({
  gender,
  onGenderChange,
}: GenderToggleProps) {
  const pillClass =
    gender === "men"
      ? "gender-toggle__pill gender-toggle__pill--boys"
      : "gender-toggle__pill gender-toggle__pill--girls";

  return (
    <fieldset className="flex justify-center">
      <legend className="sr-only">Gender filter</legend>
      <div
        id="gender-shell"
        className="gender-toggle"
        role="group"
        aria-label="Gender toggle"
      >
        {/* Sliding pill */}
        <div className={pillClass} aria-hidden="true" />

        <button
          type="button"
          data-gender="men"
          className={`gender-toggle__btn ${
            gender === "men" ? "gender-toggle__btn--active" : ""
          }`}
          aria-pressed={gender === "men"}
          onClick={() => onGenderChange("men")}
        >
          Boys
        </button>
        <button
          type="button"
          data-gender="women"
          className={`gender-toggle__btn ${
            gender === "women" ? "gender-toggle__btn--active" : ""
          }`}
          aria-pressed={gender === "women"}
          onClick={() => onGenderChange("women")}
        >
          Girls
        </button>
      </div>
    </fieldset>
  );
}
