"use client";

import React from "react";
import type { PerfumeResult } from "@/lib/api";

function BottleIcon() {
  return (
    <div
      className="flex h-10 w-7 shrink-0 items-center justify-center text-stone-300 group-hover:text-stone-400"
      aria-hidden="true"
    >
      <svg
        className="h-9 w-4"
        viewBox="0 0 16 32"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="M8 2v3"
          stroke="currentColor"
          strokeWidth="1.2"
          strokeLinecap="round"
        />
        <rect
          x="4.5"
          y="5"
          width="7"
          height="24"
          rx="1.5"
          stroke="currentColor"
          strokeWidth="1.2"
          fill="none"
        />
      </svg>
    </div>
  );
}

function deriveScentTags(result: PerfumeResult): string[] {
  const tags: string[] = [];
  const seen = new Set<string>();

  function addTag(tag: string) {
    if (!tag) return;
    const t = tag.trim();
    if (!t) return;
    const key = t.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    tags.push(t);
  }

  function seasonTag(rawSeason: string): string {
    const s = (rawSeason || "").trim().toLowerCase();
    if (!s || s === "nan" || s === "—") return "";
    if (s.includes("winter")) return "Winter";
    if (s.includes("summer")) return "Summer";
    if (s.includes("spring")) return "Spring";
    if (s.includes("fall") || s.includes("autumn")) return "Fall";
    if (s.includes("all")) return "All Season";
    return rawSeason.trim();
  }

  const season = seasonTag(result.season);
  addTag(season);

  const perfumeName = (result.perfume_name || "").toLowerCase();
  const rules: { re: RegExp; tag: string }[] = [
    { re: /\boud\b/, tag: "Oud" },
    { re: /\brose\b/, tag: "Rose" },
    { re: /\bvanilla\b/, tag: "Vanilla" },
    { re: /\b(marine|sea)\b/, tag: "Marine" },
    { re: /\bfresh\b/, tag: "Fresh" },
    { re: /\b(spice|spicy)\b/, tag: "Spicy" },
    { re: /\b(wood|woody|cedar)\b/, tag: "Woody" },
    { re: /\bamber\b/, tag: "Amber" },
    { re: /\bmusk\b/, tag: "Musk" },
    { re: /\b(floral|flower)\b/, tag: "Floral" },
    { re: /\b(citrus|lemon|bergamot|orange)\b/, tag: "Citrus" },
  ];
  rules.forEach((rule) => {
    if (rule.re.test(perfumeName)) addTag(rule.tag);
  });

  const seasonKey = season.toLowerCase();
  let defaults = ["Balanced", "Elegant"];
  if (seasonKey.includes("winter")) defaults = ["Warm", "Cozy"];
  else if (seasonKey.includes("summer")) defaults = ["Fresh", "Airy"];
  else if (seasonKey.includes("spring")) defaults = ["Soft", "Bright"];
  else if (seasonKey.includes("fall")) defaults = ["Rich", "Smooth"];

  defaults.forEach(addTag);
  addTag("Refined");

  return tags.slice(0, 3);
}

interface ResultCardProps {
  result: PerfumeResult;
  isFeatured: boolean;
}

export default function ResultCard({ result, isFeatured }: ResultCardProps) {
  const brand =
    result.brand && String(result.brand) !== "nan" ? result.brand : "—";
  const season = result.season || "—";
  const tags = deriveScentTags(result);

  return (
    <article
      className={`boutique-card group cursor-default rounded-2xl ${
        isFeatured ? "featured-result" : ""
      }`}
    >
      <div className="boutique-card-surface flex gap-4 rounded-2xl p-5 sm:p-[1.35rem]">
        <div className="result-icon-wrap flex shrink-0 items-start pt-0.5 transition-transform duration-300 ease-out">
          <BottleIcon />
        </div>
        <div className="min-w-0 flex-1">
          <a
            href={result.image_search_url || "#"}
            target="_blank"
            rel="noopener noreferrer"
            className="result-name font-sans text-lg font-semibold text-stone-900 transition-colors duration-300 hover:text-stone-700 break-words decoration-stone-300 underline-offset-4 hover:underline"
          >
            {result.perfume_name || "Unknown"}
          </a>
          <p className="mt-1 text-sm text-stone-500">
            {brand} · {season}
          </p>
          <p className="mt-2 truncate text-[0.72rem] tracking-[0.02em] text-stone-400">
            {tags.join(" · ")}
          </p>
          {result.similarity != null && (
            <p className="mt-2 text-xs tracking-wide text-stone-400">
              Match · {Number(result.similarity).toFixed(3)}
            </p>
          )}
        </div>
      </div>
    </article>
  );
}
