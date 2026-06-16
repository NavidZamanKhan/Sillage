/**
 * Scent summary, vibe chip, and match score utilities for Sillage 2.0.
 *
 * These helpers generate human-friendly card content from the API result fields.
 * The API does NOT return notes/accords directly, so we derive everything from:
 *   - perfume_name, brand, season
 *   - matched_note_terms, matched_vibe_terms, matched_season_terms
 *   - similarity, final_score
 */

import type { PerfumeResult } from "./api";

/* ------------------------------------------------------------------ */
/* Match score                                                        */
/* ------------------------------------------------------------------ */

/**
 * Convert similarity / final_score to a 0-100 percentage for display.
 * Uses `final_score` when available (it's typically > similarity due to
 * reranking bonuses), capped at 99.
 */
export function formatMatchScore(result: PerfumeResult): number {
  const raw = result.final_score ?? result.similarity ?? 0;
  // The reranking engine can push scores above 1.0 via bonuses.
  // We normalise into [0, 99] for display.
  const pct = Math.round(Math.min(raw, 1.0) * 100);
  return Math.max(1, Math.min(pct, 99));
}

/* ------------------------------------------------------------------ */
/* Vibe / accord chips (3 per card)                                   */
/* ------------------------------------------------------------------ */

const NAME_NOTE_RULES: { re: RegExp; tag: string }[] = [
  { re: /\boud\b/i, tag: "Oud" },
  { re: /\brose\b/i, tag: "Rose" },
  { re: /\bvanilla\b/i, tag: "Vanilla" },
  { re: /\b(marine|sea|aqua)\b/i, tag: "Aquatic" },
  { re: /\bfresh\b/i, tag: "Fresh" },
  { re: /\b(spice|spicy)\b/i, tag: "Spicy" },
  { re: /\b(wood|woody|cedar|sandalwood)\b/i, tag: "Woody" },
  { re: /\bamber\b/i, tag: "Amber" },
  { re: /\bmusk\b/i, tag: "Musk" },
  { re: /\b(floral|flower|jasmine|lily)\b/i, tag: "Floral" },
  { re: /\b(citrus|lemon|bergamot|orange|lime)\b/i, tag: "Citrus" },
  { re: /\b(leather)\b/i, tag: "Leather" },
  { re: /\b(tobacco)\b/i, tag: "Tobacco" },
  { re: /\b(saffron)\b/i, tag: "Saffron" },
  { re: /\b(incense)\b/i, tag: "Incense" },
  { re: /\b(lavender)\b/i, tag: "Aromatic" },
  { re: /\b(coconut|tropical)\b/i, tag: "Tropical" },
  { re: /\b(powder|powdery)\b/i, tag: "Powdery" },
];

const SEASON_VIBES: Record<string, string[]> = {
  winter: ["Warm", "Cozy"],
  summer: ["Fresh", "Bright"],
  spring: ["Soft", "Bright"],
  fall: ["Rich", "Smooth"],
  autumn: ["Rich", "Smooth"],
};

function capitalize(s: string): string {
  if (!s) return s;
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/**
 * Generate exactly 3 short vibe chips from the result data.
 */
export function generateVibeChips(result: PerfumeResult): string[] {
  const seen = new Set<string>();
  const chips: string[] = [];

  const push = (tag: string) => {
    const t = tag.trim();
    if (!t || seen.has(t.toLowerCase())) return;
    seen.add(t.toLowerCase());
    chips.push(t);
  };

  // 1. From matched terms (most relevant)
  if (result.matched_note_terms) {
    for (const t of result.matched_note_terms) push(capitalize(t));
  }
  if (result.matched_vibe_terms) {
    for (const t of result.matched_vibe_terms) push(capitalize(t));
  }
  if (result.matched_season_terms) {
    for (const t of result.matched_season_terms) push(capitalize(t));
  }

  // 2. From perfume name heuristics
  const name = result.perfume_name || "";
  for (const rule of NAME_NOTE_RULES) {
    if (chips.length >= 3) break;
    if (rule.re.test(name)) push(rule.tag);
  }

  // 3. Season-derived vibes
  const season = (result.season || "").toLowerCase().trim();
  if (season && season !== "nan" && season !== "—") {
    for (const [key, vibes] of Object.entries(SEASON_VIBES)) {
      if (season.includes(key)) {
        for (const v of vibes) push(v);
      }
    }
  }

  // 4. Generic fallbacks
  const fallbacks = ["Elegant", "Refined", "Modern", "Balanced"];
  for (const f of fallbacks) {
    if (chips.length >= 3) break;
    push(f);
  }

  return chips.slice(0, 3);
}

/* ------------------------------------------------------------------ */
/* Scent summary generation                                           */
/* ------------------------------------------------------------------ */

/** Helper to detect a note/vibe in the perfume name. */
function nameHas(name: string, ...terms: string[]): boolean {
  const n = name.toLowerCase();
  return terms.some((t) => n.includes(t.toLowerCase()));
}

/**
 * Templates for summary generation. Each returns a string or null (skip).
 */
type SummaryTemplate = (ctx: SummaryContext) => string | null;

interface SummaryContext {
  name: string;
  brand: string;
  season: string;
  noteTerms: string[];
  vibeTerms: string[];
  seasonTerms: string[];
  chips: string[];
}

const ADJECTIVES_BY_VIBE: Record<string, string[]> = {
  fresh: ["bright", "crisp", "clean"],
  dark: ["deep", "mysterious", "intense"],
  warm: ["warm", "enveloping", "cozy"],
  sweet: ["luscious", "sweet", "indulgent"],
  clean: ["pristine", "clean", "airy"],
  sexy: ["sensual", "magnetic", "alluring"],
  fruity: ["juicy", "vibrant", "playful"],
  elegant: ["refined", "polished", "graceful"],
  luxury: ["opulent", "luxurious", "distinguished"],
};

const ADJECTIVES_BY_NOTE: Record<string, string> = {
  oud: "rich oud",
  rose: "romantic rose",
  vanilla: "smooth vanilla",
  amber: "warm amber",
  musk: "clean musky",
  citrus: "bright citrus",
  leather: "bold leather",
  tobacco: "smoky tobacco",
  saffron: "radiant saffron",
  woody: "woody",
  fresh: "airy",
  floral: "delicate floral",
  spicy: "spiced",
  incense: "resinous incense",
  sandalwood: "warm sandalwood",
};

const CLOSINGS = [
  "a clean musky dry down",
  "polished freshness",
  "soft vanilla warmth",
  "relaxed summer energy",
  "a smooth finish",
  "quiet confidence",
  "lingering depth",
  "an elegant trail",
  "gentle warmth",
  "a whisper of musk",
];

const TEMPLATES: SummaryTemplate[] = [
  // Template 1: "A [adj] [note] fragrance with [note2] and [closing]."
  (ctx) => {
    if (ctx.noteTerms.length >= 2) {
      const adj = ADJECTIVES_BY_NOTE[ctx.noteTerms[0]] || capitalize(ctx.noteTerms[0]);
      const second = ADJECTIVES_BY_NOTE[ctx.noteTerms[1]] || ctx.noteTerms[1];
      const closing = CLOSINGS[Math.abs(ctx.name.length) % CLOSINGS.length];
      return `A ${adj} fragrance with ${second} warmth and ${closing}.`;
    }
    return null;
  },

  // Template 2: "A [vibe] [note] scent with [season] energy."
  (ctx) => {
    if (ctx.noteTerms.length >= 1 && ctx.vibeTerms.length >= 1) {
      const vibeAdjs = ADJECTIVES_BY_VIBE[ctx.vibeTerms[0]] || ["refined"];
      const adj = vibeAdjs[Math.abs(ctx.name.length) % vibeAdjs.length];
      const note = ADJECTIVES_BY_NOTE[ctx.noteTerms[0]] || ctx.noteTerms[0];
      const closing = ctx.seasonTerms.length > 0
        ? `${ctx.seasonTerms[0]} energy`
        : "polished freshness";
      return `A ${adj} ${note} scent with ${closing}.`;
    }
    return null;
  },

  // Template 3: use chips
  (ctx) => {
    if (ctx.chips.length >= 3) {
      const adj = ctx.chips[0].toLowerCase();
      const note = ctx.chips[1].toLowerCase();
      const closing = CLOSINGS[(ctx.name.length + 3) % CLOSINGS.length];
      return `A ${adj} ${note} fragrance wrapped in ${closing}.`;
    }
    return null;
  },

  // Template 4: season-based
  (ctx) => {
    const s = ctx.season.toLowerCase();
    if (s.includes("winter")) {
      return "A warm, enveloping scent with rich depth and cozy sophistication.";
    }
    if (s.includes("summer")) {
      return "A breezy, refreshing scent with clean woods and relaxed summer energy.";
    }
    if (s.includes("spring")) {
      return "A soft, airy fragrance with bright green notes and gentle warmth.";
    }
    if (s.includes("fall") || s.includes("autumn")) {
      return "A smooth, rich fragrance with amber warmth and spiced elegance.";
    }
    return null;
  },

  // Template 5: name heuristics
  (ctx) => {
    if (nameHas(ctx.name, "oud")) {
      return "A smooth rose and oud fragrance wrapped in soft vanilla warmth.";
    }
    if (nameHas(ctx.name, "rose")) {
      return "A romantic rose fragrance with velvety petals and quiet depth.";
    }
    if (nameHas(ctx.name, "vanilla")) {
      return "A luscious vanilla scent with creamy sweetness and amber warmth.";
    }
    if (nameHas(ctx.name, "blue", "aqua", "marine", "sea")) {
      return "A breezy citrus scent with clean woods and relaxed summer energy.";
    }
    if (nameHas(ctx.name, "noir", "dark", "night")) {
      return "A deep, mysterious fragrance with smoky elegance and quiet intensity.";
    }
    return null;
  },
];

/**
 * Generate a short, natural scent summary for a perfume result.
 */
export function generateScentSummary(result: PerfumeResult): string {
  const ctx: SummaryContext = {
    name: result.perfume_name || "",
    brand: result.brand || "",
    season: result.season || "",
    noteTerms: result.matched_note_terms || [],
    vibeTerms: result.matched_vibe_terms || [],
    seasonTerms: result.matched_season_terms || [],
    chips: generateVibeChips(result),
  };

  // Try each template in order; first non-null wins
  for (const template of TEMPLATES) {
    const summary = template(ctx);
    if (summary) return summary;
  }

  // Ultimate fallback
  return "A beautifully composed fragrance with refined character and lasting elegance.";
}

/* ------------------------------------------------------------------ */
/* Card color palette                                                 */
/* ------------------------------------------------------------------ */

export type CardColor =
  | "blue"
  | "sage"
  | "lavender"
  | "peach"
  | "blush"
  | "beige";

const CARD_COLORS: CardColor[] = [
  "blue",
  "sage",
  "lavender",
  "peach",
  "blush",
  "beige",
];

/** Get a card color by global index (cycles through palette). */
export function getCardColor(index: number): CardColor {
  return CARD_COLORS[index % CARD_COLORS.length];
}
