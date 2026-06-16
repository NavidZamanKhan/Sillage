/**
 * API layer for Sillage 2.0 frontend.
 * All calls to the Django backend go through this module.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

export interface PerfumeResult {
  index?: number;
  perfume_name: string;
  brand: string;
  gender?: string;
  season: string;
  cosine_distance?: number;
  similarity: number;
  final_score?: number;
  is_vip?: boolean;
  matched_note_terms?: string[];
  matched_vibe_terms?: string[];
  matched_season_terms?: string[];
  luxury_boost_applied?: number;
  image_search_url: string;
}

export interface SearchResponse {
  query: string;
  gender: string | null;
  limit: number;
  results: PerfumeResult[];
  error?: string;
}

export async function searchPerfumes({
  query,
  gender,
  limit,
}: {
  query: string;
  gender?: string | null;
  limit?: number;
}): Promise<SearchResponse> {
  const params = new URLSearchParams();
  params.set("query", query);
  if (gender) params.set("gender", gender);
  if (limit) params.set("limit", String(limit));

  const url = `${API_BASE}/api/search/?${params.toString()}`;

  const res = await fetch(url);
  const data = await res.json();

  if (!res.ok) {
    throw new Error(data.error || "Search failed");
  }

  return data as SearchResponse;
}
