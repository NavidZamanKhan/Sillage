"use client";

import React from "react";

interface LoadMoreButtonProps {
  remaining: number;
  onClick: () => void;
}

export default function LoadMoreButton({
  remaining,
  onClick,
}: LoadMoreButtonProps) {
  const count = Math.min(remaining, 4);

  return (
    <div className="flex flex-col items-center gap-2 mt-10 mb-6">
      <button
        type="button"
        id="load-more-btn"
        className="load-more-btn"
        onClick={onClick}
      >
        <span aria-hidden="true">+</span>
        Load {count} more
      </button>
      <p className="text-xs" style={{ color: "#8a9a8c" }}>
        {count} more scents in the trail
      </p>
    </div>
  );
}
