"use client";

import React, { useRef, useEffect, useCallback } from "react";
import gsap from "gsap";

interface GenderToggleProps {
  gender: "men" | "women";
  onGenderChange: (gender: "men" | "women") => void;
}

export default function GenderToggle({
  gender,
  onGenderChange,
}: GenderToggleProps) {
  const shellRef = useRef<HTMLDivElement>(null);
  const pillRef = useRef<HTMLDivElement>(null);

  const layoutPill = useCallback(
    (animate: boolean) => {
      const shell = shellRef.current;
      const pill = pillRef.current;
      if (!shell || !pill) return;

      const pad = 4;
      const w = shell.offsetWidth;
      const inner = w - pad * 2;
      const seg = inner / 2;
      const idx = gender === "women" ? 1 : 0;
      const x = pad + idx * seg;
      const h = shell.offsetHeight - pad * 2;

      const props = {
        x,
        width: seg,
        height: h,
        top: pad,
        borderRadius: 9999,
      };

      if (animate) {
        gsap.to(pill, { ...props, duration: 0.38, ease: "power2.inOut" });
      } else {
        gsap.set(pill, props);
      }
    },
    [gender]
  );

  // Initial layout — must wait for first paint so offsetWidth is correct
  const mountedRef = useRef(false);
  useEffect(() => {
    requestAnimationFrame(() => {
      layoutPill(false);
      document.body.setAttribute("data-theme", gender);
      mountedRef.current = true;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Animate on gender change (skip the initial mount)
  useEffect(() => {
    if (!mountedRef.current) return;
    layoutPill(true);
    document.body.setAttribute("data-theme", gender);
  }, [gender, layoutPill]);

  useEffect(() => {
    const onResize = () => layoutPill(false);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [layoutPill]);


  return (
    <fieldset className="flex justify-center">
      <legend className="sr-only">Gender filter</legend>
      <div
        ref={shellRef}
        id="gender-shell"
        className="toggle-shell-glass glass-reflection relative flex h-11 w-full max-w-[220px] rounded-full p-1 sm:max-w-[240px]"
        role="group"
        aria-label="Gender toggle"
      >
        <div
          ref={pillRef}
          id="gender-pill"
          className="toggle-pill-liquid pointer-events-none absolute left-0 top-1 rounded-full"
          aria-hidden="true"
        />
        <button
          type="button"
          data-gender="men"
          className={`relative z-10 flex-1 basis-0 rounded-full py-2 text-sm font-medium transition hover:text-stone-800 cursor-pointer ${
            gender === "men"
              ? "text-stone-900 font-semibold"
              : "text-stone-500"
          }`}
          aria-pressed={gender === "men"}
          onClick={() => onGenderChange("men")}
        >
          Men
        </button>
        <button
          type="button"
          data-gender="women"
          className={`relative z-10 flex-1 basis-0 rounded-full py-2 text-sm font-medium transition hover:text-stone-800 cursor-pointer ${
            gender === "women"
              ? "text-stone-900 font-semibold"
              : "text-stone-500"
          }`}
          aria-pressed={gender === "women"}
          onClick={() => onGenderChange("women")}
        >
          Women
        </button>
      </div>
    </fieldset>
  );
}
