"use client";

import {
  FileAudio,
  Pause,
  Play,
  RotateCcw,
  Volume2,
  VolumeX,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

// Deterministic fake waveform bars seeded from an index
function generateFakeWaveform(count: number): number[] {
  const bars: number[] = [];
  for (let i = 0; i < count; i++) {
    const x = i / count;
    const v =
      0.15 +
      0.7 *
        Math.abs(
          Math.sin(x * 31.4) * 0.4 +
            Math.sin(x * 17.8 + 1.2) * 0.3 +
            Math.sin(x * 9.1 + 2.5) * 0.2 +
            Math.sin(x * 54.3 + 0.8) * 0.1
        );
    bars.push(Number.parseFloat(Math.min(1, Math.max(0.06, v)).toFixed(4)));
  }
  return bars;
}

const BAR_COUNT = 160;

interface WaveformPlayerProps {
  audioSize: string;
  filename: string;
  totalLength: string;
}

export function WaveformPlayer({
  audioSize,
  totalLength,
}: WaveformPlayerProps) {
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0.36); // start at 36%
  const [speed, setSpeed] = useState("0.5");
  const [volume, setVolume] = useState(0.8);
  const [muted, setMuted] = useState(false);
  const [hoveredBar, setHoveredBar] = useState<number | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const barsData = useRef(() => {
    const heights = generateFakeWaveform(BAR_COUNT);
    return heights.map((h, idx) => ({ h, id: `bar-${idx.toString()}` }));
  }).current;
  const bars = useRef(barsData()).current;

  const speeds = ["0.5", "0.75", "1.0", "1.25", "1.5", "2.0"];

  // Fake playback animation
  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(() => {
        setProgress((p) => {
          if (p >= 1) {
            setPlaying(false);
            return 0;
          }
          return p + 0.002 * Number.parseFloat(speed);
        });
      }, 50);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [playing, speed]);

  const handleWaveformClick = useCallback((idx: number) => {
    setProgress(idx / BAR_COUNT);
  }, []);

  const handleReset = () => {
    setProgress(0);
    setPlaying(false);
  };

  const activeBar = Math.floor(progress * BAR_COUNT);

  return (
    <div className="overflow-hidden rounded-xl border border-white/5 bg-[var(--waveform-bg)] shadow-lg">
      {/* Canvas area */}
      <div
        aria-label="Audio waveform"
        aria-valuemax={100}
        aria-valuemin={0}
        aria-valuenow={Math.round(progress * 100)}
        className="relative h-32 cursor-crosshair select-none"
        role="slider"
        tabIndex={0}
      >
        {/* Time ruler */}
        <div className="pointer-events-none absolute top-0 right-0 left-0 flex justify-between px-3 pt-1.5">
          {[0, 1, 2, 3].map((t) => (
            <span className="font-mono text-[9px] text-white/25" key={t}>
              {t}
            </span>
          ))}
        </div>

        {/* Waveform bars — event delegation instead of 480 individual handlers */}
        {/* biome-ignore lint/a11y/noStaticElementInteractions: event delegation for waveform bars */}
        {/* biome-ignore lint/a11y/noNoninteractiveElementInteractions: event delegation for waveform bars */}
        <div
          className="absolute inset-0 flex items-center gap-px px-3 pt-5 pb-2"
          onClick={(e) => {
            const idx = (e.target as HTMLElement).dataset.bar;
            if (idx) {
              handleWaveformClick(Number.parseInt(idx, 10));
            }
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              const idx = (e.target as HTMLElement).dataset.bar;
              if (idx) {
                handleWaveformClick(Number.parseInt(idx, 10));
              }
            }
          }}
          onMouseLeave={() => setHoveredBar(null)}
          onMouseMove={(e) => {
            const idx = (e.target as HTMLElement).dataset.bar;
            setHoveredBar(idx != null ? Number.parseInt(idx, 10) : null);
          }}
        >
          {bars.map((bar, i) => {
            const isActive = i <= activeBar;
            const isHovered = hoveredBar !== null && i <= hoveredBar;
            let barColor: string;
            if (isActive) {
              barColor = "var(--waveform-bar)";
            } else if (isHovered) {
              barColor = "rgba(88,196,220,0.35)";
            } else {
              barColor = "rgba(255,255,255,0.12)";
            }
            return (
              <div
                className="flex-1 rounded-sm transition-colors duration-75"
                data-bar={i}
                key={bar.id}
                style={{
                  height: `${(bar.h * 100).toFixed(2)}%`,
                  backgroundColor: barColor,
                }}
              />
            );
          })}
        </div>

        {/* Playhead */}
        <div
          className="pointer-events-none absolute top-0 bottom-0 w-px bg-primary/80 shadow-[0_0_6px_1px_var(--waveform-bar)]"
          style={{ left: `calc(${progress * 100}% )` }}
        />
      </div>

      {/* Controls bar */}
      <div className="flex flex-wrap items-center gap-3 border-white/5 border-t bg-[var(--waveform-bg)] px-4 py-2.5">
        {/* File info */}
        <div className="flex shrink-0 items-center gap-1.5 font-mono text-white/40 text-xs">
          <FileAudio className="h-3.5 w-3.5" />
          <span>{audioSize}</span>
          <span className="text-white/20">·</span>
          <span>{totalLength}</span>
        </div>

        <div className="flex-1" />

        {/* Speed selector */}
        <div className="flex items-center gap-1.5">
          <span className="font-mono text-[10px] text-white/40 uppercase tracking-wide">
            Speed
          </span>
          <div className="flex items-center gap-0.5 rounded-md bg-white/5 p-0.5">
            {speeds.map((s) => (
              <button
                className={`rounded px-2 py-0.5 font-mono text-[10px] transition-colors ${
                  speed === s
                    ? "bg-primary text-primary-foreground"
                    : "text-white/40 hover:text-white/70"
                }`}
                key={s}
                onClick={() => setSpeed(s)}
                type="button"
              >
                {s}x
              </button>
            ))}
          </div>
        </div>

        {/* Volume */}
        <div className="flex items-center gap-2">
          <button
            aria-label={muted ? "Unmute" : "Mute"}
            className="text-white/40 transition-colors hover:text-white/70"
            onClick={() => setMuted(!muted)}
            type="button"
          >
            {muted ? (
              <VolumeX className="h-3.5 w-3.5" />
            ) : (
              <Volume2 className="h-3.5 w-3.5" />
            )}
          </button>
          <input
            aria-label="Volume"
            className="h-1 w-20 cursor-pointer accent-primary"
            max={1}
            min={0}
            onChange={(e) => {
              setVolume(Number.parseFloat(e.target.value));
              setMuted(false);
            }}
            step={0.01}
            type="range"
            value={muted ? 0 : volume}
          />
        </div>

        {/* Reset */}
        <button
          aria-label="Reset playback"
          className="rounded-md p-1.5 text-white/40 transition-colors hover:bg-white/10 hover:text-white/70"
          onClick={handleReset}
          type="button"
        >
          <RotateCcw className="h-3.5 w-3.5" />
        </button>

        {/* Play/Pause */}
        <button
          aria-label={playing ? "Pause" : "Play"}
          className="flex items-center gap-2 rounded-md bg-primary px-4 py-1.5 font-semibold text-primary-foreground text-xs shadow transition-opacity hover:opacity-90"
          onClick={() => setPlaying(!playing)}
          type="button"
        >
          {playing ? (
            <>
              <Pause className="h-3.5 w-3.5" />
              Pause
            </>
          ) : (
            <>
              <Play className="h-3.5 w-3.5" />
              Play
            </>
          )}
        </button>
      </div>
    </div>
  );
}
