import React from "react";
import { collectOnce, startPolling } from "./data";
import { App } from "./App";

const once = process.argv.includes("--once");

if (once) {
  // Headless single-frame render: run all reads once, render one frame, exit.
  const { createTestRenderer } = await import("@opentui/core/testing");
  const { createRoot, flushSync } = await import("@opentui/react");

  const snapshot = await collectOnce();
  // Width: --width=N flag > COLUMNS env > tty width > default. Lets us test the
  // responsive layout headlessly (e.g. COLUMNS=50 bun run once).
  const widthFlag = process.argv.find((a) => a.startsWith("--width="));
  const cols = widthFlag
    ? parseInt(widthFlag.split("=")[1], 10)
    : Math.min(parseInt(process.env.COLUMNS ?? "", 10) || process.stdout.columns || 124, 130);

  const { renderer, flush, captureCharFrame } = await createTestRenderer({
    width: cols,
    height: 48,
  });

  const root = createRoot(renderer);
  flushSync(() => root.render(<App snapshot={snapshot} />));
  await flush(); // settle React commit + flex layout

  const frame = captureCharFrame().replace(/(\n\s*)+$/g, "\n");
  process.stdout.write(frame);
  process.exit(0);
} else {
  // Live: background pollers fill the cache; UI ticks on its own fast cadence.
  const { installTerminalSafetyNet, resetTerminal } = await import("./terminal");
  installTerminalSafetyNet(); // restore terminal on any exit path (signals/crash)

  startPolling();
  const { createCliRenderer } = await import("@opentui/core");
  const { createRoot } = await import("@opentui/react");

  // This dashboard is keyboard-only. OpenTUI defaults useMouse and
  // enableMouseMovement to TRUE, which enables SGR mouse tracking
  // (\e[?1000h/1002h/1003h/1006h) and leaks raw position escapes to the screen.
  // Disable both so mouse tracking is never turned on in the first place.
  const renderer = await createCliRenderer({
    targetFps: 30,
    useMouse: false,
    enableMouseMovement: false,
    // Extra safety net inside OpenTUI's own teardown.
    onDestroy: () => resetTerminal(),
  });

  createRoot(renderer).render(<App />);
}
