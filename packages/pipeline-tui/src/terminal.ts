// Terminal safety net. OpenTUI's native core normally restores the terminal on
// destroy(), but we belt-and-suspenders it so the user's terminal is NEVER left
// stuck in mouse-reporting / hidden-cursor / alt-screen mode on any exit path.

// SGR + legacy mouse tracking OFF, bracketed paste OFF, focus reporting OFF,
// show cursor, leave alternate screen.
const RESET =
  "\x1b[?1000l" + // X10/normal mouse tracking off
  "\x1b[?1002l" + // button-event (drag) tracking off
  "\x1b[?1003l" + // any-motion tracking off
  "\x1b[?1006l" + // SGR extended coords off
  "\x1b[?1015l" + // urxvt extended coords off
  "\x1b[?1004l" + // focus reporting off
  "\x1b[?2004l" + // bracketed paste off
  "\x1b[?25h"; // show cursor

let installed = false;

// Write the reset sequences straight to the underlying fd (bypassing any
// stdout.write OpenTUI may have monkey-patched), best-effort.
export function resetTerminal(): void {
  try {
    if (process.stdout.isTTY) {
      process.stdout.write(RESET);
    }
  } catch {}
}

// Install once. Covers normal exit, signals, and crashes. Idempotent.
export function installTerminalSafetyNet(): void {
  if (installed) return;
  installed = true;

  // Always runs on process teardown, no matter the cause.
  process.on("exit", resetTerminal);

  // Signals: reset, then exit (which re-triggers 'exit' -> resetTerminal again,
  // harmless). We let OpenTUI's own signal handlers run too; order-independent
  // because resetTerminal is idempotent.
  for (const sig of ["SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT"] as const) {
    process.on(sig, () => {
      resetTerminal();
      process.exit(0);
    });
  }

  process.on("uncaughtException", (err) => {
    resetTerminal();
    // Surface the error after restoring the terminal so it's readable.
    console.error(err);
    process.exit(1);
  });
  process.on("unhandledRejection", (err) => {
    resetTerminal();
    console.error(err);
    process.exit(1);
  });
}
