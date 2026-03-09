"use client";

import { ChevronRight, Menu, Mic2, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { cn } from "@/lib/utils";

const NAV_LINKS = [
  { label: "My Tasks", href: "/my-tasks" },
  { label: "Apply for Work", href: "/apply" },
  { label: "Task Statistics", href: "/statistics" },
  { label: "Personal Info", href: "/personal" },
] as const;

export function Navbar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 bg-[var(--nav-bg)] text-[var(--nav-foreground)] shadow-md">
      <div className="mx-auto flex h-12 max-w-screen-2xl items-center justify-between px-4">
        {/* Brand */}
        <Link
          className="flex shrink-0 items-center gap-2 transition-opacity hover:opacity-90"
          href="/"
        >
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary">
            <Mic2 className="h-4 w-4 text-primary-foreground" />
          </div>
          <span className="font-semibold text-[var(--nav-foreground)] text-sm tracking-wide">
            SpeechLab
          </span>
          <span className="ml-1 hidden font-mono text-[var(--nav-foreground)]/40 text-xs sm:inline">
            uTrans
          </span>
        </Link>

        {/* Desktop nav */}
        <nav className="hidden items-center gap-1 md:flex">
          {NAV_LINKS.map(({ label, href }) => (
            <Link
              className={cn(
                "rounded-md px-3 py-1.5 font-medium text-xs transition-colors",
                pathname === href
                  ? "bg-white/15 text-[var(--nav-foreground)]"
                  : "text-[var(--nav-foreground)]/60 hover:bg-white/10 hover:text-[var(--nav-foreground)]"
              )}
              href={href}
              key={href}
            >
              {label}
            </Link>
          ))}
        </nav>

        {/* Right side */}
        <div className="flex items-center gap-3">
          <div className="hidden items-center gap-1.5 rounded-md bg-white/10 px-2.5 py-1 sm:flex">
            <div className="h-1.5 w-1.5 animate-pulse rounded-full bg-primary" />
            <span className="font-mono text-[var(--nav-foreground)]/80 text-xs">
              ID: 10109
            </span>
          </div>
          <button
            aria-label="Toggle menu"
            className="rounded-md p-1.5 text-[var(--nav-foreground)] transition-colors hover:bg-white/10 md:hidden"
            onClick={() => setMobileOpen(!mobileOpen)}
            type="button"
          >
            {mobileOpen ? (
              <X className="h-4 w-4" />
            ) : (
              <Menu className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {mobileOpen && (
        <div className="border-white/10 border-t md:hidden">
          {NAV_LINKS.map(({ label, href }) => (
            <Link
              className={cn(
                "flex w-full items-center justify-between border-white/5 border-b px-4 py-3 text-sm transition-colors",
                pathname === href
                  ? "bg-white/15 text-[var(--nav-foreground)]"
                  : "text-[var(--nav-foreground)]/80 hover:bg-white/10"
              )}
              href={href}
              key={href}
              onClick={() => setMobileOpen(false)}
            >
              {label}
              <ChevronRight className="h-4 w-4 opacity-50" />
            </Link>
          ))}
        </div>
      )}
    </header>
  );
}
