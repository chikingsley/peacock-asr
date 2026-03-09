"use client";

import {
  BadgeCheck,
  Bell,
  CheckCircle2,
  Languages,
  Mic2,
  Shield,
  Star,
  User,
} from "lucide-react";
import { useState } from "react";
import { Navbar } from "@/components/navbar";
import {
  DEFAULT_USER_PROFILE,
  LANGUAGES,
  PROFICIENCY_LEVELS,
} from "@/lib/data";
import { cn } from "@/lib/utils";

export default function PersonalInfoPage() {
  const [saved, setSaved] = useState(false);
  const [form, setForm] = useState({ ...DEFAULT_USER_PROFILE });

  const set = (key: string, val: unknown) =>
    setForm((f) => ({ ...f, [key]: val }));

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const toggleSecondary = (lang: string) => {
    set(
      "secondaryLangs",
      form.secondaryLangs.includes(lang)
        ? form.secondaryLangs.filter((l) => l !== lang)
        : [...form.secondaryLangs, lang]
    );
  };

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <Navbar />
      <main className="mx-auto flex w-full max-w-screen-lg flex-1 flex-col gap-6 px-4 py-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="font-bold text-foreground text-xl">
              Personal Information
            </h1>
            <p className="mt-0.5 text-muted-foreground text-sm">
              Manage your account details and preferences
            </p>
          </div>
          <div className="flex items-center gap-2 rounded-xl border border-primary/20 bg-primary/10 px-4 py-2">
            <BadgeCheck className="h-4 w-4 text-primary" />
            <span className="font-semibold text-primary text-xs">
              Verified Annotator
            </span>
            <span className="font-mono text-primary/60 text-xs">·</span>
            <div className="flex items-center gap-0.5">
              {[1, 2, 3, 4, 5].map((s) => (
                <Star
                  className={cn(
                    "h-3 w-3",
                    s <= 4
                      ? "fill-[var(--score-warn)] text-[var(--score-warn)]"
                      : "text-border"
                  )}
                  key={s}
                />
              ))}
            </div>
          </div>
        </div>
        <div className="grid gap-6 md:grid-cols-3">
          <div className="flex flex-col gap-4">
            <div className="flex flex-col items-center gap-3 rounded-xl border border-border bg-card p-5 shadow-sm">
              <div className="flex h-20 w-20 items-center justify-center rounded-full border-4 border-primary/20 bg-primary/10">
                <User className="h-9 w-9 text-primary" />
              </div>
              <div className="text-center">
                <p className="font-bold text-foreground">{form.name}</p>
                <p className="mt-0.5 font-mono text-muted-foreground text-xs">
                  ID: {form.userId}
                </p>
              </div>
              <div className="grid w-full grid-cols-2 gap-2 border-border border-t pt-2">
                {[
                  { label: "Tasks Done", value: "235", icon: CheckCircle2 },
                  { label: "Avg Score", value: "8.4", icon: Mic2 },
                ].map(({ label, value, icon: Icon }) => (
                  <div
                    className="rounded-lg bg-muted/60 p-2.5 text-center"
                    key={label}
                  >
                    <Icon className="mx-auto mb-1 h-3.5 w-3.5 text-muted-foreground" />
                    <p className="font-bold font-mono text-base text-foreground">
                      {value}
                    </p>
                    <p className="text-[10px] text-muted-foreground">{label}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="flex flex-col gap-4 md:col-span-2">
            <div className="overflow-hidden rounded-xl border border-border bg-card shadow-sm">
              <div className="flex items-center gap-2 border-border border-b bg-muted/60 px-4 py-3">
                <User className="h-4 w-4 text-primary" />
                <h2 className="font-semibold text-foreground text-sm">
                  Account Details
                </h2>
              </div>
              <div className="grid gap-4 p-4">
                {[
                  {
                    label: "Full Name",
                    key: "name",
                    type: "text",
                    value: form.name,
                  },
                  {
                    label: "Email",
                    key: "email",
                    type: "email",
                    value: form.email,
                  },
                  {
                    label: "Timezone",
                    key: "timezone",
                    type: "text",
                    value: form.timezone,
                  },
                ].map(({ label, key, type, value }) => (
                  <div className="flex flex-col gap-1.5" key={key}>
                    <label
                      className="font-semibold text-muted-foreground text-xs"
                      htmlFor={`personal-${key}`}
                    >
                      {label}
                    </label>
                    <input
                      className="rounded-lg border border-border bg-background px-3 py-2 font-medium text-foreground text-sm focus:outline-none focus:ring-1 focus:ring-primary"
                      id={`personal-${key}`}
                      onChange={(e) => set(key, e.target.value)}
                      type={type}
                      value={value}
                    />
                  </div>
                ))}
              </div>
            </div>
            <div className="overflow-hidden rounded-xl border border-border bg-card shadow-sm">
              <div className="flex items-center gap-2 border-border border-b bg-muted/60 px-4 py-3">
                <Languages className="h-4 w-4 text-primary" />
                <h2 className="font-semibold text-foreground text-sm">
                  Language Skills
                </h2>
              </div>
              <div className="flex flex-col gap-4 p-4">
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="flex flex-col gap-1.5">
                    <label
                      className="font-semibold text-muted-foreground text-xs"
                      htmlFor="personal-primaryLang"
                    >
                      Primary Language
                    </label>
                    <select
                      className="cursor-pointer rounded-lg border border-border bg-background px-3 py-2 font-medium text-foreground text-sm focus:outline-none focus:ring-1 focus:ring-primary"
                      id="personal-primaryLang"
                      onChange={(e) => set("primaryLang", e.target.value)}
                      value={form.primaryLang}
                    >
                      {LANGUAGES.map((l) => (
                        <option key={l}>{l}</option>
                      ))}
                    </select>
                  </div>
                  <div className="flex flex-col gap-1.5">
                    <label
                      className="font-semibold text-muted-foreground text-xs"
                      htmlFor="personal-proficiency"
                    >
                      Proficiency
                    </label>
                    <select
                      className="cursor-pointer rounded-lg border border-border bg-background px-3 py-2 font-medium text-foreground text-sm focus:outline-none focus:ring-1 focus:ring-primary"
                      id="personal-proficiency"
                      onChange={(e) => set("proficiency", e.target.value)}
                      value={form.proficiency}
                    >
                      {PROFICIENCY_LEVELS.map((p) => (
                        <option key={p}>{p}</option>
                      ))}
                    </select>
                  </div>
                </div>
                <div className="flex flex-col gap-2">
                  <span className="font-semibold text-muted-foreground text-xs">
                    Additional Languages
                  </span>
                  <div className="flex flex-wrap gap-2">
                    {LANGUAGES.filter((l) => l !== form.primaryLang).map(
                      (lang) => {
                        const active = form.secondaryLangs.includes(lang);
                        return (
                          <button
                            className={cn(
                              "rounded-full border px-3 py-1.5 font-medium text-xs transition-all",
                              active
                                ? "border-primary bg-primary text-primary-foreground"
                                : "border-border bg-muted text-muted-foreground hover:border-primary/50 hover:text-foreground"
                            )}
                            key={lang}
                            onClick={() => toggleSecondary(lang)}
                            type="button"
                          >
                            {lang}
                          </button>
                        );
                      }
                    )}
                  </div>
                </div>
              </div>
            </div>
            <div className="overflow-hidden rounded-xl border border-border bg-card shadow-sm">
              <div className="flex items-center gap-2 border-border border-b bg-muted/60 px-4 py-3">
                <Bell className="h-4 w-4 text-primary" />
                <h2 className="font-semibold text-foreground text-sm">
                  Notifications
                </h2>
              </div>
              <div className="flex flex-col gap-3 p-4">
                {[
                  {
                    key: "notifyNewTask",
                    label: "New task assigned",
                    val: form.notifyNewTask,
                  },
                  {
                    key: "notifyScore",
                    label: "Score review completed",
                    val: form.notifyScore,
                  },
                  {
                    key: "notifyPayment",
                    label: "Payment processed",
                    val: form.notifyPayment,
                  },
                ].map(({ key, label, val }) => (
                  <label
                    className="flex cursor-pointer items-center justify-between gap-3 py-1"
                    key={key}
                  >
                    <span className="text-foreground text-sm">{label}</span>
                    <button
                      aria-checked={val}
                      className={cn(
                        "relative h-5 w-10 rounded-full transition-colors",
                        val ? "bg-primary" : "bg-border"
                      )}
                      onClick={() => set(key, !val)}
                      role="switch"
                      type="button"
                    >
                      <span
                        className={cn(
                          "absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform",
                          val ? "translate-x-5" : "translate-x-0"
                        )}
                      />
                    </button>
                  </label>
                ))}
              </div>
            </div>
            <div className="flex justify-end">
              <button
                className={cn(
                  "flex items-center gap-2 rounded-lg px-6 py-2.5 font-semibold text-sm shadow transition-all",
                  saved
                    ? "scale-95 bg-[var(--score-good)] text-white"
                    : "bg-primary text-primary-foreground hover:opacity-90"
                )}
                onClick={handleSave}
                type="button"
              >
                {saved ? (
                  <>
                    <CheckCircle2 className="h-4 w-4" />
                    Saved
                  </>
                ) : (
                  <>
                    <Shield className="h-4 w-4" />
                    Save Changes
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
