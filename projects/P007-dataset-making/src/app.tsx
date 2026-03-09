import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";

const AnnotatorPage = lazy(() =>
  import("@/routes/annotator-page").then((module) => ({
    default: module.AnnotatorPage
  }))
);

export function App() {
  return (
    <div className="min-h-screen bg-[var(--background)] text-[var(--foreground)] antialiased">
      <Suspense
        fallback={
          <div className="flex min-h-screen items-center justify-center">
            <div className="rounded-2xl border border-[color:var(--border)] bg-[var(--card)] px-6 py-4 text-sm text-[var(--muted-foreground)] shadow-sm">
              Preparing the review desk
            </div>
          </div>
        }
      >
        <Routes>
          <Route element={<AnnotatorPage />} path="/" />
        </Routes>
      </Suspense>
    </div>
  );
}
