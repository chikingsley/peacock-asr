"use client";

import { Award, TrendingUp } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { MONTHLY_SCORES, WEEKLY_ACTIVITY } from "@/lib/data";

const PRIMARY = "oklch(0.58 0.18 225)";
const GRID_LINE = "oklch(0.88 0.01 247)";
const TOOLTIP_STYLE = {
  fontSize: 12,
  borderRadius: 8,
  border: "1px solid oklch(0.88 0.01 247)",
  background: "oklch(1 0 0)",
  color: "oklch(0.16 0.015 247)",
};

export function StatisticsCharts() {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {/* Tasks per day */}
      <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
        <div className="mb-4 flex items-center gap-2">
          <TrendingUp className="h-4 w-4 text-primary" />
          <h2 className="font-semibold text-foreground text-sm">
            Tasks This Week
          </h2>
        </div>
        <ResponsiveContainer height={180} width="100%">
          <BarChart barCategoryGap="30%" data={WEEKLY_ACTIVITY}>
            <XAxis
              axisLine={false}
              dataKey="day"
              tick={{ fontSize: 11 }}
              tickLine={false}
            />
            <YAxis
              axisLine={false}
              tick={{ fontSize: 11 }}
              tickLine={false}
              width={28}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              cursor={{ fill: "oklch(0.93 0.008 247)" }}
            />
            <Bar dataKey="tasks" fill={PRIMARY} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Avg score trend */}
      <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
        <div className="mb-4 flex items-center gap-2">
          <Award className="h-4 w-4 text-primary" />
          <h2 className="font-semibold text-foreground text-sm">
            Quality Score Trend
          </h2>
        </div>
        <ResponsiveContainer height={180} width="100%">
          <LineChart data={MONTHLY_SCORES}>
            <CartesianGrid stroke={GRID_LINE} strokeDasharray="3 3" />
            <XAxis
              axisLine={false}
              dataKey="week"
              tick={{ fontSize: 11 }}
              tickLine={false}
            />
            <YAxis
              axisLine={false}
              domain={[6, 10]}
              tick={{ fontSize: 11 }}
              tickLine={false}
              width={28}
            />
            <Tooltip contentStyle={TOOLTIP_STYLE} />
            <Line
              dataKey="avg"
              dot={{ r: 3, fill: PRIMARY }}
              stroke={PRIMARY}
              strokeWidth={2}
              type="monotone"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
