import { useState, useEffect, useMemo } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  LineChart,
  Line,
  BarChart,
  Bar,
  Cell,
  PieChart,
  Pie,
} from "recharts";

// ─── Shared helpers ────────────────────────────

function dayToLabel(day: number, matYear: number): string {
  const base = new Date(matYear - 1, 8, 1);
  base.setDate(base.getDate() + day);
  return base.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

const COLORS = {
  accepted: "#34d399",
  waitlisted: "#fbbf24",
  rejected: "#fb7185",
  you: "#60a5fa",
};

// ─── #1: LSAT/GPA Scatter Plot ─────────────────

interface ScatterPoint {
  lsat: number;
  gpa: number;
  result: string;
  matriculating_year: number;
}

export function LsatGpaScatter({
  schoolName,
  userLsat,
  userGpa,
  year,
}: {
  schoolName: string;
  userLsat: number;
  userGpa: number;
  year?: number;
}) {
  const [points, setPoints] = useState<ScatterPoint[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!schoolName) return;
    setLoading(true);
    const params = year ? `?year=${year}` : "";
    fetch(`/api/viz/scatter/${encodeURIComponent(schoolName)}${params}`)
      .then((r) => r.json())
      .then((d) => setPoints(d.points || []))
      .finally(() => setLoading(false));
  }, [schoolName, year]);

  const { accepted, waitlisted, rejected } = useMemo(() => {
    const a: ScatterPoint[] = [];
    const w: ScatterPoint[] = [];
    const r: ScatterPoint[] = [];
    for (const p of points) {
      if (p.result === "accepted") a.push(p);
      else if (p.result === "waitlisted") w.push(p);
      else r.push(p);
    }
    return { accepted: a, waitlisted: w, rejected: r };
  }, [points]);

  if (!schoolName) return null;

  return (
    <div className="rounded-2xl border border-slate-700/50 bg-slate-800/50 p-6 backdrop-blur-sm">
      <h2 className="mb-1 text-lg font-semibold text-slate-200">
        Where Do You Stand?
      </h2>
      <p className="mb-4 text-xs text-slate-500">
        Your stats (blue) vs. historical applicants{year ? ` (${year} cycle)` : ""}
        {" · "}{points.length} data points
      </p>
      {loading ? (
        <div className="flex h-64 items-center justify-center text-sm text-slate-500">Loading...</div>
      ) : (
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" strokeOpacity={0.5} />
              <XAxis
                dataKey="lsat"
                type="number"
                domain={["dataMin - 2", "dataMax + 2"]}
                name="LSAT"
                tick={{ fontSize: 10, fill: "#94a3b8" }}
                stroke="#475569"
                label={{ value: "LSAT", position: "bottom", fill: "#94a3b8", fontSize: 11 }}
              />
              <YAxis
                dataKey="gpa"
                type="number"
                domain={[2.0, 4.3]}
                name="GPA"
                tick={{ fontSize: 10, fill: "#94a3b8" }}
                stroke="#475569"
                width={35}
                label={{ value: "GPA", angle: -90, position: "insideLeft", fill: "#94a3b8", fontSize: 11 }}
              />
              <Tooltip
                content={({ payload }) => {
                  if (!payload?.length) return null;
                  const d = payload[0].payload;
                  return (
                    <div className="rounded border border-slate-600 bg-slate-900 px-2 py-1 text-xs">
                      LSAT: {d.lsat} · GPA: {d.gpa?.toFixed(2)}
                      {d.result && <span className="ml-1 capitalize"> · {d.result}</span>}
                    </div>
                  );
                }}
              />
              <Scatter name="Rejected" data={rejected} fill={COLORS.rejected} fillOpacity={0.35} r={3} />
              <Scatter name="Waitlisted" data={waitlisted} fill={COLORS.waitlisted} fillOpacity={0.4} r={3} />
              <Scatter name="Accepted" data={accepted} fill={COLORS.accepted} fillOpacity={0.5} r={3} />
              <Scatter
                name="You"
                data={[{ lsat: userLsat, gpa: userGpa }]}
                fill={COLORS.you}
                r={7}
                strokeWidth={2}
                stroke="#fff"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
      <div className="mt-2 flex justify-center gap-5 text-xs text-slate-400">
        <span className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-400" /> Accepted</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 rounded-full bg-amber-400" /> Waitlisted</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 rounded-full bg-rose-400" /> Rejected</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 rounded-full border-2 border-white bg-blue-400" /> You</span>
      </div>
    </div>
  );
}

// ─── #3: Cycle-over-Cycle Median Drift ─────────

interface DriftYear {
  matriculating_year: number;
  median_lsat: number;
  median_gpa: number;
  p25_lsat: number;
  p75_lsat: number;
  p25_gpa: number;
  p75_gpa: number;
  count: number;
}

export function MedianDrift({ schoolName }: { schoolName: string }) {
  const [data, setData] = useState<DriftYear[]>([]);
  const [metric, setMetric] = useState<"lsat" | "gpa">("lsat");

  useEffect(() => {
    if (!schoolName) return;
    fetch(`/api/viz/median_drift/${encodeURIComponent(schoolName)}`)
      .then((r) => r.json())
      .then((d) => setData(d.yearly || []));
  }, [schoolName]);

  if (!schoolName || data.length === 0) return null;

  const medianKey = metric === "lsat" ? "median_lsat" : "median_gpa";
  const p25Key = metric === "lsat" ? "p25_lsat" : "p25_gpa";
  const p75Key = metric === "lsat" ? "p75_lsat" : "p75_gpa";

  // Add band data
  const chartData = data.map((d) => ({
    ...d,
    band: [(d as unknown as Record<string, number>)[p25Key], (d as unknown as Record<string, number>)[p75Key]],
  }));

  return (
    <div className="rounded-2xl border border-slate-700/50 bg-slate-800/50 p-6 backdrop-blur-sm">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-200">Median Drift</h2>
          <p className="text-xs text-slate-500">How accepted {metric.toUpperCase()} has shifted over time</p>
        </div>
        <div className="flex gap-1 rounded-lg bg-slate-700/50 p-0.5">
          {(["lsat", "gpa"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMetric(m)}
              className={`rounded-md px-3 py-1 text-xs font-medium transition-all ${
                metric === m ? "bg-indigo-600 text-white" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {m.toUpperCase()}
            </button>
          ))}
        </div>
      </div>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" strokeOpacity={0.5} />
            <XAxis
              dataKey="matriculating_year"
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              stroke="#475569"
            />
            <YAxis
              domain={metric === "lsat" ? ["dataMin - 2", "dataMax + 2"] : ["dataMin - 0.1", "dataMax + 0.1"]}
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              stroke="#475569"
              width={40}
              tickFormatter={(v: number) => metric === "gpa" ? v.toFixed(1) : String(v)}
            />
            <Tooltip
              content={({ payload }) => {
                if (!payload?.length) return null;
                const d = payload[0].payload as DriftYear;
                return (
                  <div className="rounded border border-slate-600 bg-slate-900 px-3 py-2 text-xs">
                    <div className="font-medium text-slate-300">{d.matriculating_year} cycle</div>
                    <div className="text-indigo-400">
                      Median: {metric === "lsat" ? d.median_lsat : d.median_gpa?.toFixed(2)}
                    </div>
                    <div className="text-slate-500">
                      25th-75th: {metric === "lsat" ? `${d.p25_lsat}–${d.p75_lsat}` : `${d.p25_gpa?.toFixed(2)}–${d.p75_gpa?.toFixed(2)}`}
                    </div>
                    <div className="text-slate-600">{d.count} accepted</div>
                  </div>
                );
              }}
            />
            <Line
              type="monotone"
              dataKey={p25Key}
              stroke="#6366f1"
              strokeOpacity={0.2}
              strokeDasharray="4 4"
              dot={false}
            />
            <Line
              type="monotone"
              dataKey={p75Key}
              stroke="#6366f1"
              strokeOpacity={0.2}
              strokeDasharray="4 4"
              dot={false}
            />
            <Line
              type="monotone"
              dataKey={medianKey}
              stroke="#818cf8"
              strokeWidth={2.5}
              dot={{ fill: "#818cf8", r: 3 }}
              activeDot={{ r: 5, fill: "#a5b4fc" }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ─── #4: Wave Calendar Heatmap ─────────────────

interface WeekBucket {
  day_start: number;
  total: number;
  accepted: number;
  waitlisted: number;
  rejected: number;
}

export function WaveHeatmap({
  schoolName,
  matYear,
}: {
  schoolName: string;
  matYear: number;
}) {
  const [weeks, setWeeks] = useState<WeekBucket[]>([]);

  useEffect(() => {
    if (!schoolName) return;
    fetch(`/api/viz/wave_heatmap/${encodeURIComponent(schoolName)}`)
      .then((r) => r.json())
      .then((d) => setWeeks(d.weeks || []));
  }, [schoolName]);

  if (!schoolName || weeks.length === 0) return null;

  const maxTotal = Math.max(...weeks.map((w) => w.total), 1);

  return (
    <div className="rounded-2xl border border-slate-700/50 bg-slate-800/50 p-6 backdrop-blur-sm">
      <h2 className="mb-1 text-lg font-semibold text-slate-200">Decision Wave Calendar</h2>
      <p className="mb-4 text-xs text-slate-500">
        Weekly decision volume, color = dominant outcome
      </p>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={weeks} margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" strokeOpacity={0.3} />
            <XAxis
              dataKey="day_start"
              tickFormatter={(d: number) => dayToLabel(d, matYear)}
              tick={{ fontSize: 9, fill: "#94a3b8" }}
              stroke="#475569"
              interval={3}
            />
            <YAxis
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              stroke="#475569"
              width={30}
            />
            <Tooltip
              content={({ payload }) => {
                if (!payload?.length) return null;
                const d = payload[0].payload as WeekBucket;
                return (
                  <div className="rounded border border-slate-600 bg-slate-900 px-3 py-2 text-xs">
                    <div className="font-medium text-slate-300">
                      {dayToLabel(d.day_start, matYear)} – {dayToLabel(d.day_start + 6, matYear)}
                    </div>
                    <div className="text-slate-400">{d.total} decisions</div>
                    <div className="text-emerald-400">{d.accepted} accepted</div>
                    <div className="text-amber-400">{d.waitlisted} waitlisted</div>
                    <div className="text-rose-400">{d.rejected} rejected</div>
                  </div>
                );
              }}
            />
            <Bar dataKey="total" radius={[2, 2, 0, 0]}>
              {weeks.map((w, i) => {
                const dominant =
                  w.accepted >= w.waitlisted && w.accepted >= w.rejected
                    ? COLORS.accepted
                    : w.waitlisted >= w.rejected
                    ? COLORS.waitlisted
                    : COLORS.rejected;
                const opacity = 0.3 + 0.7 * (w.total / maxTotal);
                return <Cell key={i} fill={dominant} fillOpacity={opacity} />;
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 flex justify-center gap-5 text-xs text-slate-400">
        <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-4 rounded-sm bg-emerald-400" /> Accept-heavy</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-4 rounded-sm bg-amber-400" /> WL-heavy</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-4 rounded-sm bg-rose-400" /> Reject-heavy</span>
      </div>
    </div>
  );
}

// ─── #5: Applicants Like You ───────────────────

interface SimilarData {
  total: number;
  accepted: number;
  waitlisted: number;
  rejected: number;
}

export function ApplicantsLikeYou({
  schoolName,
  lsat,
  gpa,
}: {
  schoolName: string;
  lsat: number;
  gpa: number;
}) {
  const [data, setData] = useState<SimilarData | null>(null);

  useEffect(() => {
    if (!schoolName) return;
    fetch(
      `/api/viz/similar_applicants/${encodeURIComponent(schoolName)}?lsat=${lsat}&gpa=${gpa}`
    )
      .then((r) => r.json())
      .then((d) => setData(d));
  }, [schoolName, lsat, gpa]);

  if (!schoolName || !data || data.total === 0) return null;

  const pieData = [
    { name: "Accepted", value: data.accepted, fill: COLORS.accepted },
    { name: "Waitlisted", value: data.waitlisted, fill: COLORS.waitlisted },
    { name: "Rejected", value: data.rejected, fill: COLORS.rejected },
  ];

  return (
    <div className="rounded-2xl border border-slate-700/50 bg-slate-800/50 p-6 backdrop-blur-sm">
      <h2 className="mb-1 text-lg font-semibold text-slate-200">
        Applicants Like You
      </h2>
      <p className="mb-3 text-xs text-slate-500">
        Historical outcomes for LSAT {lsat}±2 / GPA {gpa.toFixed(1)}±0.1 · {data.total} applicants
      </p>
      <div className="flex items-center gap-4">
        <div className="h-36 w-36 flex-shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={30}
                outerRadius={55}
                paddingAngle={3}
                dataKey="value"
                strokeWidth={0}
              >
                {pieData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Pie>
              <Tooltip
                content={({ payload }) => {
                  if (!payload?.length) return null;
                  const d = payload[0];
                  return (
                    <div className="rounded border border-slate-600 bg-slate-900 px-2 py-1 text-xs">
                      {d.name}: {d.value} ({data.total > 0 ? Math.round(((d.value as number) / data.total) * 100) : 0}%)
                    </div>
                  );
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="flex-1 space-y-2">
          {pieData.map((entry) => {
            const pct = data.total > 0 ? Math.round((entry.value / data.total) * 100) : 0;
            return (
              <div key={entry.name} className="flex items-center gap-2">
                <div className="h-2 flex-1 rounded-full bg-slate-700/50 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{ width: `${pct}%`, backgroundColor: entry.fill }}
                  />
                </div>
                <span className="w-20 text-right text-xs font-medium" style={{ color: entry.fill }}>
                  {entry.name} {pct}%
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ─── #6: Days Waiting Distribution ─────────────

interface WaitBucket {
  bucket: number;
  result: string;
  count: number;
}

export function WaitTimeDistribution({ schoolName }: { schoolName: string }) {
  const [raw, setRaw] = useState<WaitBucket[]>([]);

  useEffect(() => {
    if (!schoolName) return;
    fetch(`/api/viz/wait_times/${encodeURIComponent(schoolName)}`)
      .then((r) => r.json())
      .then((d) => setRaw(d.buckets || []));
  }, [schoolName]);

  // Pivot: each bucket → {bucket, accepted, waitlisted, rejected}
  const chartData = useMemo(() => {
    const map = new Map<number, { bucket: number; accepted: number; waitlisted: number; rejected: number }>();
    for (const r of raw) {
      if (!map.has(r.bucket)) {
        map.set(r.bucket, { bucket: r.bucket, accepted: 0, waitlisted: 0, rejected: 0 });
      }
      const entry = map.get(r.bucket)!;
      if (r.result === "accepted") entry.accepted += r.count;
      else if (r.result === "waitlisted") entry.waitlisted += r.count;
      else entry.rejected += r.count;
    }
    return Array.from(map.values()).sort((a, b) => a.bucket - b.bucket);
  }, [raw]);

  if (!schoolName || chartData.length === 0) return null;

  return (
    <div className="rounded-2xl border border-slate-700/50 bg-slate-800/50 p-6 backdrop-blur-sm">
      <h2 className="mb-1 text-lg font-semibold text-slate-200">
        Days to Decision
      </h2>
      <p className="mb-4 text-xs text-slate-500">
        How long applicants waited from submission to decision
      </p>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" strokeOpacity={0.3} />
            <XAxis
              dataKey="bucket"
              tickFormatter={(d: number) => `${d}d`}
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              stroke="#475569"
            />
            <YAxis
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              stroke="#475569"
              width={35}
            />
            <Tooltip
              content={({ payload }) => {
                if (!payload?.length) return null;
                const d = payload[0].payload;
                return (
                  <div className="rounded border border-slate-600 bg-slate-900 px-3 py-2 text-xs">
                    <div className="font-medium text-slate-300">{d.bucket}–{d.bucket + 13} days</div>
                    <div className="text-emerald-400">Accepted: {d.accepted}</div>
                    <div className="text-amber-400">Waitlisted: {d.waitlisted}</div>
                    <div className="text-rose-400">Rejected: {d.rejected}</div>
                  </div>
                );
              }}
            />
            <Bar dataKey="accepted" stackId="a" fill={COLORS.accepted} fillOpacity={0.8} radius={[0, 0, 0, 0]} />
            <Bar dataKey="waitlisted" stackId="a" fill={COLORS.waitlisted} fillOpacity={0.8} />
            <Bar dataKey="rejected" stackId="a" fill={COLORS.rejected} fillOpacity={0.8} radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 flex justify-center gap-5 text-xs text-slate-400">
        <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-4 rounded-sm bg-emerald-400" /> Accepted</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-4 rounded-sm bg-amber-400" /> Waitlisted</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-4 rounded-sm bg-rose-400" /> Rejected</span>
      </div>
    </div>
  );
}
