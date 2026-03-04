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
  ReferenceLine,
  Legend,
} from "recharts";

const API_BASE = import.meta.env.VITE_API_BASE || "";

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
    fetch(`${API_BASE}/api/viz/scatter/${encodeURIComponent(schoolName)}${params}`)
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
    <div className="nb-card">
      <h2 className="mb-1 text-lg font-black">
        Where Do You Stand?
      </h2>
      <p className="mb-4 text-xs font-medium text-neutral-700">
        Your stats (blue) vs. historical applicants{year ? ` (${year} cycle)` : ""}
        {" · "}{points.length} data points
      </p>
      {loading ? (
        <div className="flex h-64 items-center justify-center text-sm font-medium text-neutral-700">Loading...</div>
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
                    <div className="border-2 border-black bg-white px-2 py-1 text-xs shadow-[3px_3px_0_0_#000]">
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
      <div className="mt-2 flex justify-center gap-5 text-xs font-medium text-neutral-700">
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
    fetch(`${API_BASE}/api/viz/median_drift/${encodeURIComponent(schoolName)}`)
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
    <div className="nb-card">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-black">Median Drift</h2>
          <p className="text-xs font-medium text-neutral-700">How accepted {metric.toUpperCase()} has shifted over time</p>
        </div>
        <div className="flex gap-1 border-2 border-black bg-white p-0.5 shadow-[3px_3px_0_0_#000]">
          {(["lsat", "gpa"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMetric(m)}
              className={`rounded-md px-3 py-1 text-xs font-medium transition-all ${
                metric === m ? "bg-indigo-600 text-white" : "text-black hover:bg-neutral-100"
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
                  <div className="border-2 border-black bg-white px-3 py-2 text-xs shadow-[3px_3px_0_0_#000]">
                    <div className="font-black text-black">{d.matriculating_year} cycle</div>
                    <div className="font-bold text-indigo-700">
                      Median: {metric === "lsat" ? d.median_lsat : d.median_gpa?.toFixed(2)}
                    </div>
                    <div className="font-medium text-neutral-700">
                      25th-75th: {metric === "lsat" ? `${d.p25_lsat}–${d.p75_lsat}` : `${d.p25_gpa?.toFixed(2)}–${d.p75_gpa?.toFixed(2)}`}
                    </div>
                    <div className="font-medium text-neutral-600">{d.count} accepted</div>
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
    fetch(`${API_BASE}/api/viz/wave_heatmap/${encodeURIComponent(schoolName)}`)
      .then((r) => r.json())
      .then((d) => setWeeks(d.weeks || []));
  }, [schoolName]);

  if (!schoolName || weeks.length === 0) return null;

  const maxTotal = Math.max(...weeks.map((w) => w.total), 1);

  return (
    <div className="nb-card">
      <h2 className="mb-1 text-lg font-black">Decision Wave Calendar</h2>
      <p className="mb-4 text-xs font-medium text-neutral-700">
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
                  <div className="border-2 border-black bg-white px-3 py-2 text-xs shadow-[3px_3px_0_0_#000]">
                    <div className="font-black text-black">
                      {dayToLabel(d.day_start, matYear)} – {dayToLabel(d.day_start + 6, matYear)}
                    </div>
                    <div className="font-medium text-neutral-700">{d.total} decisions</div>
                    <div className="font-bold text-emerald-700">{d.accepted} accepted</div>
                    <div className="font-bold text-amber-700">{d.waitlisted} waitlisted</div>
                    <div className="font-bold text-rose-700">{d.rejected} rejected</div>
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
      <div className="mt-2 flex justify-center gap-5 text-xs font-medium text-neutral-700">
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
      `${API_BASE}/api/viz/similar_applicants/${encodeURIComponent(schoolName)}?lsat=${lsat}&gpa=${gpa}`
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
    <div className="nb-card">
      <h2 className="mb-1 text-lg font-black">
        Applicants Like You
      </h2>
      <p className="mb-3 text-xs font-medium text-neutral-700">
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
                    <div className="border-2 border-black bg-white px-2 py-1 text-xs shadow-[3px_3px_0_0_#000]">
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
                <div className="h-2 flex-1 border-2 border-black bg-white overflow-hidden">
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
    fetch(`${API_BASE}/api/viz/wait_times/${encodeURIComponent(schoolName)}`)
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
    <div className="nb-card">
      <h2 className="mb-1 text-lg font-black">
        Days to Decision
      </h2>
      <p className="mb-4 text-xs font-medium text-neutral-700">
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
                  <div className="border-2 border-black bg-white px-3 py-2 text-xs shadow-[3px_3px_0_0_#000]">
                    <div className="font-black text-black">{d.bucket}–{d.bucket + 13} days</div>
                    <div className="font-bold text-emerald-700">Accepted: {d.accepted}</div>
                    <div className="font-bold text-amber-700">Waitlisted: {d.waitlisted}</div>
                    <div className="font-bold text-rose-700">Rejected: {d.rejected}</div>
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
      <div className="mt-2 flex justify-center gap-5 text-xs font-medium text-neutral-700">
        <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-4 rounded-sm bg-emerald-400" /> Accepted</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-4 rounded-sm bg-amber-400" /> Waitlisted</span>
        <span className="flex items-center gap-1.5"><span className="inline-block h-2 w-4 rounded-sm bg-rose-400" /> Rejected</span>
      </div>
    </div>
  );
}

// ─── #6: Cycle Pace ("Is This Cycle Slow?") ───────────────

interface PaceCurvePoint { day: number; count: number }
interface PaceCycleData { curve: PaceCurvePoint[]; raw_total: number }
interface CyclePaceResponse {
  today_day: number;
  today_date: string;
  cycles: Record<string, PaceCycleData>;
  pct_vs_prior_year: number | null;
  pct_vs_3yr_avg: number | null;
  prior_year: number | null;
  current_count: number | null;
  prior_count: number | null;
  avg_past_count: number | null;
  current_mat_year: number;
  past_mat_years: number[];
  error?: string;
}

const CYCLE_COLORS: Record<string, string> = {
  current: "#4f46e5",
  lag1:    "#f59e0b",
  lag2:    "#10b981",
  lag3:    "#94a3b8",
};

function cyclePaceLabel(day: number): string {
  const base = new Date(2025, 8, 1);
  base.setDate(base.getDate() + day);
  return base.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

export function CyclePace({ schoolList }: { schoolList: string[] }) {
  const [schoolName, setSchoolName] = useState("ALL");
  const [query, setQuery] = useState("");
  const [showDropdown, setShowDropdown] = useState(false);
  const [data, setData] = useState<CyclePaceResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    const param = schoolName === "ALL" ? "" : `?school_name=${encodeURIComponent(schoolName)}`;
    fetch(`${API_BASE}/api/cycle_pace${param}`)
      .then((r) => r.json())
      .then((d: CyclePaceResponse) => setData(d))
      .finally(() => setLoading(false));
  }, [schoolName]);

  const chartData = useMemo(() => {
    if (!data?.cycles) return [];
    const yearKeys = Object.keys(data.cycles).sort();
    const firstCurve = data.cycles[yearKeys[0]]?.curve ?? [];
    return firstCurve.map((pt) => {
      const row: Record<string, number> = { day: pt.day };
      for (const yr of yearKeys) {
        const match = data.cycles[yr].curve.find((p) => p.day === pt.day);
        row[yr] = match ? match.count : 0;
      }
      return row;
    });
  }, [data]);

  const pct = data?.pct_vs_prior_year;
  const isSlower = pct != null && pct < -5;
  const isFaster = pct != null && pct > 5;

  const headlineBg = isSlower ? "bg-rose-100 border-rose-600"
    : isFaster ? "bg-emerald-100 border-emerald-600"
    : "bg-amber-100 border-amber-600";
  const headlineText = isSlower ? "text-rose-800"
    : isFaster ? "text-emerald-800"
    : "text-amber-800";
  const priorYr = data?.prior_year ?? (data?.current_mat_year ?? 2026) - 1;
  const headlineLabel = loading ? "LOADING..."
    : pct == null ? "INSUFFICIENT DATA"
    : isSlower ? `THIS CYCLE IS ${Math.abs(pct).toFixed(1)}% SLOWER THAN ${priorYr}`
    : isFaster ? `THIS CYCLE IS ${Math.abs(pct).toFixed(1)}% FASTER THAN ${priorYr}`
    : `THIS CYCLE IS ON PACE WITH ${priorYr}`;

  const currentYear = data?.current_mat_year ?? 2026;
  const pastYears = (data?.past_mat_years ?? []).sort();
  const yearColorMap: Record<string, string> = {};
  if (pastYears.length >= 3) yearColorMap[String(pastYears[0])] = CYCLE_COLORS.lag3;
  if (pastYears.length >= 2) yearColorMap[String(pastYears[1])] = CYCLE_COLORS.lag2;
  if (pastYears.length >= 1) yearColorMap[String(pastYears[pastYears.length - 1])] = CYCLE_COLORS.lag1;
  yearColorMap[String(currentYear)] = CYCLE_COLORS.current;

  const filteredSchools = useMemo(
    () => ["ALL", ...schoolList].filter((s) => s.toLowerCase().includes(query.toLowerCase())),
    [query, schoolList]
  );

  return (
    <div className="space-y-4">
      {/* School selector */}
      <div className="flex flex-wrap items-center gap-3">
        <span className="nb-label">School</span>
        <div className="relative w-72">
          <input
            className="nb-input text-sm"
            value={showDropdown ? query : schoolName === "ALL" ? "All Schools (Consolidated)" : schoolName}
            placeholder="Search schools..."
            onFocus={() => { setShowDropdown(true); setQuery(""); }}
            onBlur={() => setTimeout(() => setShowDropdown(false), 150)}
            onChange={(e) => setQuery(e.target.value)}
          />
          {showDropdown && (
            <ul className="absolute z-50 mt-0.5 max-h-52 w-full overflow-auto border-2 border-black bg-white shadow-[4px_4px_0_0_#000]">
              {filteredSchools.slice(0, 40).map((s) => (
                <li
                  key={s}
                  className="cursor-pointer px-3 py-1.5 text-xs font-medium hover:bg-indigo-600 hover:text-white"
                  onMouseDown={() => { setSchoolName(s); setQuery(""); setShowDropdown(false); }}
                >
                  {s === "ALL" ? "All Schools (Consolidated)" : s}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {/* Headline verdict */}
      <div className={`border-2 p-5 ${headlineBg}`}>
        <p className={`text-2xl font-black tracking-tight ${headlineText}`}>{headlineLabel}</p>
        {!loading && pct != null && (
          <p className="mt-1.5 text-xs font-medium text-neutral-700">
            As of {data?.today_date} ·{" "}
            {data?.current_count?.toLocaleString()} decisions in {currentYear} cycle vs{" "}
            {data?.prior_count?.toLocaleString()} at same point in {priorYr}
            {data?.pct_vs_3yr_avg != null && (
              <span className="ml-2 text-neutral-500">
                (vs 3-yr avg: {data.pct_vs_3yr_avg > 0 ? "+" : ""}{data.pct_vs_3yr_avg.toFixed(1)}% — inflated by LSD.law growth)
              </span>
            )}
          </p>
        )}
      </div>

      {/* Multi-line chart */}
      {chartData.length > 0 && (
        <div className="nb-card">
          <h3 className="mb-1 text-sm font-black">Cumulative Decisions by Day of Cycle</h3>
          <p className="mb-4 text-xs font-medium text-neutral-600">
            Raw cumulative decision count per cycle. Higher lines in recent years reflect LSD.law platform growth, not necessarily a faster cycle. Dashed vertical = today.
          </p>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 4, right: 12, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="day"
                  tickFormatter={(d: number) => cyclePaceLabel(d)}
                  tick={{ fontSize: 9, fill: "#64748b" }}
                  interval={29}
                  stroke="#cbd5e1"
                />
                <YAxis
                  tickFormatter={(v: number) => v >= 1000 ? `${(v/1000).toFixed(0)}k` : String(v)}
                  tick={{ fontSize: 10, fill: "#64748b" }}
                  stroke="#cbd5e1"
                  width={42}
                  domain={[0, "auto"]}
                />
                <Tooltip
                  content={({ payload, label }) => {
                    if (!payload?.length) return null;
                    return (
                      <div className="border-2 border-black bg-white px-3 py-2 text-xs shadow-[3px_3px_0_0_#000]">
                        <div className="mb-1 font-black">{cyclePaceLabel(label as number)}</div>
                        {payload
                          .slice()
                          .sort((a, b) => (b.value as number) - (a.value as number))
                          .map((p) => (
                            <div key={String(p.dataKey)} style={{ color: p.color }} className="font-bold">
                              {p.dataKey === String(currentYear) ? `${p.dataKey} ★` : p.dataKey}:{" "}
                              {(p.value as number).toLocaleString()} decisions
                            </div>
                          ))}
                      </div>
                    );
                  }}
                />
                {data && (
                  <ReferenceLine
                    x={data.today_day}
                    stroke="#0a0a0a"
                    strokeDasharray="4 3"
                    strokeWidth={2}
                    label={{ value: "Today", position: "insideTopRight", fontSize: 9, fill: "#0a0a0a", fontWeight: 700 }}
                  />
                )}
                {Object.keys(yearColorMap)
                  .sort()
                  .map((yr) => (
                    <Line
                      key={yr}
                      type="monotone"
                      dataKey={yr}
                      stroke={yearColorMap[yr]}
                      strokeWidth={yr === String(currentYear) ? 3 : 1.5}
                      dot={false}
                      strokeDasharray={yr === String(currentYear) ? undefined : "5 3"}
                    />
                  ))}
                <Legend
                  formatter={(value) =>
                    value === String(currentYear) ? `${value} (this cycle)` : String(value)
                  }
                  wrapperStyle={{ fontSize: 11, fontWeight: 700 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
