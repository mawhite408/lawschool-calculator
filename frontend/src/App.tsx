import { useState, useEffect, useRef, useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  CartesianGrid,
} from "recharts";
import {
  LsatGpaScatter,
  MedianDrift,
  ApplicantsLikeYou,
  WaitTimeDistribution,
  CyclePace,
} from "./Charts";

const API_BASE = import.meta.env.VITE_API_BASE || "";

interface WaveEntry {
  start: number;
  end: number;
  center: number;
  count: number;
  accepted: number;
  waitlisted: number;
  rejected: number;
}

interface PredictionResult {
  probabilities: { accepted: number; waitlisted: number; rejected: number };
  school_context: {
    school_median_lsat: number;
    school_median_gpa: number;
    school_25_lsat: number;
    school_75_lsat: number;
    school_25_gpa: number;
    school_75_gpa: number;
    school_accept_rate: number;
    school_count: number;
  };
  wave_info: {
    total_waves: number;
    waves_passed: number;
    waves_remaining: number;
    passed_waves: WaveEntry[];
    upcoming_waves: WaveEntry[];
  } | null;
}

const INPUT_CLS =
  "nb-input";


function ProbabilityBar({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm font-bold">
        <span className="text-black">{label}</span>
        <span className={color}>{value.toFixed(1)}%</span>
      </div>
      <div className="h-3 w-full border-2 border-black bg-white overflow-hidden">
        <div
          className={`h-full transition-all duration-700 ease-out ${color.replace("text-", "bg-")}`}
          style={{ width: `${Math.max(value, 0.5)}%` }}
        />
      </div>
    </div>
  );
}

function SchoolCombobox({
  schools,
  value,
  onChange,
}: {
  schools: string[];
  value: string;
  onChange: (v: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const ref = useRef<HTMLDivElement>(null);

  const filtered = useMemo(() => {
    if (!search) return schools;
    const q = search.toLowerCase();
    return schools.filter((s) => s.toLowerCase().includes(q));
  }, [schools, search]);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  return (
    <div ref={ref} className="relative">
      <input
        type="text"
        className={INPUT_CLS}
        placeholder="Search schools..."
        value={open ? search : value}
        onFocus={() => {
          setOpen(true);
          setSearch("");
        }}
        onChange={(e) => setSearch(e.target.value)}
      />
      {open && (
        <ul className="absolute z-50 mt-1 max-h-60 w-full overflow-auto border-2 border-black bg-white py-1 shadow-[6px_6px_0_0_#000]">
          {filtered.length === 0 ? (
            <li className="px-3 py-2 text-sm font-medium text-neutral-600">
              No schools found
            </li>
          ) : (
            filtered.map((s) => (
              <li
                key={s}
                className={`cursor-pointer px-3 py-1.5 text-sm font-medium hover:bg-indigo-600 hover:text-white ${s === value ? "bg-indigo-600 text-white" : "text-black"}`}
                onClick={() => {
                  onChange(s);
                  setOpen(false);
                  setSearch("");
                }}
              >
                {s}
              </li>
            ))
          )}
        </ul>
      )}
    </div>
  );
}

interface TimelinePoint {
  day: number;
  date: string;
  accepted: number;
  waitlisted: number;
  rejected: number;
  confidence: number;
}

interface TimelineData {
  timeline: TimelinePoint[];
  actual_current_day: number | null;
  wave_markers: { day: number; date: string }[];
  cycle_start: string;
}

function formatDayLabel(day: number, matYear: number): string {
  const base = new Date(matYear - 1, 8, 1);
  base.setDate(base.getDate() + day);
  return base.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function AdmissionTimeline({
  data,
  matYear,
}: {
  data: TimelineData;
  matYear: number;
}) {
  const currentDay = data.actual_current_day;

  // Confidence threshold: below 5% of decisions, predictions are low-confidence
  const CONF_THRESHOLD = 0.05;

  // Find the first day where confidence crosses threshold
  const confCrossDay = useMemo(() => {
    for (const p of data.timeline) {
      if (p.confidence >= CONF_THRESHOLD) return p.day;
    }
    return data.timeline[data.timeline.length - 1]?.day ?? 0;
  }, [data.timeline]);

  // Build chart data: split into low-confidence, past, and future series
  const chartData = data.timeline.map((p) => {
    const isLowConf = p.confidence < CONF_THRESHOLD;
    const isPast = currentDay !== null && p.day <= currentDay;
    const isFuture = currentDay === null || p.day >= currentDay;

    return {
      ...p,
      label: formatDayLabel(p.day, matYear),
      // Low-confidence region (very faded)
      acceptedLow: isLowConf ? p.accepted : undefined,
      waitlistedLow: isLowConf ? p.waitlisted : undefined,
      rejectedLow: isLowConf ? p.rejected : undefined,
      // Past region (faded, but confident)
      acceptedPast: !isLowConf && isPast ? p.accepted : undefined,
      waitlistedPast: !isLowConf && isPast ? p.waitlisted : undefined,
      rejectedPast: !isLowConf && isPast ? p.rejected : undefined,
      // Future region (full opacity, confident)
      acceptedFuture: !isLowConf && isFuture ? p.accepted : undefined,
      waitlistedFuture: !isLowConf && isFuture ? p.waitlisted : undefined,
      rejectedFuture: !isLowConf && isFuture ? p.rejected : undefined,
    };
  });

  // Bridge points: duplicate the crossover point into both series so lines connect
  // Find the index closest to confCrossDay
  const crossIdx = chartData.findIndex((p) => p.day >= confCrossDay);
  if (crossIdx > 0 && chartData[crossIdx]) {
    const p = chartData[crossIdx];
    // Also set the low-conf value at the cross point so lines meet
    p.acceptedLow = p.accepted;
    p.waitlistedLow = p.waitlisted;
    p.rejectedLow = p.rejected;
  }

  const CustomTooltip = ({ active, payload, label: _label }: { active?: boolean; payload?: Array<{ payload: TimelinePoint }>; label?: string }) => {
    if (!active || !payload || !payload.length) return null;
    const p = payload[0].payload;
    const confPct = Math.round(p.confidence * 100);
    const isLow = p.confidence < CONF_THRESHOLD;
    return (
      <div className="border-2 border-black bg-white px-3 py-2 text-xs shadow-[4px_4px_0_0_#000]">
        <div className="mb-1 flex items-center gap-2 font-bold text-black">
          {formatDayLabel(p.day, matYear)}
          {isLow && (
            <span className="border border-black bg-amber-200 px-1.5 py-0.5 text-[10px] font-bold text-black">
              Low Data
            </span>
          )}
        </div>
        <div className="text-emerald-700">Accepted: {p.accepted}%</div>
        <div className="text-amber-700">Waitlisted: {p.waitlisted}%</div>
        <div className="text-rose-700">Rejected: {p.rejected}%</div>
        <div className="mt-1 font-medium text-neutral-700">{confPct}% of decisions made by this point</div>
      </div>
    );
  };

  // X-axis ticks: every ~30 days
  const ticks = data.timeline
    .filter((_, i) => i % 10 === 0)
    .map((p) => p.day);

  return (
    <div className="nb-card">
      <h2 className="mb-1 text-lg font-black">
        Odds Across the Cycle
      </h2>
      <p className="mb-4 text-xs font-medium text-neutral-700">
        Probability of each outcome assuming no decision at each point
      </p>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
            <defs>
              <linearGradient id="gradAccept" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#34d399" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="gradWL" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#fbbf24" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#fbbf24" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="gradReject" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#fb7185" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#fb7185" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="gradAcceptPast" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#34d399" stopOpacity={0.1} />
                <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="gradWLPast" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#fbbf24" stopOpacity={0.1} />
                <stop offset="95%" stopColor="#fbbf24" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="gradRejectPast" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#fb7185" stopOpacity={0.1} />
                <stop offset="95%" stopColor="#fb7185" stopOpacity={0} />
              </linearGradient>
              {/* Very low opacity for low-confidence region */}
              <linearGradient id="gradLow" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#94a3b8" stopOpacity={0.05} />
                <stop offset="95%" stopColor="#94a3b8" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" strokeOpacity={0.5} />
            <XAxis
              dataKey="day"
              ticks={ticks}
              tickFormatter={(d: number) => formatDayLabel(d, matYear)}
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              stroke="#475569"
            />
            <YAxis
              domain={[0, 100]}
              tickFormatter={(v: number) => `${v}%`}
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              stroke="#475569"
              width={40}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Low-confidence region (very faded, dashed) */}
            <Area type="monotone" dataKey="acceptedLow" stroke="#34d399" strokeOpacity={0.15} strokeWidth={1} strokeDasharray="4 3" fill="url(#gradLow)" connectNulls={false} dot={false} />
            <Area type="monotone" dataKey="waitlistedLow" stroke="#fbbf24" strokeOpacity={0.15} strokeWidth={1} strokeDasharray="4 3" fill="url(#gradLow)" connectNulls={false} dot={false} />
            <Area type="monotone" dataKey="rejectedLow" stroke="#fb7185" strokeOpacity={0.15} strokeWidth={1} strokeDasharray="4 3" fill="url(#gradLow)" connectNulls={false} dot={false} />

            {/* Past (faded but confident) areas */}
            <Area type="monotone" dataKey="acceptedPast" stroke="#34d399" strokeOpacity={0.3} strokeWidth={1} fill="url(#gradAcceptPast)" connectNulls={false} dot={false} />
            <Area type="monotone" dataKey="waitlistedPast" stroke="#fbbf24" strokeOpacity={0.3} strokeWidth={1} fill="url(#gradWLPast)" connectNulls={false} dot={false} />
            <Area type="monotone" dataKey="rejectedPast" stroke="#fb7185" strokeOpacity={0.3} strokeWidth={1} fill="url(#gradRejectPast)" connectNulls={false} dot={false} />

            {/* Future (full opacity, confident) areas */}
            <Area type="monotone" dataKey="acceptedFuture" stroke="#34d399" strokeWidth={2} fill="url(#gradAccept)" connectNulls={false} dot={false} />
            <Area type="monotone" dataKey="waitlistedFuture" stroke="#fbbf24" strokeWidth={2} fill="url(#gradWL)" connectNulls={false} dot={false} />
            <Area type="monotone" dataKey="rejectedFuture" stroke="#fb7185" strokeWidth={2} fill="url(#gradReject)" connectNulls={false} dot={false} />

            {/* Confidence threshold marker */}
            <ReferenceLine
              x={confCrossDay}
              stroke="#000"
              strokeDasharray="3 3"
              strokeWidth={1}
              strokeOpacity={0.35}
              label={{
                value: "5% decided",
                position: "insideTopRight",
                fill: "#000",
                fontSize: 9,
              }}
            />

            {/* Current day marker */}
            {currentDay !== null && (
              <ReferenceLine
                x={currentDay}
                stroke="#e2e8f0"
                strokeDasharray="6 4"
                strokeWidth={2}
                label={{
                  value: "Today",
                  position: "top",
                  fill: "#e2e8f0",
                  fontSize: 11,
                  fontWeight: 600,
                }}
              />
            )}

          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-3 flex flex-wrap justify-center gap-x-6 gap-y-1 text-xs">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-4 border-2 border-black bg-emerald-400" />
          Accepted
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-4 border-2 border-black bg-amber-400" />
          Waitlisted
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-4 border-2 border-black bg-rose-400" />
          Rejected
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-0 w-4 border-t-2 border-dashed border-white" />
          Today
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-0 w-4 border-t-2 border-dashed border-black" />
          Low Data
        </span>
      </div>
    </div>
  );
}

function App() {
  const [schools, setSchools] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [timelineData, setTimelineData] = useState<TimelineData | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Form state
  const [lsat, setLsat] = useState(170);
  const [gpa, setGpa] = useState(3.8);
  const [schoolName, setSchoolName] = useState("");
  const [sentDate, setSentDate] = useState("");
  const [currentDate, setCurrentDate] = useState(
    new Date().toISOString().slice(0, 10)
  );
  const [matriculatingYear, setMatriculatingYear] = useState(2026);
  const [urm, setUrm] = useState(false);
  const [isInternational, setIsInternational] = useState(false);
  const [nonTrad, setNonTrad] = useState(false);
  const [isInState, setIsInState] = useState(false);
  const [isFeeWaived, setIsFeeWaived] = useState(false);
  const [isMilitary, setIsMilitary] = useState(false);
  const [cAndF, setCAndF] = useState(false);
  const [softs, setSofts] = useState("");
  const [yearsOut, setYearsOut] = useState("");

  useEffect(() => {
    fetch(`${API_BASE}/api/schools`)
      .then((r) => r.json())
      .then((d) => setSchools(d.schools))
      .catch(() => setError("Failed to load school list"));
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!schoolName) {
      setError("Please select a school");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const body: Record<string, unknown> = {
        lsat,
        gpa,
        school_name: schoolName,
        matriculating_year: matriculatingYear,
        current_date: currentDate,
        urm,
        is_international: isInternational,
        non_trad: nonTrad,
        is_in_state: isInState,
        is_fee_waived: isFeeWaived,
        is_military: isMilitary,
        c_and_f: cAndF,
      };
      if (sentDate) body.sent_date = sentDate;
      if (softs) body.softs = softs;
      if (yearsOut !== "") body.years_out = parseInt(yearsOut);

      const [resp, timelineResp] = await Promise.all([
        fetch(`${API_BASE}/api/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }),
        fetch(`${API_BASE}/api/predict_timeline`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }),
      ]);
      const data = await resp.json();
      const tlData = await timelineResp.json();
      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
      }
      if (!tlData.error) {
        setTimelineData(tlData);
      }
    } catch {
      setError("Failed to get prediction");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-[var(--nb-bg)] text-black">
      <div className="mx-auto max-w-6xl px-4 py-10 sm:py-14">
        {/* Header */}
        <div className="mb-10">
          <div className="inline-flex items-center gap-2 border-2 border-black bg-white px-3 py-1 text-xs font-bold uppercase tracking-widest shadow-[4px_4px_0_0_#000]">
            LSD.law Data Model
          </div>
          <h1 className="mt-4 text-4xl font-black tracking-tight sm:text-6xl">
            Law School Admissions Calculator
          </h1>
          <p className="mt-3 max-w-2xl text-sm font-medium text-neutral-700">
            LightGBM predictions trained on 700K+ decisions with cycle-year
            and timing features.
          </p>
        </div>

        <div className="grid gap-8 lg:grid-cols-[1fr_440px]">
          {/* Form */}
          <form
            onSubmit={handleSubmit}
            className="nb-card space-y-6"
          >
            {/* Core Stats */}
            <div>
              <h2 className="mb-4 text-lg font-black">
                Core Stats
              </h2>
              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <label className="nb-label mb-1 block">
                    LSAT Score
                  </label>
                  <input
                    type="number"
                    min={120}
                    max={180}
                    value={lsat}
                    onChange={(e) => setLsat(parseInt(e.target.value) || 120)}
                    className={INPUT_CLS}
                  />
                  <div className="mt-1 text-xs font-medium text-neutral-600">120 - 180</div>
                </div>
                <div>
                  <label className="nb-label mb-1 block">
                    GPA
                  </label>
                  <input
                    type="number"
                    min={2.0}
                    max={4.3}
                    step={0.01}
                    value={gpa}
                    onChange={(e) => setGpa(parseFloat(e.target.value) || 2.0)}
                    className={INPUT_CLS}
                  />
                  <div className="mt-1 text-xs font-medium text-neutral-600">
                    2.00 - 4.30 (LSAC scale)
                  </div>
                </div>
              </div>
            </div>

            {/* School & Timing */}
            <div>
              <h2 className="mb-4 text-lg font-black">
                School & Timing
              </h2>
              <div className="space-y-4">
                <div>
                  <label className="nb-label mb-1 block">
                    School
                  </label>
                  <SchoolCombobox
                    schools={schools}
                    value={schoolName}
                    onChange={setSchoolName}
                  />
                </div>
                <div className="grid gap-4 sm:grid-cols-3">
                  <div>
                    <label className="nb-label mb-1 block">
                      App Sent Date
                    </label>
                    <input
                      type="date"
                      value={sentDate}
                      onChange={(e) => setSentDate(e.target.value)}
                      className={INPUT_CLS}
                    />
                  </div>
                  <div>
                    <label className="nb-label mb-1 block">
                      Today's Date
                    </label>
                    <input
                      type="date"
                      value={currentDate}
                      onChange={(e) => setCurrentDate(e.target.value)}
                      className={INPUT_CLS}
                    />
                    <div className="mt-1 text-xs font-medium text-neutral-600">
                      Where are you in the cycle?
                    </div>
                  </div>
                  <div>
                    <label className="nb-label mb-1 block">
                      Cycle (Fall)
                    </label>
                    <select
                      value={matriculatingYear}
                      onChange={(e) =>
                        setMatriculatingYear(parseInt(e.target.value))
                      }
                      className="nb-select"
                    >
                      {[2024, 2025, 2026, 2027].map((y) => (
                        <option key={y} value={y}>
                          {y}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
            </div>

            {/* Profile */}
            <div>
              <h2 className="mb-4 text-lg font-black">
                Applicant Profile
              </h2>
              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <label className="nb-label mb-1 block">
                    Softs Tier
                  </label>
                  <select
                    value={softs}
                    onChange={(e) => setSofts(e.target.value)}
                    className="nb-select"
                  >
                    <option value="">Unknown</option>
                    <option value="T1">T1 - Extraordinary</option>
                    <option value="T2">T2 - Strong</option>
                    <option value="T3">T3 - Average</option>
                    <option value="T4">T4 - Below Average</option>
                  </select>
                </div>
                <div>
                  <label className="nb-label mb-1 block">
                    Years Out of School
                  </label>
                  <input
                    type="number"
                    min={0}
                    max={50}
                    value={yearsOut}
                    onChange={(e) => setYearsOut(e.target.value)}
                    placeholder="Optional"
                    className={INPUT_CLS}
                  />
                </div>
              </div>

              {/* Toggles */}
              <div className="mt-4 flex flex-wrap gap-3">
                {[
                  { label: "URM", val: urm, set: setUrm },
                  {
                    label: "International",
                    val: isInternational,
                    set: setIsInternational,
                  },
                  { label: "Non-Trad", val: nonTrad, set: setNonTrad },
                  { label: "In-State", val: isInState, set: setIsInState },
                  {
                    label: "Fee Waived",
                    val: isFeeWaived,
                    set: setIsFeeWaived,
                  },
                  { label: "Military", val: isMilitary, set: setIsMilitary },
                  { label: "C&F Issues", val: cAndF, set: setCAndF },
                ].map(({ label, val, set }) => (
                  <button
                    type="button"
                    key={label}
                    onClick={() => set(!val)}
                    className={`border-2 border-black px-3 py-1 text-xs font-bold transition-transform active:translate-x-[1px] active:translate-y-[1px] ${
                      val ? "bg-indigo-600 text-white" : "bg-white text-black"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className={`nb-button ${loading ? "opacity-70" : ""}`}
            >
              {loading ? "Calculating..." : "Calculate Admission Odds"}
            </button>
          </form>

          {/* Results Panel */}
          <div className="space-y-6">
            {error && (
              <div className="border-2 border-black bg-rose-100 p-4 text-sm font-medium text-black shadow-[6px_6px_0_0_#000]">
                {error}
              </div>
            )}

            {result && (
              <>
                {/* Probabilities */}
                <div className="nb-card">
                  <h2 className="mb-1 text-lg font-black">
                    Predicted Odds
                  </h2>
                  <p className="mb-4 text-xs font-medium text-neutral-700">
                    Conditional on no decision received yet
                  </p>
                  <div className="space-y-4">
                    <ProbabilityBar
                      label="Accepted"
                      value={result.probabilities.accepted}
                      color="text-emerald-600"
                    />
                    <ProbabilityBar
                      label="Waitlisted"
                      value={result.probabilities.waitlisted}
                      color="text-amber-600"
                    />
                    <ProbabilityBar
                      label="Rejected"
                      value={result.probabilities.rejected}
                      color="text-rose-600"
                    />
                  </div>
                </div>


                {/* School Context */}
                <div className="nb-card">
                  <h2 className="mb-4 text-lg font-black">
                    School Context
                  </h2>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between font-medium text-neutral-700">
                      <span>Historical Accept Rate</span>
                      <span className="font-black text-black">
                        {result.school_context.school_accept_rate}%
                      </span>
                    </div>
                    <div className="flex justify-between font-medium text-neutral-700">
                      <span>Median LSAT</span>
                      <span className="font-black text-black">
                        {result.school_context.school_median_lsat}
                      </span>
                    </div>
                    <div className="flex justify-between font-medium text-neutral-700">
                      <span>LSAT 25th - 75th</span>
                      <span className="font-black text-black">
                        {result.school_context.school_25_lsat} -{" "}
                        {result.school_context.school_75_lsat}
                      </span>
                    </div>
                    <div className="flex justify-between font-medium text-neutral-700">
                      <span>Median GPA</span>
                      <span className="font-black text-black">
                        {result.school_context.school_median_gpa?.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between font-medium text-neutral-700">
                      <span>GPA 25th - 75th</span>
                      <span className="font-black text-black">
                        {result.school_context.school_25_gpa?.toFixed(2)} -{" "}
                        {result.school_context.school_75_gpa?.toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between font-medium text-neutral-700">
                      <span>Total Data Points</span>
                      <span className="font-black text-black">
                        {result.school_context.school_count?.toLocaleString()}
                      </span>
                    </div>
                  </div>
                </div>
              </>
            )}

            {/* Timeline Chart */}
            {timelineData && (
              <AdmissionTimeline
                data={timelineData}
                matYear={matriculatingYear}
              />
            )}

            {!result && !error && (
              <div className="flex h-64 items-center justify-center border-2 border-dashed border-black bg-white shadow-[6px_6px_0_0_#000]">
                <p className="text-center text-sm font-medium text-neutral-700">
                  Fill in your details and click
                  <br />
                  <span className="font-black text-black">
                    Calculate Admission Odds
                  </span>
                  <br />
                  <br />
                  <span className="text-xs font-medium text-neutral-600">
                    Set "Today's Date" to see how your odds
                    <br />
                    evolve as the cycle progresses
                  </span>
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Full-width Visualization Section */}
        {result && schoolName && (
          <div className="mt-8 space-y-6">
            <h2 className="text-center text-xl font-black">
              School Analytics
            </h2>
            <LsatGpaScatter
              schoolName={schoolName}
              userLsat={lsat}
              userGpa={gpa}
              year={matriculatingYear}
            />
            <ApplicantsLikeYou
              schoolName={schoolName}
              lsat={lsat}
              gpa={gpa}
            />
            <div className="grid gap-6 lg:grid-cols-2">
              <MedianDrift schoolName={schoolName} />
            </div>
            <WaitTimeDistribution
              schoolName={schoolName}
              daysWaiting={
                sentDate && currentDate
                  ? Math.max(0, Math.round((new Date(currentDate).getTime() - new Date(sentDate).getTime()) / 86400000))
                  : undefined
              }
            />
          </div>
        )}

        {/* Cycle Pulse — always visible */}
        <div className="mt-12">
          <div className="mb-4 flex items-center gap-3">
            <h2 className="text-xl font-black">Cycle Pulse</h2>
            <span className="nb-pill bg-indigo-100 text-indigo-800">NEW</span>
          </div>
          <p className="mb-4 text-sm font-medium text-neutral-700">
            Is this admissions cycle moving faster or slower than usual?
            Compare decision volume across cycles — all schools or a specific one.
          </p>
          <CyclePace schoolList={schools} />
        </div>

        {/* Footer */}
        <div className="mt-12 text-center text-xs font-medium text-neutral-700">
          Data sourced from{" "}
          <a
            href="https://lsd.law"
            className="underline decoration-2 underline-offset-2"
            target="_blank"
          >
            LSD.law
          </a>
          . Model: LightGBM GBDT with cycle-timing features, trained on
          700K+ decisions. For informational purposes only.
        </div>
      </div>
    </div>
  );
}

export default App;
