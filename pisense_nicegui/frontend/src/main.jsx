import React, { useCallback, useEffect, useId, useState } from "react";
import ReactDOM from "react-dom/client";
import { Cpu } from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import "./index.css";

const POLL_INTERVAL_MS = 3000;

/** Modalità NiceGUI: i dati arrivano da Python tramite `CustomEvent('pisense-metrics')`, senza fetch REST. */
function isNiceGuiMode() {
  return typeof window !== "undefined" && window.__PISENSE_MODE__ === "nicegui";
}

function cn(...classes) {
  return classes.filter(Boolean).join(" ");
}

function Card({ className, children, ...props }) {
  return (
    <div
      className={cn(
        "rounded-lg border border-border bg-card text-card-foreground shadow-sm",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

function CardHeader({ className, children, ...props }) {
  return (
    <div className={cn("flex flex-col space-y-1.5 p-6", className)} {...props}>
      {children}
    </div>
  );
}

function CardTitle({ className, children, ...props }) {
  return (
    <div
      className={cn("text-2xl font-semibold leading-none tracking-tight font-headline", className)}
      {...props}
    >
      {children}
    </div>
  );
}

function CardContent({ className, children, ...props }) {
  return (
    <div className={cn("p-6 pt-0", className)} {...props}>
      {children}
    </div>
  );
}

function Gauge({ label, value, max = 100, unit, color, icon, onClick }) {
  const id = useId();
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const progress = Math.min(Math.max(value, 0), max);
  const offset = circumference - (progress / max) * circumference;
  const glowId = `glow-${id.replace(/:/g, "")}`;

  return (
    <button
      type="button"
      onClick={onClick}
      className="group flex flex-col items-center gap-2 rounded-lg p-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary"
    >
      <div className="relative h-36 w-36">
        <svg className="h-full w-full" viewBox="0 0 140 140" aria-labelledby={`${id}-title`} role="img">
          <title id={`${id}-title`}>{`${label}: ${Number(value).toFixed(1)} ${unit}`}</title>
          <defs>
            <filter id={glowId}>
              <feDropShadow dx="0" dy="0" stdDeviation="3.5" floodColor={color} />
            </filter>
          </defs>
          <circle cx="70" cy="70" r={radius} fill="none" strokeWidth="12" className="stroke-muted/30" />
          <circle
            cx="70"
            cy="70"
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth="12"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            transform="rotate(-90 70 70)"
            className="transition-[stroke-dashoffset] duration-500 ease-in-out"
            style={{ filter: `url(#${glowId})` }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center text-foreground">
          <div className="text-xl">{icon}</div>
          <div className="font-code text-2xl font-bold" style={{ color }}>
            {Number(value).toFixed(1)}
          </div>
          <div className="text-xs text-muted-foreground">{unit}</div>
        </div>
      </div>
      <div className="text-sm font-medium text-muted-foreground transition-colors group-hover:text-foreground">
        {label}
      </div>
    </button>
  );
}

function Clock() {
  const [time, setTime] = useState("--:--:--");
  const [date, setDate] = useState("");

  useEffect(() => {
    const updateClock = () => {
      const now = new Date();
      setTime(now.toLocaleTimeString("it-IT"));
      setDate(
        now.toLocaleDateString("it-IT", {
          weekday: "long",
          day: "numeric",
          month: "long",
        })
      );
    };
    updateClock();
    const timerId = setInterval(updateClock, 1000);
    return () => clearInterval(timerId);
  }, []);

  return (
    <div className="text-right">
      <div className="font-code text-2xl font-extrabold tracking-wider text-secondary md:text-3xl">{time}</div>
      <div className="mt-1 text-xs text-muted-foreground">{date}</div>
    </div>
  );
}

function Header({ status, uptime }) {
  return (
    <header className="mb-6 flex flex-col items-start justify-between gap-4 md:mb-8 md:flex-row md:items-center">
      <div className="flex items-center gap-4">
        <Cpu className="h-12 w-12 text-primary/90" />
        <div>
          <h1 className="font-headline text-2xl font-bold md:text-3xl">
            PiSense <span className="text-primary/80">Hub</span>
          </h1>
          <div className="mt-1 flex items-center gap-2">
            <div className="h-2.5 w-2.5 animate-pulse rounded-full bg-green-400 shadow-[0_0_8px_#34d399]" />
            <span className="text-xs font-medium text-green-400">{status}</span>
            <span className="font-code text-xs text-muted-foreground">• {uptime}</span>
          </div>
        </div>
      </div>
      <Clock />
    </header>
  );
}

function PiStatusGauges({ stats, onGaugeClick }) {
  return (
    <Card className="bg-card/60 p-4 backdrop-blur-sm sm:p-6">
      <div className="flex flex-wrap justify-around gap-4">
        <Gauge
          label="CPU"
          value={stats.cpu}
          unit="%"
          color="hsl(var(--chart-1))"
          icon="⚡"
          onClick={() => onGaugeClick("cpu")}
        />
        <Gauge
          label="RAM"
          value={stats.ram}
          unit="%"
          color="hsl(var(--chart-2))"
          icon="🧠"
          onClick={() => onGaugeClick("ram")}
        />
        <Gauge
          label="Disk"
          value={stats.disk}
          unit="%"
          color="hsl(var(--chart-3))"
          icon="💾"
          onClick={() => onGaugeClick("disk")}
        />
        <Gauge
          label="Temp"
          value={stats.temp}
          max={85}
          unit="°C"
          color="hsl(var(--chart-4))"
          icon="🌡️"
          onClick={() => onGaugeClick("temp")}
        />
      </div>
    </Card>
  );
}

function PerformanceChart({ data }) {
  return (
    <div
      className="flex aspect-video h-full w-full justify-center text-xs [&_.recharts-cartesian-axis-tick_text]:fill-muted-foreground [&_.recharts-cartesian-grid_line]:stroke-border/50"
      data-chart="pisense"
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{
            top: 5,
            right: 10,
            left: -10,
            bottom: 0,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.5)" vertical={false} />
          <XAxis
            dataKey="time"
            tickLine={false}
            axisLine={false}
            tickMargin={8}
            tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
          />
          <YAxis
            tickLine={false}
            axisLine={false}
            tickMargin={8}
            domain={[0, 100]}
            tickFormatter={(v) => `${v}%`}
            tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
          />
          <Tooltip
            cursor={{ stroke: "hsl(var(--border))", strokeWidth: 1, strokeDasharray: "3 3" }}
            contentStyle={{
              backgroundColor: "hsl(var(--card))",
              border: "1px solid hsl(var(--border) / 0.5)",
              borderRadius: "0.5rem",
              fontSize: "0.75rem",
            }}
            labelStyle={{ fontWeight: 700, fontFamily: "Source Code Pro, monospace" }}
          />
          <Line
            type="monotone"
            dataKey="cpu"
            name="CPU"
            stroke="hsl(var(--chart-1))"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
          />
          <Line
            type="monotone"
            dataKey="ram"
            name="RAM"
            stroke="hsl(var(--chart-2))"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function SummaryCard({ stats }) {
  const items = [
    { label: "CPU Temp", value: `${stats.temp.toFixed(1)}°C`, icon: "🌡️", color: "text-rose-400" },
    { label: "RAM Usage", value: stats.memInfo, icon: "🧠", color: "text-violet-400" },
    { label: "Network Down", value: `${stats.netRx.toFixed(2)} MB/s`, icon: "↓", color: "text-cyan-400" },
    { label: "Network Up", value: `${stats.netTx.toFixed(2)} MB/s`, icon: "↑", color: "text-amber-400" },
  ];

  return (
    <Card className="bg-card/60 backdrop-blur-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-3 text-lg">
          <span className="text-2xl">📝</span>
          Riepilogo
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-3">
          {items.map((item) => (
            <div key={item.label} className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-3">
                <span className="text-lg">{item.icon}</span>
                <span className="text-muted-foreground">{item.label}</span>
              </div>
              <span className={cn("font-code font-bold", item.color)}>{item.value}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function Footer({ lastUpdated, memTotalLabel }) {
  return (
    <footer className="mt-8 w-full py-4 text-center">
      <p className="font-code text-xs text-muted-foreground">
        PiSense Hub{memTotalLabel ? ` • ${memTotalLabel}` : ""} • Aggiornato alle{" "}
        {lastUpdated.toLocaleTimeString("it-IT")}
      </p>
    </footer>
  );
}

const defaultStats = {
  cpu: 0,
  ram: 0,
  disk: 0,
  temp: 0,
  uptime: "—",
  memInfo: "—",
  status: "Connessione…",
  netRx: 0,
  netTx: 0,
};

function App() {
  const [stats, setStats] = useState(defaultStats);
  const [chartData, setChartData] = useState([]);
  const [lastUpdated, setLastUpdated] = useState(() => new Date());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [memFooter, setMemFooter] = useState("");

  const applyPayload = useCallback((payload) => {
    const snap = payload.snapshot || {};
    const summary = payload.summary || {};

    setStats({
      cpu: snap.cpu ?? 0,
      ram: snap.ram ?? 0,
      disk: snap.disk ?? summary.disk_percent ?? 0,
      temp: snap.temp ?? 0,
      uptime: summary.uptime || "—",
      memInfo: summary.memory_info || "—",
      status: summary.status || "Online",
      netRx: snap.net_rx ?? 0,
      netTx: snap.net_tx ?? 0,
    });

    const pts = payload.points || [];
    const mapped = pts.slice(-40).map((p) => ({
      time: String(p.index),
      cpu: p.cpu,
      ram: p.ram,
    }));
    setChartData(mapped);

    const memStr = summary.memory_info || "";
    const totalMatch = memStr.match(/\/\s*(\d+)\s*MB\)/);
    if (totalMatch) {
      setMemFooter(`${totalMatch[1]} MB RAM`);
    }

    setLastUpdated(new Date());
    setError("");
    setLoading(false);
  }, []);

  const fetchMetrics = useCallback(async () => {
    try {
      const response = await fetch("/api/metrics");
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const payload = await response.json();
      applyPayload(payload);
    } catch (e) {
      setError("Impossibile recuperare i dati dal backend. Verifica che FastAPI sia in esecuzione.");
      setStats((s) => ({ ...s, status: "Offline" }));
    } finally {
      setLoading(false);
    }
  }, [applyPayload]);

  useEffect(() => {
    if (isNiceGuiMode()) {
      const onMetrics = (e) => {
        if (e.detail) {
          applyPayload(e.detail);
        }
      };
      window.addEventListener("pisense-metrics", onMetrics);
      return () => window.removeEventListener("pisense-metrics", onMetrics);
    }
    fetchMetrics();
    const t = window.setInterval(fetchMetrics, POLL_INTERVAL_MS);
    return () => window.clearInterval(t);
  }, [fetchMetrics, applyPayload]);

  const handleGaugeClick = useCallback((name) => {
    if (isNiceGuiMode()) {
      try {
        window.__PISENSE_GAUGE_CLICK__ = name;
      } catch {
        /* ignore */
      }
      setNotice(`Selezione gauge: ${name} (evento verso Python NiceGUI)`);
    } else {
      setNotice(`Selezione: ${name} (endpoint backend: /api/metrics)`);
    }
    window.setTimeout(() => setNotice(""), 3200);
  }, []);

  return (
    <>
      <div className="bg-grid" />
      <div className="orb orb1" />
      <div className="orb orb2" />
      <div className="orb orb3" />
      <main className="relative z-10">
        <div className="container mx-auto p-4 sm:p-6 md:p-8">
          <Header status={stats.status} uptime={stats.uptime} />

          {error && (
            <div
              className="mb-4 rounded-lg border border-red-500/40 bg-red-950/40 px-4 py-3 text-sm text-red-200"
              role="alert"
            >
              {error}
            </div>
          )}
          {notice && (
            <div className="mb-4 rounded-lg border border-primary/30 bg-primary/10 px-4 py-2 text-center text-sm text-foreground">
              {notice}
            </div>
          )}

          <div className="mb-6 md:mb-8">
            <PiStatusGauges stats={stats} onGaugeClick={handleGaugeClick} />
          </div>

          <div className="grid grid-cols-1 gap-6 md:gap-8 lg:grid-cols-3">
            <Card className="bg-card/60 p-4 backdrop-blur-sm sm:p-6 lg:col-span-2">
              <h2 className="mb-4 font-headline text-lg font-bold">Performance</h2>
              <div className="h-64">
                {loading && chartData.length === 0 ? (
                  <div className="flex h-full items-center justify-center rounded-lg border border-dashed border-border/60 text-sm text-muted-foreground">
                    Caricamento storico…
                  </div>
                ) : (
                  <PerformanceChart data={chartData} />
                )}
              </div>
            </Card>

            <SummaryCard stats={stats} />
          </div>

          <Footer lastUpdated={lastUpdated} memTotalLabel={memFooter} />
        </div>
      </main>
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
