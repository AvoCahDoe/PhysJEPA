import { NavLink, Outlet } from "react-router-dom";

const NAV = [
  { to: "/", label: "Home", end: true },
  { to: "/docs", label: "Docs" },
  { to: "/try", label: "Try" },
  { to: "/results", label: "Results" },
];

const GITHUB = "https://github.com/AvoCahDoe/PhysJEPA";
const REPORT = "https://github.com/AvoCahDoe/PhysJEPA/blob/main/docs/report.md";

export function AppShell() {
  return (
    <div className="shell">
      <header className="site-header">
        <div className="site-header-inner">
          <NavLink to="/" className="brand">
            <span className="brand-mark">◉</span>
            <span>
              <strong>PhysJEPA</strong>
              <small>JEPA physics diagnostic</small>
            </span>
          </NavLink>
          <nav className="site-nav">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
          <div className="header-actions">
            <a href={GITHUB} target="_blank" rel="noopener noreferrer" className="btn-ghost">
              GitHub
            </a>
            <a href={REPORT} target="_blank" rel="noopener noreferrer" className="btn-ghost">
              Report
            </a>
          </div>
        </div>
      </header>
      <main className="site-main">
        <Outlet />
      </main>
      <footer className="site-footer">
        <p>
          Mid-scale run <code>paper_mid</code> · 1k episodes · 80 epochs ·{" "}
          <a href="https://physjepa.vercel.app">physjepa.vercel.app</a>
        </p>
      </footer>
    </div>
  );
}
