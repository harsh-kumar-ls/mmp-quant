# MMP-Quant — React UI

A React + TypeScript + Tailwind version of the web UI (10 Nov 2026), built
alongside the original (`src/mmpquant/webapp/static/`, plain HTML/JS), which
stays the fallback. Both are clients of the same FastAPI server: every number
comes from `mmpquant serve`; this app adds no scoring behaviour.

## View it (no Node needed once built)

    mmpquant serve            # from the project root
    open http://127.0.0.1:8000/           # the React UI (main site)
    open http://127.0.0.1:8000/classic/   # the original UI (fallback)

When `webapp-react/dist/` exists, `mmpquant serve` serves it at `/` (and
`/react/`); without a build, `/` serves the original UI.

## Develop (needs Node 20+)

    cd webapp-react
    npm install
    npm run dev               # http://localhost:5173 — /api is proxied to :8000
    npm run build             # writes dist/, picked up by mmpquant serve

Components follow shadcn / 21st.dev conventions (`cn()` in `src/lib/utils.ts`,
Tailwind theme tokens in `src/index.css`), so catalogue components drop in.

## Tests

Browser tests live with the original UI's: `pytest -m e2e tests/e2e`
(`TestReactUI` — numbers vs result.json, drawer formula and navigation,
timing margin and window formula, coherence, rating set, phone width).
They skip if `dist/` is not built.
