import { useMemo } from 'react'
import katex from 'katex'

export const SLIDE_COLORS = {
  state: '#1E641E',
  action: '#BE1E1E',
  reward: '#C87800',
  transition: '#1450C8',
  obsLikelihood: '#0082A0',
  observation: '#96500A',
  belief: '#821EAA',
  gamma: '#646464',
  value: '#0F172A',
} as const

const SLIDE_MACRO_COLORS: Record<string, string> = {
  cS: SLIDE_COLORS.state,
  cA: SLIDE_COLORS.action,
  cR: SLIDE_COLORS.reward,
  cT: SLIDE_COLORS.transition,
  cZ: SLIDE_COLORS.obsLikelihood,
  cO: SLIDE_COLORS.observation,
  cB: SLIDE_COLORS.belief,
  cG: SLIDE_COLORS.gamma,
  cV: SLIDE_COLORS.value,
}

function expandMacros(tex: string): string {
  let out = tex
  for (const [name, color] of Object.entries(SLIDE_MACRO_COLORS)) {
    const pattern = new RegExp(`\\\\${name}\\{`, 'g')
    out = out.replace(pattern, `\\textcolor{${color}}{`)
  }
  return out
}

function renderTex(tex: string, displayMode: boolean): string {
  const expanded = expandMacros(tex)
  try {
    return katex.renderToString(expanded, {
      displayMode,
      throwOnError: false,
      strict: 'ignore',
    })
  } catch (err) {
    return `<span style="color:#c00">${(err as Error).message}</span>`
  }
}

export function Eq({ children }: { children: string }) {
  const html = useMemo(() => renderTex(children, true), [children])
  return <div className="katex-block" dangerouslySetInnerHTML={{ __html: html }} />
}

export function IEq({ children }: { children: string }) {
  const html = useMemo(() => renderTex(children, false), [children])
  return <span className="katex-inline" dangerouslySetInnerHTML={{ __html: html }} />
}

export function ColorLegend() {
  return (
    <div className="color-legend">
      <span style={{ color: SLIDE_COLORS.state }}>state s</span>
      <span style={{ color: SLIDE_COLORS.action }}>action a</span>
      <span style={{ color: SLIDE_COLORS.reward }}>reward R</span>
      <span style={{ color: SLIDE_COLORS.transition }}>transition T</span>
      <span style={{ color: SLIDE_COLORS.obsLikelihood }}>obs likelihood</span>
      <span style={{ color: SLIDE_COLORS.observation }}>observation o</span>
      <span style={{ color: SLIDE_COLORS.belief }}>belief b</span>
      <span style={{ color: SLIDE_COLORS.gamma }}>discount γ</span>
    </div>
  )
}
