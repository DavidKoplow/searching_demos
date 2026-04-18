import {
  ACTIONS,
  getAllTiles,
  getTileType,
  slipActionOutcomes,
  type Action,
  type TileId,
} from './game'

export type RewardConfig = {
  goal: number
  trap: number
  stuck: number
  normal: number
  step: number
}

export type VIIterationSnapshot = {
  iteration: number
  V: Record<TileId, number>
  Q: Record<TileId, Record<Action, number>>
  policy: Record<TileId, Action | null>
  maxDelta: number
}

export type VIResult = {
  snapshots: VIIterationSnapshot[]
  converged: boolean
  rewards: RewardConfig
  gamma: number
  goalTile: TileId
}

export const DEFAULT_VI_REWARDS: RewardConfig = {
  goal: 10,
  trap: -10,
  stuck: 0,
  normal: 0,
  step: -1,
}

function tileReward(tile: TileId, goalTile: TileId, rewards: RewardConfig): number {
  if (tile === goalTile) return rewards.goal
  const kind = getTileType(tile)
  if (kind === 'T') return rewards.trap
  if (kind === 'S') return rewards.stuck
  return rewards.normal
}

function isTerminal(tile: TileId, goalTile: TileId): boolean {
  return tile === goalTile || getTileType(tile) === 'T'
}

export function computeQ(
  tile: TileId,
  action: Action,
  V: Record<TileId, number>,
  gamma: number,
  goalTile: TileId,
  rewards: RewardConfig,
  actionSuccessProb = 1,
): number {
  if (isTerminal(tile, goalTile)) {
    return tileReward(tile, goalTile, rewards)
  }
  const outcomes = slipActionOutcomes(tile, action, actionSuccessProb, goalTile)
  let expected = 0
  for (const outcome of outcomes) {
    const destTerminal = isTerminal(outcome.destination, goalTile)
    const r = destTerminal
      ? rewards.step
      : rewards.step + tileReward(outcome.destination, goalTile, rewards)
    expected += outcome.probability * (r + gamma * V[outcome.destination])
  }
  return expected
}

function bestAction(
  tile: TileId,
  V: Record<TileId, number>,
  gamma: number,
  goalTile: TileId,
  rewards: RewardConfig,
  actionSuccessProb = 1,
): { action: Action | null; qValues: Record<Action, number> } {
  const qValues = {} as Record<Action, number>
  let best: Action | null = null
  let bestQ = Number.NEGATIVE_INFINITY
  for (const a of ACTIONS) {
    const q = computeQ(tile, a, V, gamma, goalTile, rewards, actionSuccessProb)
    qValues[a] = q
    if (q > bestQ) {
      bestQ = q
      best = a
    }
  }
  if (isTerminal(tile, goalTile)) {
    best = null
  }
  return { action: best, qValues }
}

export type VIMode = 'max' | 'random'

export function runValueIteration(
  goalTile: TileId,
  gamma: number,
  maxIterations: number,
  rewards: RewardConfig = DEFAULT_VI_REWARDS,
  epsilon = 1e-4,
  mode: VIMode = 'max',
  actionSuccessProb = 1,
): VIResult {
  const tiles = getAllTiles()
  let V: Record<TileId, number> = {} as Record<TileId, number>
  for (const tile of tiles) V[tile] = 0

  const pickAction = (
    tile: TileId,
    Vcur: Record<TileId, number>,
  ): { action: Action | null; qValues: Record<Action, number> } => {
    const qValues = {} as Record<Action, number>
    for (const a of ACTIONS) {
      qValues[a] = computeQ(tile, a, Vcur, gamma, goalTile, rewards, actionSuccessProb)
    }
    if (isTerminal(tile, goalTile)) {
      return { action: null, qValues }
    }
    if (mode === 'max') {
      return bestAction(tile, Vcur, gamma, goalTile, rewards, actionSuccessProb)
    }
    return { action: null, qValues }
  }

  const snapshots: VIIterationSnapshot[] = []
  const initialQ = {} as Record<TileId, Record<Action, number>>
  const initialPolicy = {} as Record<TileId, Action | null>
  for (const tile of tiles) {
    const { action, qValues } = pickAction(tile, V)
    initialQ[tile] = qValues
    initialPolicy[tile] = action
  }
  snapshots.push({
    iteration: 0,
    V: { ...V },
    Q: initialQ,
    policy: initialPolicy,
    maxDelta: Number.POSITIVE_INFINITY,
  })

  let converged = false
  for (let iter = 1; iter <= maxIterations; iter += 1) {
    const nextV: Record<TileId, number> = { ...V }
    const Q = {} as Record<TileId, Record<Action, number>>
    const policy = {} as Record<TileId, Action | null>
    let maxDelta = 0
    for (const tile of tiles) {
      const { action, qValues } = pickAction(tile, V)
      Q[tile] = qValues
      policy[tile] = action
      if (isTerminal(tile, goalTile)) {
        nextV[tile] = tileReward(tile, goalTile, rewards)
      } else if (mode === 'random') {
        let sum = 0
        for (const a of ACTIONS) sum += qValues[a]
        nextV[tile] = sum / ACTIONS.length
      } else {
        const v = action !== null ? qValues[action] : 0
        nextV[tile] = v
      }
      const delta = Math.abs(nextV[tile] - V[tile])
      if (delta > maxDelta) maxDelta = delta
    }
    V = nextV
    snapshots.push({ iteration: iter, V: { ...V }, Q, policy, maxDelta })
    if (maxDelta < epsilon) {
      converged = true
      break
    }
  }

  return { snapshots, converged, rewards, gamma, goalTile }
}
