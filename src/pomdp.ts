import {
  ACTIONS,
  getAllTiles,
  getTileType,
  sampleSlipTransition,
  slipActionOutcomes,
  type Action,
  type TileId,
} from './game'
import {
  DEFAULT_VI_REWARDS,
  computeQ,
  runValueIteration,
  type RewardConfig,
  type VIIterationSnapshot,
} from './valueIteration'

export type POMDPObservation = 'N' | 'S' | 'T' | 'G'

export const POMDP_OBSERVATIONS: POMDPObservation[] = ['N', 'S', 'T', 'G']

export function observationTypeFor(tile: TileId, goalTile: TileId): POMDPObservation {
  if (tile === goalTile) return 'G'
  return getTileType(tile)
}

const OBSERVATION_LABELS: Record<POMDPObservation, string> = {
  N: 'Normal',
  S: 'Stuck',
  T: 'Trap',
  G: 'Goal',
}

export function observationLabel(o: POMDPObservation): string {
  return OBSERVATION_LABELS[o]
}

export type Belief = Record<TileId, number>

export type POMDPStepFrame = {
  step: number
  phase: 'initial' | 'action-selected' | 'transition' | 'observation' | 'belief-update' | 'terminated'
  belief: Belief
  priorBelief: Belief
  trueTile: TileId
  observation: POMDPObservation | null
  action: Action | null
  qmdpValues: Record<Action, number>
  message: string
  beliefAfterAction?: Belief
  terminated?: boolean
}

export type POMDPResult = {
  frames: POMDPStepFrame[]
  mdpSnapshots: VIIterationSnapshot[]
  gamma: number
  observationAccuracy: number
  finalSnapshot: VIIterationSnapshot
  goalTile: TileId
  rewards: RewardConfig
  terminated: boolean
  terminationReason: 'goal' | 'trap' | 'max-steps' | null
}

export type POMDPConfig = {
  start: TileId
  goalTile: TileId
  gamma: number
  observationAccuracy: number
  actionSuccessProb?: number
  rewards?: RewardConfig
  maxSteps?: number
  viIterations?: number
  rng?: () => number
  initialBelief?: Belief
}

function uniformBelief(tiles: TileId[]): Belief {
  const b = {} as Belief
  const p = 1 / tiles.length
  for (const t of tiles) b[t] = p
  return b
}

export function pointBelief(tiles: TileId[], tile: TileId): Belief {
  const b = {} as Belief
  for (const t of tiles) b[t] = t === tile ? 1 : 0
  return b
}

function normalize(b: Belief): Belief {
  let sum = 0
  for (const k of Object.keys(b) as TileId[]) sum += b[k]
  if (sum <= 0) return { ...b }
  const out = {} as Belief
  for (const k of Object.keys(b) as TileId[]) out[k] = b[k] / sum
  return out
}

export function observationProb(
  tile: TileId,
  observation: POMDPObservation,
  accuracy: number,
  goalTile: TileId,
): number {
  const actual = observationTypeFor(tile, goalTile)
  if (actual === observation) return accuracy
  return (1 - accuracy) / 3
}

export function updateBeliefAfterAction(
  belief: Belief,
  action: Action,
  goalTile: TileId,
  tiles: TileId[],
  actionSuccessProb = 1,
): Belief {
  const next = {} as Belief
  for (const t of tiles) next[t] = 0
  for (const t of tiles) {
    const mass = belief[t]
    if (mass <= 0) continue
    const outcomes = slipActionOutcomes(t, action, actionSuccessProb, goalTile)
    for (const out of outcomes) {
      next[out.destination] += mass * out.probability
    }
  }
  return normalize(next)
}

export function updateBeliefAfterObservation(
  belief: Belief,
  observation: POMDPObservation,
  accuracy: number,
  tiles: TileId[],
  goalTile: TileId,
): Belief {
  const out = {} as Belief
  for (const t of tiles) {
    out[t] = belief[t] * observationProb(t, observation, accuracy, goalTile)
  }
  return normalize(out)
}

export function qmdpAction(
  belief: Belief,
  Q: Record<TileId, Record<Action, number>>,
  tiles: TileId[],
): { action: Action; values: Record<Action, number> } {
  const values = {} as Record<Action, number>
  let best: Action = ACTIONS[0]
  let bestV = Number.NEGATIVE_INFINITY
  for (const a of ACTIONS) {
    let v = 0
    for (const t of tiles) v += belief[t] * Q[t][a]
    values[a] = v
    if (v > bestV) {
      bestV = v
      best = a
    }
  }
  return { action: best, values }
}

function sampleObservation(
  tile: TileId,
  accuracy: number,
  rng: () => number,
  goalTile: TileId,
): POMDPObservation {
  const actualType = observationTypeFor(tile, goalTile)
  const roll = rng()
  if (roll < accuracy) return actualType
  const others = POMDP_OBSERVATIONS.filter((t) => t !== actualType)
  return others[Math.floor(rng() * others.length)]
}

export function runPOMDP(config: POMDPConfig): POMDPResult {
  const {
    start,
    goalTile,
    gamma,
    observationAccuracy,
    actionSuccessProb = 1,
    rewards = DEFAULT_VI_REWARDS,
    maxSteps = 25,
    viIterations = 60,
    rng = Math.random,
  } = config
  const tiles = getAllTiles()

  const vi = runValueIteration(goalTile, gamma, viIterations, rewards, 1e-4, 'max', actionSuccessProb)
  const finalSnapshot = vi.snapshots[vi.snapshots.length - 1]
  const Q = finalSnapshot.Q

  const frames: POMDPStepFrame[] = []
  let trueTile = start
  const startBelief = config.initialBelief ?? uniformBelief(tiles)
  let belief = { ...startBelief }
  const initialObs = sampleObservation(trueTile, observationAccuracy, rng, goalTile)
  belief = updateBeliefAfterObservation(belief, initialObs, observationAccuracy, tiles, goalTile)

  frames.push({
    step: 0,
    phase: 'initial',
    belief: { ...belief },
    priorBelief: { ...startBelief },
    trueTile,
    observation: initialObs,
    action: null,
    qmdpValues: {} as Record<Action, number>,
    message: `Initial observation: the tile reads as ${observationLabel(initialObs)}. Belief updated from uniform.`,
  })

  let terminated = false
  let terminationReason: 'goal' | 'trap' | 'max-steps' | null = null
  let step = 1

  while (step <= maxSteps) {
    const { action, values } = qmdpAction(belief, Q, tiles)

    frames.push({
      step,
      phase: 'action-selected',
      belief: { ...belief },
      priorBelief: { ...belief },
      trueTile,
      observation: null,
      action,
      qmdpValues: values,
      message: `Q_MDP selects ${action}. Expected Q-values: ${ACTIONS
        .map((a) => `${a}=${values[a].toFixed(2)}`)
        .join(', ')}.`,
    })

    const priorBelief = { ...belief }
    const sampled = sampleSlipTransition(trueTile, action, actionSuccessProb, rng, goalTile)
    const prevTile = trueTile
    trueTile = sampled.destination

    const beliefAfterAction = updateBeliefAfterAction(belief, action, goalTile, tiles, actionSuccessProb)

    frames.push({
      step,
      phase: 'transition',
      belief: beliefAfterAction,
      priorBelief,
      trueTile,
      observation: null,
      action,
      qmdpValues: values,
      beliefAfterAction,
      message: `Transitioned ${prevTile} --${action}--> ${trueTile}. Belief propagated through the transition model.`,
    })

    if (trueTile === goalTile) {
      terminated = true
      terminationReason = 'goal'
    } else if (getTileType(trueTile) === 'T') {
      terminated = true
      terminationReason = 'trap'
    }

    if (terminated) {
      frames.push({
        step,
        phase: 'terminated',
        belief: beliefAfterAction,
        priorBelief,
        trueTile,
        observation: null,
        action,
        qmdpValues: values,
        message:
          terminationReason === 'goal'
            ? `Reached the goal ${trueTile}.`
            : `Fell into trap ${trueTile}.`,
        terminated: true,
      })
      belief = beliefAfterAction
      break
    }

    const observation = sampleObservation(trueTile, observationAccuracy, rng, goalTile)
    belief = updateBeliefAfterObservation(beliefAfterAction, observation, observationAccuracy, tiles, goalTile)

    frames.push({
      step,
      phase: 'belief-update',
      belief: { ...belief },
      priorBelief: beliefAfterAction,
      trueTile,
      observation,
      action,
      qmdpValues: values,
      message: `Tile under the agent reads as ${observationLabel(observation)}. Bayes update sharpens the belief.`,
    })

    step += 1
  }

  if (!terminated && step > maxSteps) {
    terminationReason = 'max-steps'
  }

  return {
    frames,
    mdpSnapshots: vi.snapshots,
    gamma,
    observationAccuracy,
    finalSnapshot,
    goalTile,
    rewards,
    terminated,
    terminationReason,
  }
}

export function qmdpActionForTile(
  Q: Record<TileId, Record<Action, number>>,
  tile: TileId,
): { action: Action; values: Record<Action, number> } {
  const values = {} as Record<Action, number>
  let best: Action = ACTIONS[0]
  let bestV = Number.NEGATIVE_INFINITY
  for (const a of ACTIONS) {
    values[a] = Q[tile][a]
    if (values[a] > bestV) {
      bestV = values[a]
      best = a
    }
  }
  return { action: best, values }
}

export { computeQ }
