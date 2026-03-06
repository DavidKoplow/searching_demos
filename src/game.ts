export const COLUMNS = ['A', 'B', 'C', 'D', 'E', 'F'] as const
export const BOARD_ROWS = [3, 2, 1] as const
export const ACTIONS = ['up', 'down', 'left', 'right'] as const

export type Column = (typeof COLUMNS)[number]
export type Row = 1 | 2 | 3
export type Action = (typeof ACTIONS)[number]
export type TileType = 'N' | 'S' | 'T'
export type Observation = TileType
export type TileId = `${Row}${Column}`

type MoveAttempt = {
  destination: TileId
  explanation: string
}

export type TransitionBranch = {
  label: string
  probability: number
  destination: TileId
  explanation: string
}

export type TransitionOutcome = {
  destination: TileId
  probability: number
  explanation: string
}

const COLUMN_TO_INDEX: Record<Column, number> = {
  A: 0,
  B: 1,
  C: 2,
  D: 3,
  E: 4,
  F: 5,
}

const INDEX_TO_COLUMN: Record<number, Column> = {
  0: 'A',
  1: 'B',
  2: 'C',
  3: 'D',
  4: 'E',
  5: 'F',
}

const TILE_TYPES: Record<TileId, TileType> = {
  '1A': 'S',
  '1B': 'N',
  '1C': 'N',
  '1D': 'N',
  '1E': 'N',
  '1F': 'N',
  '2A': 'T',
  '2B': 'N',
  '2C': 'S',
  '2D': 'N',
  '2E': 'N',
  '2F': 'N',
  '3A': 'N',
  '3B': 'N',
  '3C': 'T',
  '3D': 'N',
  '3E': 'N',
  '3F': 'N',
}

const WALLS = new Set<string>([
  '1A:up',
  '2A:down',
  '2A:up',
  '3A:down',
  '2D:right',
  '2E:left',
  '1E:right',
  '1F:left',
  '3E:right',
  '3F:left',
])

const ALL_TILES = BOARD_ROWS.flatMap((row) =>
  COLUMNS.map((column) => `${row}${column}` as TileId),
)

export function getAllTiles() {
  return [...ALL_TILES]
}

function parseTile(tile: TileId) {
  const [rowValue, columnValue] = [...tile] as [`${Row}`, Column]
  return {
    row: Number(rowValue) as Row,
    column: columnValue,
  }
}

function formatTile(row: number, columnIndex: number): TileId {
  return `${row}${INDEX_TO_COLUMN[columnIndex]}` as TileId
}

function isInsideBoard(row: number, columnIndex: number) {
  return row >= 1 && row <= 3 && columnIndex >= 0 && columnIndex < COLUMNS.length
}

function attemptMove(tile: TileId, action: Action): MoveAttempt {
  const { row, column } = parseTile(tile)
  const columnIndex = COLUMN_TO_INDEX[column]

  if (WALLS.has(`${tile}:${action}`)) {
    return {
      destination: tile,
      explanation: `A thick wall blocks moving ${action} from ${tile}, so you stay put.`,
    }
  }

  const deltas: Record<Action, [number, number]> = {
    up: [1, 0],
    down: [-1, 0],
    left: [0, -1],
    right: [0, 1],
  }

  const [rowDelta, columnDelta] = deltas[action]
  const nextRow = row + rowDelta
  const nextColumnIndex = columnIndex + columnDelta

  if (!isInsideBoard(nextRow, nextColumnIndex)) {
    return {
      destination: tile,
      explanation: `The move ${action} hits the outer wall, so you remain on ${tile}.`,
    }
  }

  const destination = formatTile(nextRow, nextColumnIndex)

  return {
    destination,
    explanation: `The chosen action moves you ${action} from ${tile} to ${destination}.`,
  }
}

function mergeOutcomes(branches: TransitionBranch[]) {
  const merged = new Map<TileId, TransitionOutcome>()

  for (const branch of branches) {
    const existing = merged.get(branch.destination)

    if (existing) {
      existing.probability += branch.probability
      existing.explanation = `${existing.explanation} ${branch.explanation}`
      continue
    }

    merged.set(branch.destination, {
      destination: branch.destination,
      probability: branch.probability,
      explanation: branch.explanation,
    })
  }

  return [...merged.values()].sort((left, right) => right.probability - left.probability)
}

export function getTileType(tile: TileId) {
  return TILE_TYPES[tile]
}

export function describeAction(action: Action) {
  return action[0].toUpperCase() + action.slice(1)
}

export function getRandomStartTile(rng: () => number = Math.random) {
  const randomIndex = Math.floor(rng() * ALL_TILES.length)
  return ALL_TILES[randomIndex]
}

function getAbsorbingExplanation(tile: TileId, goalTile?: TileId) {
  if (goalTile && tile === goalTile) {
    return `Goal tiles are absorbing, so ${tile} keeps you there forever.`
  }

  return `Trap tiles are absorbing, so ${tile} keeps you there forever.`
}

export function transitionForAction(tile: TileId, action: Action, goalTile?: TileId) {
  const tileType = getTileType(tile)

  if (tileType === 'T' || (goalTile && tile === goalTile)) {
    const absorbingExplanation = getAbsorbingExplanation(tile, goalTile)

    return {
      branches: [
        {
          label: tileType === 'T' ? 'trap' : 'goal',
          probability: 1,
          destination: tile,
          explanation: absorbingExplanation,
        },
      ],
      outcomes: [
        {
          destination: tile,
          probability: 1,
          explanation: absorbingExplanation,
        },
      ],
    }
  }

  const attemptedMove = attemptMove(tile, action)

  if (tileType === 'N') {
    const branches: TransitionBranch[] = [
      {
        label: 'normal-move',
        probability: 1,
        destination: attemptedMove.destination,
        explanation: `Normal tiles obey the chosen action. ${attemptedMove.explanation}`,
      },
    ]

    return {
      branches,
      outcomes: mergeOutcomes(branches),
    }
  }

  const branches: TransitionBranch[] = [
    {
      label: 'stuck-hold',
      probability: 0.5,
      destination: tile,
      explanation: `Stuck tiles have a 50% chance to waste the turn, so you stay on ${tile}.`,
    },
    {
      label: 'stuck-move',
      probability: 0.5,
      destination: attemptedMove.destination,
      explanation: `The other 50% of the time the stuck tile lets you act normally. ${attemptedMove.explanation}`,
    },
  ]

  return {
    branches,
    outcomes: mergeOutcomes(branches),
  }
}

export function sampleTransition(tile: TileId, action: Action, rng: () => number = Math.random, goalTile?: TileId) {
  const transition = transitionForAction(tile, action, goalTile)
  const randomValue = rng()
  let cumulativeProbability = 0

  for (const branch of transition.branches) {
    cumulativeProbability += branch.probability

    if (randomValue <= cumulativeProbability) {
      return {
        destination: branch.destination,
        explanation: branch.explanation,
      }
    }
  }

  const fallbackBranch = transition.branches[transition.branches.length - 1]
  return {
    destination: fallbackBranch.destination,
    explanation: fallbackBranch.explanation,
  }
}

export function expectedTurnsForDestination(tile: TileId, action: Action, destination: TileId, goalTile?: TileId) {
  const probability = transitionForAction(tile, action, goalTile).outcomes.reduce((total, outcome) => {
    if (outcome.destination !== destination) {
      return total
    }

    return total + outcome.probability
  }, 0)

  if (probability <= 0) {
    return Number.POSITIVE_INFINITY
  }

  return 1 / probability
}

export function deterministicTransition(tile: TileId, action: Action, goalTile?: TileId): MoveAttempt {
  if (getTileType(tile) === 'T' || (goalTile && tile === goalTile)) {
    return {
      destination: tile,
      explanation: getAbsorbingExplanation(tile, goalTile),
    }
  }

  return attemptMove(tile, action)
}

export function tileClassName(tile: TileId) {
  const tileType = getTileType(tile)
  const { row, column } = parseTile(tile)
  const classes = [`tile-${tileType.toLowerCase()}`]

  const wallClassesByAction: Partial<Record<Action, string>> = {
    up: 'wall-top',
    down: 'wall-bottom',
    left: 'wall-left',
    right: 'wall-right',
  }

  for (const action of ACTIONS) {
    const wallClass = wallClassesByAction[action]
    if (wallClass && WALLS.has(`${tile}:${action}`)) {
      classes.push(wallClass)
    }
  }

  if (row === 3) {
    classes.push('edge-top')
  }

  if (row === 1) {
    classes.push('edge-bottom')
  }

  if (column === 'A') {
    classes.push('edge-left')
  }

  if (column === COLUMNS[COLUMNS.length - 1]) {
    classes.push('edge-right')
  }

  return classes.join(' ')
}
