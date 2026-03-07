import {
  ACTIONS,
  deterministicTransition,
  expectedTurnsForDestination,
  getAllTiles,
  getTileType,
  sampleTransition,
  type Action,
  type TileId,
} from './game'
import {
  DEFAULT_ASTAR_HEURISTIC_EXPRESSION,
  getAStarHeuristicDefinition,
  type AStarHeuristicAnalysis,
} from './astarHeuristic'

export type AlgorithmKind = 'astar' | 'mcts'

export type AStarFramePhase =
  | 'init'
  | 'loop'
  | 'expand'
  | 'consider-neighbor'
  | 'update-neighbor'
  | 'update-queue'
  | 'skip-neighbor'
  | 'goal-found'
  | 'finished'

export type AStarFrame = {
  kind: 'astar'
  step: number
  phase: AStarFramePhase
  message: string
  deterministicAssumption: string
  current: TileId | null
  action: Action | null
  neighbor: TileId | null
  openSet: TileId[]
  closedSet: TileId[]
  pathPreview: TileId[]
  reachedGoal: boolean
  scoreRows: Array<{
    tile: TileId
    g: number | null
    h: number
    f: number | null
    parent: TileId | null
    actionFromParent: Action | null
  }>
  activeTreeNodeId: number | null
  tree: AStarTreeNodeSnapshot[]
  rejectedChild: {
    tile: TileId
    parentTreeNodeId: number
    actionFromParent: Action
    g: number
    h: number
    f: number
    reason: 'already-closed' | 'not-better'
  } | null
}

export type AStarResult = {
  frames: AStarFrame[]
  path: TileId[]
  actions: Action[]
  reachedGoal: boolean
  heuristicAnalysis: AStarHeuristicAnalysis
}

export type MCTSFramePhase =
  | 'init'
  | 'loop'
  | 'selection'
  | 'expansion'
  | 'expansion-else'
  | 'rollout'
  | 'backprop'
  | 'iteration-end'
  | 'finished'

export type MCTSTreeNodeSnapshot = {
  id: number
  tile: TileId
  parentId: number | null
  actionFromParent: Action | null
  visits: number
  valueSum: number
  depth: number
  parentVisits: number | null
  ucb: number | null
}

export type AStarTreeNodeStatus = 'active' | 'queued' | 'pruned'

export type AStarTreeNodeSnapshot = {
  id: number
  tile: TileId
  parentId: number | null
  actionFromParent: Action | null
  depth: number
  g: number
  h: number
  f: number
  status: AStarTreeNodeStatus
}

export type MCTSFrame = {
  kind: 'mcts'
  step: number
  decisionStep: number
  iteration: number
  explorationConstant: number
  gamma: number
  phase: MCTSFramePhase
  message: string
  activeNodeId: number | null
  selectionPathIds: number[]
  rolloutTiles: TileId[]
  tree: MCTSTreeNodeSnapshot[]
  bestRootAction: Action | null
  bestRootVisits: number
}

export type MCTSResult = {
  frames: MCTSFrame[]
  tree: MCTSTreeNodeSnapshot[]
  selectedAction: Action | null
  selectedActionVisits: number
  nextTile: TileId
  transitionExplanation: string | null
  terminated: boolean
  terminationReason: 'goal' | 'trap' | 'no-action' | null
}

type AStarConfig = {
  start: TileId
  goal: TileId
  heuristicExpression?: string
}

type MCTSConfig = {
  start: TileId
  goal: TileId
  decisionStep?: number
  iterations: number
  explorationConstant: number
  gamma: number
  rolloutHorizon: number
  goalReward?: number
  trapReward?: number
  stuckReward?: number
  normalReward?: number
  rng?: () => number
}

type InternalMCTSNode = {
  id: number
  tile: TileId
  parentId: number | null
  actionFromParent: Action | null
  depth: number
  visits: number
  valueSum: number
  untriedActions: Action[]
  childrenByAction: Partial<Record<Action, number>>
}

type InternalAStarTreeNode = {
  id: number
  tile: TileId
  parentId: number | null
  actionFromParent: Action | null
  depth: number
  g: number
  h: number
  f: number
}

function reconstructPath(cameFrom: Partial<Record<TileId, { prev: TileId; action: Action }>>, end: TileId) {
  const tiles: TileId[] = [end]
  const actions: Action[] = []
  let cursor = end

  while (cameFrom[cursor]) {
    const entry = cameFrom[cursor]
    if (!entry) {
      break
    }
    tiles.unshift(entry.prev)
    actions.unshift(entry.action)
    cursor = entry.prev
  }

  return { tiles, actions }
}

function makeScoreRows(
  gScore: Partial<Record<TileId, number>>,
  fScore: Partial<Record<TileId, number>>,
  cameFrom: Partial<Record<TileId, { prev: TileId; action: Action }>>,
  goal: TileId,
  heuristic: (source: TileId, target: TileId) => number,
) {
  return getAllTiles().map((tile) => ({
    tile,
    g: gScore[tile] ?? null,
    h: heuristic(tile, goal),
    f: fScore[tile] ?? null,
    parent: cameFrom[tile]?.prev ?? null,
    actionFromParent: cameFrom[tile]?.action ?? null,
  }))
}

function snapshotAStarTree(
  nodes: Map<number, InternalAStarTreeNode>,
  bestTreeNodeByTile: Partial<Record<TileId, number>>,
  openSet: Set<TileId>,
  activeTreeNodeId: number | null,
): AStarTreeNodeSnapshot[] {
  return [...nodes.values()]
    .map((node) => {
      let status: AStarTreeNodeStatus = 'pruned'
      if (node.id === activeTreeNodeId) {
        status = 'active'
      } else if (bestTreeNodeByTile[node.tile] === node.id && openSet.has(node.tile)) {
        status = 'queued'
      }

      return {
        id: node.id,
        tile: node.tile,
        parentId: node.parentId,
        actionFromParent: node.actionFromParent,
        depth: node.depth,
        g: node.g,
        h: node.h,
        f: node.f,
        status,
      }
    })
    .sort((left, right) => left.id - right.id)
}

export function runAStarDemo(config: AStarConfig): AStarResult {
  const { start, goal } = config
  const deterministicAssumption =
    'A* uses expected turn cost for each action outcome; leaving a stuck tile in a successful direction costs 2 expected turns.'
  const heuristicDefinition = getAStarHeuristicDefinition(
    config.heuristicExpression ?? DEFAULT_ASTAR_HEURISTIC_EXPRESSION,
    goal,
  )
  const heuristic = heuristicDefinition.evaluator

  const openSet = new Set<TileId>([start])
  const closedSet = new Set<TileId>()
  const cameFrom: Partial<Record<TileId, { prev: TileId; action: Action }>> = {}
  const treeNodes = new Map<number, InternalAStarTreeNode>()

  const gScore: Partial<Record<TileId, number>> = { [start]: 0 }
  const fScore: Partial<Record<TileId, number>> = { [start]: heuristic(start, goal) }
  const rootTreeNode: InternalAStarTreeNode = {
    id: 0,
    tile: start,
    parentId: null,
    actionFromParent: null,
    depth: 0,
    g: 0,
    h: heuristic(start, goal),
    f: heuristic(start, goal),
  }
  treeNodes.set(rootTreeNode.id, rootTreeNode)
  const bestTreeNodeByTile: Partial<Record<TileId, number>> = {
    [start]: rootTreeNode.id,
  }
  let nextTreeNodeId = 1

  const frames: AStarFrame[] = []
  let step = 0
  let reachedGoal = false
  let finalPath: TileId[] = [start]
  let finalActions: Action[] = []

  const pushFrame = (payload: Omit<AStarFrame, 'kind' | 'step' | 'deterministicAssumption' | 'tree'>) => {
    frames.push({
      kind: 'astar',
      step: step++,
      deterministicAssumption,
      tree: snapshotAStarTree(treeNodes, bestTreeNodeByTile, openSet, payload.activeTreeNodeId),
      ...payload,
    })
  }

  pushFrame({
    phase: 'init',
    message: `Initialized A* from ${start} to ${goal}.`,
    current: start,
    action: null,
    neighbor: null,
    openSet: [...openSet],
    closedSet: [...closedSet],
    pathPreview: [start],
    reachedGoal: false,
    scoreRows: makeScoreRows(gScore, fScore, cameFrom, goal, heuristic),
    activeTreeNodeId: rootTreeNode.id,
    rejectedChild: null,
  })

  while (openSet.size > 0) {
    pushFrame({
      phase: 'loop',
      message: `Loop: openSet has ${openSet.size} node(s), choosing next to expand.`,
      current: null,
      action: null,
      neighbor: null,
      openSet: [...openSet],
      closedSet: [...closedSet],
      pathPreview: finalPath,
      reachedGoal: false,
      scoreRows: makeScoreRows(gScore, fScore, cameFrom, goal, heuristic),
      activeTreeNodeId: null,
      rejectedChild: null,
    })
    const current = [...openSet].sort((left, right) => {
      const fDiff = (fScore[left] ?? Number.POSITIVE_INFINITY) - (fScore[right] ?? Number.POSITIVE_INFINITY)
      if (fDiff !== 0) {
        return fDiff
      }
      return left.localeCompare(right)
    })[0]
    const currentTreeNodeId = bestTreeNodeByTile[current] ?? rootTreeNode.id

    if (current === goal) {
      reachedGoal = true
      const reconstructed = reconstructPath(cameFrom, current)
      finalPath = reconstructed.tiles
      finalActions = reconstructed.actions

      pushFrame({
        phase: 'goal-found',
        message: `Goal ${goal} reached. Final plan uses ${finalPath.length - 1} actions with expected cost ${gScore[current] ?? 0}.`,
        current,
        action: null,
        neighbor: null,
        openSet: [...openSet],
        closedSet: [...closedSet],
        pathPreview: finalPath,
        reachedGoal: true,
        scoreRows: makeScoreRows(gScore, fScore, cameFrom, goal, heuristic),
        activeTreeNodeId: currentTreeNodeId,
        rejectedChild: null,
      })
      break
    }

    openSet.delete(current)
    closedSet.add(current)

    const currentPath = reconstructPath(cameFrom, current).tiles
    pushFrame({
      phase: 'expand',
      message: `Expanding ${current} from the frontier.`,
      current,
      action: null,
      neighbor: null,
      openSet: [...openSet],
      closedSet: [...closedSet],
      pathPreview: currentPath,
      reachedGoal: false,
      scoreRows: makeScoreRows(gScore, fScore, cameFrom, goal, heuristic),
      activeTreeNodeId: currentTreeNodeId,
      rejectedChild: null,
    })

    for (const action of ACTIONS) {
      const neighbor = deterministicTransition(current, action, goal).destination
      const stepCost = expectedTurnsForDestination(current, action, neighbor, goal)
      const tentativeG = (gScore[current] ?? Number.POSITIVE_INFINITY) + stepCost

      pushFrame({
        phase: 'consider-neighbor',
        message: `Considering ${action} from ${current} to ${neighbor} with expected step cost ${stepCost}.`,
        current,
        action,
        neighbor,
        openSet: [...openSet],
        closedSet: [...closedSet],
        pathPreview: currentPath,
        reachedGoal: false,
        scoreRows: makeScoreRows(gScore, fScore, cameFrom, goal, heuristic),
        activeTreeNodeId: currentTreeNodeId,
        rejectedChild: null,
      })

      if (closedSet.has(neighbor) && tentativeG >= (gScore[neighbor] ?? Number.POSITIVE_INFINITY)) {
        pushFrame({
          phase: 'skip-neighbor',
          message: `Skipped ${neighbor} because no better score was found.`,
          current,
          action,
          neighbor,
          openSet: [...openSet],
          closedSet: [...closedSet],
          pathPreview: currentPath,
          reachedGoal: false,
          scoreRows: makeScoreRows(gScore, fScore, cameFrom, goal, heuristic),
          activeTreeNodeId: currentTreeNodeId,
          rejectedChild: {
            tile: neighbor,
            parentTreeNodeId: currentTreeNodeId,
            actionFromParent: action,
            g: tentativeG,
            h: heuristic(neighbor, goal),
            f: tentativeG + heuristic(neighbor, goal),
            reason: 'already-closed',
          },
        })
        continue
      }

      if (tentativeG < (gScore[neighbor] ?? Number.POSITIVE_INFINITY)) {
        cameFrom[neighbor] = { prev: current, action }
        gScore[neighbor] = tentativeG
        fScore[neighbor] = tentativeG + heuristic(neighbor, goal)
        const childTreeNode: InternalAStarTreeNode = {
          id: nextTreeNodeId++,
          tile: neighbor,
          parentId: currentTreeNodeId,
          actionFromParent: action,
          depth: (treeNodes.get(currentTreeNodeId)?.depth ?? 0) + 1,
          g: tentativeG,
          h: heuristic(neighbor, goal),
          f: tentativeG + heuristic(neighbor, goal),
        }
        treeNodes.set(childTreeNode.id, childTreeNode)
        bestTreeNodeByTile[neighbor] = childTreeNode.id
        openSet.add(neighbor)

        const preview = reconstructPath(cameFrom, neighbor).tiles
        pushFrame({
          phase: 'update-neighbor',
          message: `Updated ${neighbor}: expected g=${tentativeG}, f=${fScore[neighbor]}.`,
          current,
          action,
          neighbor,
          openSet: [...openSet],
          closedSet: [...closedSet],
          pathPreview: preview,
          reachedGoal: false,
          scoreRows: makeScoreRows(gScore, fScore, cameFrom, goal, heuristic),
          activeTreeNodeId: currentTreeNodeId,
          rejectedChild: null,
        })
        pushFrame({
          phase: 'update-queue',
          message: `Kept ${neighbor} in openSet.`,
          current,
          action,
          neighbor,
          openSet: [...openSet],
          closedSet: [...closedSet],
          pathPreview: preview,
          reachedGoal: false,
          scoreRows: makeScoreRows(gScore, fScore, cameFrom, goal, heuristic),
          activeTreeNodeId: currentTreeNodeId,
          rejectedChild: null,
        })
      } else {
        pushFrame({
          phase: 'skip-neighbor',
          message: `Kept previous score for ${neighbor}.`,
          current,
          action,
          neighbor,
          openSet: [...openSet],
          closedSet: [...closedSet],
          pathPreview: currentPath,
          reachedGoal: false,
          scoreRows: makeScoreRows(gScore, fScore, cameFrom, goal, heuristic),
          activeTreeNodeId: currentTreeNodeId,
          rejectedChild: {
            tile: neighbor,
            parentTreeNodeId: currentTreeNodeId,
            actionFromParent: action,
            g: tentativeG,
            h: heuristic(neighbor, goal),
            f: tentativeG + heuristic(neighbor, goal),
            reason: 'not-better',
          },
        })
      }
    }
  }

  if (!reachedGoal) {
    pushFrame({
      phase: 'finished',
      message: `A* ended without reaching ${goal}.`,
      current: null,
      action: null,
      neighbor: null,
      openSet: [...openSet],
      closedSet: [...closedSet],
      pathPreview: finalPath,
      reachedGoal: false,
      scoreRows: makeScoreRows(gScore, fScore, cameFrom, goal, heuristic),
      activeTreeNodeId: bestTreeNodeByTile[start] ?? rootTreeNode.id,
      rejectedChild: null,
    })
  } else {
    pushFrame({
      phase: 'finished',
      message: `A* finished with goal ${goal}.`,
      current: goal,
      action: null,
      neighbor: null,
      openSet: [...openSet],
      closedSet: [...closedSet],
      pathPreview: finalPath,
      reachedGoal: true,
      scoreRows: makeScoreRows(gScore, fScore, cameFrom, goal, heuristic),
      activeTreeNodeId: bestTreeNodeByTile[goal] ?? bestTreeNodeByTile[start] ?? rootTreeNode.id,
      rejectedChild: null,
    })
  }

  return {
    frames,
    path: finalPath,
    actions: finalActions,
    reachedGoal,
    heuristicAnalysis: heuristicDefinition.analysis,
  }
}

function snapshotTree(
  nodes: Map<number, InternalMCTSNode>,
  explorationConstant: number,
): MCTSTreeNodeSnapshot[] {
  return [...nodes.values()]
    .map((node) => {
      const parent = node.parentId === null ? null : nodes.get(node.parentId) ?? null
      const parentVisits = parent ? Math.max(1, parent.visits) : null

      return {
        id: node.id,
        tile: node.tile,
        parentId: node.parentId,
        actionFromParent: node.actionFromParent,
        visits: node.visits,
        valueSum: node.valueSum,
        depth: node.depth,
        parentVisits,
        ucb: parentVisits === null ? null : getUctScore(node, parentVisits, explorationConstant),
      }
    })
    .sort((left, right) => left.id - right.id)
}

function getUctScore(
  child: InternalMCTSNode,
  parentVisits: number,
  explorationConstant: number,
) {
  if (child.visits === 0) {
    return Number.POSITIVE_INFINITY
  }

  const exploitation = child.valueSum / child.visits
  const exploration = explorationConstant * Math.sqrt(Math.log(Math.max(1, parentVisits)) / child.visits)
  return exploitation + exploration
}

function rewardForTile(
  tile: TileId,
  goal: TileId,
  goalReward: number = 10,
  trapReward: number = -1,
  stuckReward: number = 0,
  normalReward: number = 0,
) {
  if (tile === goal) {
    return goalReward
  }

  const tileType = getTileType(tile)
  if (tileType === 'T') {
    return trapReward
  }

  if (tileType === 'S') {
    return stuckReward
  }

  return normalReward
}

function discountedRewardForRollout(
  tiles: TileId[],
  goal: TileId,
  gamma: number,
  goalReward: number = 10,
  trapReward: number = -1,
  stuckReward: number = 0,
  normalReward: number = 0,
) {
  return tiles.reduce(
    (totalReward, tile, stepsAhead) =>
      totalReward +
      Math.pow(gamma, Math.max(0, stepsAhead)) *
        rewardForTile(tile, goal, goalReward, trapReward, stuckReward, normalReward),
    0,
  )
}

function isTerminalTile(tile: TileId, goal: TileId) {
  return tile === goal || getTileType(tile) === 'T'
}

function createRng(defaultRng?: () => number) {
  if (defaultRng) {
    return defaultRng
  }

  let state = 0x12345678
  return () => {
    state = (1664525 * state + 1013904223) >>> 0
    return state / 0x100000000
  }
}

export function runMCTSDemo(config: MCTSConfig): MCTSResult {
  const {
    start,
    goal,
    decisionStep = 0,
    explorationConstant,
    gamma,
    iterations,
    rolloutHorizon,
    goalReward = 10,
    trapReward = -1,
    stuckReward = 0,
    normalReward = 0,
  } = config
  const rng = createRng(config.rng)

  const frames: MCTSFrame[] = []
  const nodes = new Map<number, InternalMCTSNode>()
  const root: InternalMCTSNode = {
    id: 0,
    tile: start,
    parentId: null,
    actionFromParent: null,
    depth: 0,
    visits: 0,
    valueSum: 0,
    untriedActions: [...ACTIONS],
    childrenByAction: {},
  }
  nodes.set(root.id, root)
  let nextNodeId = 1
  let step = 0

  const getBestRootAction = () => {
    let bestAction: Action | null = null
    let bestVisits = -1

    for (const action of ACTIONS) {
      const childId = root.childrenByAction[action]
      if (childId === undefined) {
        continue
      }
      const child = nodes.get(childId)
      if (!child) {
        continue
      }
      if (child.visits > bestVisits) {
        bestVisits = child.visits
        bestAction = action
      }
    }

    return {
      action: bestAction,
      visits: Math.max(0, bestVisits),
    }
  }

  const pushFrame = (
    payload: Omit<
      MCTSFrame,
      | 'kind'
      | 'step'
      | 'decisionStep'
      | 'bestRootAction'
      | 'bestRootVisits'
      | 'tree'
      | 'explorationConstant'
      | 'gamma'
    >,
  ) => {
    const best = getBestRootAction()
    frames.push({
      kind: 'mcts',
      step: step++,
      decisionStep,
      explorationConstant,
      gamma,
      bestRootAction: best.action,
      bestRootVisits: best.visits,
      tree: snapshotTree(nodes, explorationConstant),
      ...payload,
    })
  }

  pushFrame({
    iteration: 0,
    phase: 'init',
    message: `Initialized planning step ${decisionStep + 1} at ${start} with c=${explorationConstant.toFixed(3)}, gamma=${gamma.toFixed(3)}, and horizon=${rolloutHorizon}.`,
    activeNodeId: root.id,
    selectionPathIds: [root.id],
    rolloutTiles: [start],
  })

  if (isTerminalTile(start, goal)) {
    const terminationReason = start === goal ? 'goal' : 'trap'
    pushFrame({
      iteration: 0,
      phase: 'finished',
      message:
        terminationReason === 'goal'
          ? `Planning step ${decisionStep + 1} stops immediately because ${start} is already the goal.`
          : `Planning step ${decisionStep + 1} stops immediately because ${start} is a trap state.`,
      activeNodeId: root.id,
      selectionPathIds: [root.id],
      rolloutTiles: [start],
    })

    return {
      frames,
      tree: snapshotTree(nodes, explorationConstant),
      selectedAction: null,
      selectedActionVisits: 0,
      nextTile: start,
      transitionExplanation: null,
      terminated: true,
      terminationReason,
    }
  }

  for (let iteration = 1; iteration <= iterations; iteration += 1) {
    const selectionPath: number[] = [root.id]
    let current = root

    pushFrame({
      iteration,
      phase: 'loop',
      message: `Iteration ${iteration} of ${iterations}: starting loop.`,
      activeNodeId: root.id,
      selectionPathIds: [root.id],
      rolloutTiles: [start],
    })
    pushFrame({
      iteration,
      phase: 'selection',
      message: `Iteration ${iteration}: starting selection at root.`,
      activeNodeId: current.id,
      selectionPathIds: [...selectionPath],
      rolloutTiles: [current.tile],
    })

    while (current.untriedActions.length === 0) {
      const childIds = ACTIONS.map((action) => current.childrenByAction[action]).filter(
        (childId): childId is number => childId !== undefined,
      )
      if (childIds.length === 0) {
        break
      }

      let bestChild: InternalMCTSNode | null = null
      let bestScore = Number.NEGATIVE_INFINITY
      for (const childId of childIds) {
        const child = nodes.get(childId)
        if (!child) {
          continue
        }
        const score = getUctScore(child, Math.max(1, current.visits), explorationConstant)
        if (score > bestScore) {
          bestScore = score
          bestChild = child
        }
      }

      if (!bestChild) {
        break
      }

      current = bestChild
      selectionPath.push(current.id)
      pushFrame({
        iteration,
        phase: 'selection',
        message: `Selection moved to node ${current.id} (${current.tile}) via UCT.`,
        activeNodeId: current.id,
        selectionPathIds: [...selectionPath],
        rolloutTiles: selectionPath.map((nodeId) => nodes.get(nodeId)?.tile ?? start),
      })
    }

    let expandedThisIteration = false
    if (!isTerminalTile(current.tile, goal) && current.untriedActions.length > 0) {
      const actionIndex = Math.floor(rng() * current.untriedActions.length)
      const [action] = current.untriedActions.splice(actionIndex, 1)
      const nextState = sampleTransition(current.tile, action, rng, goal).destination
      const child: InternalMCTSNode = {
        id: nextNodeId++,
        tile: nextState,
        parentId: current.id,
        actionFromParent: action,
        depth: current.depth + 1,
        visits: 0,
        valueSum: 0,
        untriedActions: isTerminalTile(nextState, goal) ? [] : [...ACTIONS],
        childrenByAction: {},
      }
      nodes.set(child.id, child)
      current.childrenByAction[action] = child.id
      current = child
      selectionPath.push(current.id)

      pushFrame({
        iteration,
        phase: 'expansion',
        message: `Expanded action ${action} to node ${current.id} (${current.tile}).`,
        activeNodeId: current.id,
        selectionPathIds: [...selectionPath],
        rolloutTiles: selectionPath.map((nodeId) => nodes.get(nodeId)?.tile ?? start),
      })
      expandedThisIteration = true
    }

    if (!expandedThisIteration) {
      pushFrame({
        iteration,
        phase: 'expansion-else',
        message: `Leaf has no untried action; skipping expansion, starting rollout from ${current.tile}.`,
        activeNodeId: current.id,
        selectionPathIds: [...selectionPath],
        rolloutTiles: selectionPath.map((nodeId) => nodes.get(nodeId)?.tile ?? start),
      })
    }

    const rolloutTiles: TileId[] = [current.tile]
    let rolloutState = current.tile
    const remainingSteps = Math.max(0, rolloutHorizon - current.depth)
    for (let rolloutStep = 0; rolloutStep < remainingSteps; rolloutStep += 1) {
      if (isTerminalTile(rolloutState, goal)) {
        break
      }
      const action = ACTIONS[Math.floor(rng() * ACTIONS.length)]
      rolloutState = sampleTransition(rolloutState, action, rng, goal).destination
      rolloutTiles.push(rolloutState)
    }

    pushFrame({
      iteration,
      phase: 'rollout',
      message: `Rollout visited ${rolloutTiles.length} state(s).`,
      activeNodeId: current.id,
      selectionPathIds: [...selectionPath],
      rolloutTiles,
    })

    let reward = discountedRewardForRollout(
      rolloutTiles,
      goal,
      gamma,
      goalReward,
      trapReward,
      stuckReward,
      normalReward,
    )
    for (let pathIndex = selectionPath.length - 1; pathIndex >= 0; pathIndex -= 1) {
      const nodeId = selectionPath[pathIndex]
      const node = nodes.get(nodeId)
      if (!node) {
        continue
      }
      node.visits += 1
      node.valueSum += reward

      pushFrame({
        iteration,
        phase: 'backprop',
        message: `Backprop updated node ${node.id}: visits=${node.visits}, W=${node.valueSum.toFixed(2)}.`,
        activeNodeId: node.id,
        selectionPathIds: [...selectionPath],
        rolloutTiles,
      })

      reward *= gamma
    }

    pushFrame({
      iteration,
      phase: 'iteration-end',
      message: `Iteration ${iteration} complete.`,
      activeNodeId: root.id,
      selectionPathIds: [...selectionPath],
      rolloutTiles,
    })
  }

  const best = getBestRootAction()
  let nextTile = start
  let transitionExplanation: string | null = null
  let terminated = false
  let terminationReason: MCTSResult['terminationReason'] = null

  if (best.action) {
    const executedTransition = sampleTransition(start, best.action, rng, goal)
    nextTile = executedTransition.destination
    transitionExplanation = executedTransition.explanation
    terminated = isTerminalTile(nextTile, goal)
    if (terminated) {
      terminationReason = nextTile === goal ? 'goal' : 'trap'
    }
  } else {
    terminationReason = 'no-action'
  }

  pushFrame({
    iteration: iterations,
    phase: 'finished',
    message: best.action
      ? `Planning step ${decisionStep + 1} finished ${iterations} iterations and selected ${best.action}, leading to ${nextTile}.`
      : `Planning step ${decisionStep + 1} finished ${iterations} iterations without selecting an action.`,
    activeNodeId: root.id,
    selectionPathIds: [root.id],
    rolloutTiles: best.action ? [start, nextTile] : [start],
  })

  return {
    frames,
    tree: snapshotTree(nodes, explorationConstant),
    selectedAction: best.action,
    selectedActionVisits: best.visits,
    nextTile,
    transitionExplanation,
    terminated,
    terminationReason,
  }
}
