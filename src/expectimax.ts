import {
  ACTIONS,
  getTileType,
  slipActionOutcomes,
  type Action,
  type TileId,
} from './game'
import { DEFAULT_VI_REWARDS, type RewardConfig } from './valueIteration'

export type ExpectimaxNode =
  | ExpectimaxMaxNode
  | ExpectimaxChanceNode
  | ExpectimaxLeafNode

export type ExpectimaxMaxNode = {
  id: string
  kind: 'max'
  tile: TileId
  depth: number
  horizonRemaining: number
  value: number
  bestAction: Action | null
  children: Array<{ action: Action; node: ExpectimaxNode; q: number }>
  parentId: string | null
}

export type ExpectimaxChanceNode = {
  id: string
  kind: 'chance'
  tile: TileId
  action: Action
  depth: number
  horizonRemaining: number
  value: number
  children: Array<{ probability: number; reward: number; node: ExpectimaxNode }>
  parentId: string | null
}

export type ExpectimaxLeafNode = {
  id: string
  kind: 'leaf'
  tile: TileId
  depth: number
  horizonRemaining: number
  value: number
  reason: 'terminal' | 'horizon'
  parentId: string | null
}

export type ExpectimaxResult = {
  root: ExpectimaxMaxNode
  nodes: ExpectimaxNode[]
  goalTile: TileId
  horizon: number
  gamma: number
  rewards: RewardConfig
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

export function buildExpectimaxTree(
  start: TileId,
  goalTile: TileId,
  horizon: number,
  gamma: number,
  rewards: RewardConfig = DEFAULT_VI_REWARDS,
  actionSuccessProb = 1,
): ExpectimaxResult {
  const allNodes: ExpectimaxNode[] = []
  let idCounter = 0
  const nextId = () => `n${idCounter++}`

  function build(
    tile: TileId,
    depth: number,
    horizonRemaining: number,
    parentId: string | null,
  ): ExpectimaxNode {
    if (isTerminal(tile, goalTile)) {
      const leaf: ExpectimaxLeafNode = {
        id: nextId(),
        kind: 'leaf',
        tile,
        depth,
        horizonRemaining,
        value: tileReward(tile, goalTile, rewards),
        reason: 'terminal',
        parentId,
      }
      allNodes.push(leaf)
      return leaf
    }

    if (horizonRemaining === 0) {
      const leaf: ExpectimaxLeafNode = {
        id: nextId(),
        kind: 'leaf',
        tile,
        depth,
        horizonRemaining,
        value: 0,
        reason: 'horizon',
        parentId,
      }
      allNodes.push(leaf)
      return leaf
    }

    const maxId = nextId()
    const maxNode: ExpectimaxMaxNode = {
      id: maxId,
      kind: 'max',
      tile,
      depth,
      horizonRemaining,
      value: Number.NEGATIVE_INFINITY,
      bestAction: null,
      children: [],
      parentId,
    }
    allNodes.push(maxNode)

    for (const action of ACTIONS) {
      const outcomes = slipActionOutcomes(tile, action, actionSuccessProb, goalTile)
      const chanceId = nextId()
      const chanceNode: ExpectimaxChanceNode = {
        id: chanceId,
        kind: 'chance',
        tile,
        action,
        depth: depth + 1,
        horizonRemaining,
        value: 0,
        children: [],
        parentId: maxId,
      }
      allNodes.push(chanceNode)

      let expected = 0
      for (const outcome of outcomes) {
        const child = build(outcome.destination, depth + 2, horizonRemaining - 1, chanceId)
        const destTerminal = isTerminal(outcome.destination, goalTile)
        const stepReward = destTerminal
          ? rewards.step
          : rewards.step + tileReward(outcome.destination, goalTile, rewards)
        chanceNode.children.push({
          probability: outcome.probability,
          reward: stepReward,
          node: child,
        })
        expected += outcome.probability * (stepReward + gamma * child.value)
      }
      chanceNode.value = expected
      maxNode.children.push({ action, node: chanceNode, q: expected })
      if (expected > maxNode.value) {
        maxNode.value = expected
        maxNode.bestAction = action
      }
    }

    return maxNode
  }

  const root = build(start, 0, horizon, null) as ExpectimaxMaxNode
  if (root.kind !== 'max') {
    const wrapped: ExpectimaxMaxNode = {
      id: nextId(),
      kind: 'max',
      tile: start,
      depth: 0,
      horizonRemaining: horizon,
      value: root.value,
      bestAction: null,
      children: [],
      parentId: null,
    }
    allNodes.push(wrapped)
    return { root: wrapped, nodes: allNodes, goalTile, horizon, gamma, rewards }
  }

  return { root, nodes: allNodes, goalTile, horizon, gamma, rewards }
}
