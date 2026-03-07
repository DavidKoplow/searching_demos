import { Fragment, type CSSProperties, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import dagre from 'dagre'
import {
  Background,
  Controls,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import {
  BOARD_ROWS,
  COLUMNS,
  type Action,
  type TileId,
  describeAction,
  getAllTiles,
  getRandomStartTile,
  getTileType,
  sampleTransition,
  tileClassName,
} from './game'
import {
  runAStarDemo,
  runMCTSDemo,
  type AStarFrame,
  type AStarTreeNodeSnapshot,
  type AlgorithmKind,
  type MCTSFrame,
  type MCTSTreeNodeSnapshot,
} from './algorithms'
import {
  DEFAULT_ASTAR_HEURISTIC_EXPRESSION,
  analyzeAStarHeuristic,
  type AStarAdmissibilityViolation,
  type AStarConsistencyViolation,
} from './astarHeuristic'

type ExecutionFrame = {
  kind: 'execution'
  step: number
  phase: 'execution'
  message: string
  tile: TileId
  from: TileId | null
  action: Action | null
  path: TileId[]
  mctsHistoryIndex: number | null
}

type MCTSGraphHistoryEntry = {
  decisionStep: number
  rootTile: TileId
  resultingTile: TileId
  incomingAction: Action | null
  selectedAction: Action | null
  bestRootVisits: number
  frame: MCTSFrame
  rolloutPathByNodeId: Map<number, TileId[]>
}

type TokenAnimation = {
  key: number
  className: 'avatar-slide' | 'avatar-bump'
  style: CSSProperties
}

type DemoFrame = AStarFrame | MCTSFrame | ExecutionFrame

type AlgorithmTraceLine = {
  id: string
  text: string
  indent?: number
}

type AlgorithmTraceValue = {
  label: string
  value: string
}

type AlgorithmTraceModel = {
  algorithmLabel: string
  frameMessage: string
  activeLineId: string
  /** When active line is the body of a conditional, the conditional line id (shown blue; child shown lighter). */
  activeParentLineId?: string
  lines: AlgorithmTraceLine[]
  values: AlgorithmTraceValue[]
}

type AlgorithmTraceControl = {
  label: string
  value: string
  onPrevious: () => void
  onNext: () => void
  canPrevious: boolean
  canNext: boolean
}

const ASTAR_TRACE_LINES: AlgorithmTraceLine[] = [
  { id: 'init', text: 'openSet <- { start }' },
  { id: 'loop', text: 'while openSet is not empty:' },
  { id: 'expand', text: 'current <- node in openSet with smallest f = g + h', indent: 1 },
  { id: 'goal-check', text: 'if current is goal:', indent: 1 },
  { id: 'goal', text: 'return path(current)', indent: 2 },
  { id: 'goal-else', text: 'else:', indent: 1 },
  { id: 'consider', text: 'for each neighbor of current:', indent: 2 },
  { id: 'skip-check', text: 'if neighbor is closed or not better:', indent: 3 },
  { id: 'skip', text: 'continue', indent: 4 },
  { id: 'skip-else', text: 'else:', indent: 3 },
  { id: 'update', text: 'update parent, g, h, and f', indent: 4 },
  { id: 'update-queue', text: 'keep neighbor in openSet', indent: 4 },
  { id: 'finish', text: 'return failure if no path is found' },
]

const MCTS_TRACE_LINES: AlgorithmTraceLine[] = [
  { id: 'init', text: 'root <- current state' },
  { id: 'loop', text: 'for iteration = 1 to planningBudget:' },
  { id: 'selection', text: 'select children by UCB until a leaf is reached', indent: 1 },
  { id: 'expansion-check', text: 'if leaf has an untried action:', indent: 1 },
  { id: 'expansion', text: 'expand one child', indent: 2 },
  { id: 'expansion-else', text: 'else:', indent: 1 },
  { id: 'rollout', text: 'simulate a rollout from the new state', indent: 2 },
  { id: 'backprop', text: 'backpropagate reward through visited nodes', indent: 1 },
  { id: 'iteration-end', text: 'update best root action from visit counts', indent: 1 },
  { id: 'finished', text: 'return root action with the most visits' },
]

function isAStarFrame(frame: DemoFrame | null): frame is AStarFrame {
  return frame?.kind === 'astar'
}

function isMCTSFrame(frame: DemoFrame | null): frame is MCTSFrame {
  return frame?.kind === 'mcts'
}

function isExecutionFrame(frame: DemoFrame | null): frame is ExecutionFrame {
  return frame?.kind === 'execution'
}

function getTileCoordinates(tile: TileId) {
  const rowValue = tile[0] as `${(typeof BOARD_ROWS)[number]}`
  const columnValue = tile[1] as (typeof COLUMNS)[number]

  return {
    rowIndex: BOARD_ROWS.length - Number(rowValue),
    columnIndex: COLUMNS.indexOf(columnValue),
  }
}

function getTokenPosition(tile: TileId) {
  const { rowIndex, columnIndex } = getTileCoordinates(tile)

  return {
    x: `${((columnIndex + 0.5) / COLUMNS.length) * 100}%`,
    y: `${((rowIndex + 0.5) / BOARD_ROWS.length) * 100}%`,
  }
}

function getTokenBumpOffset(action: Action) {
  const horizontal = `${((100 / COLUMNS.length) * 0.22).toFixed(3)}%`
  const vertical = `${((100 / BOARD_ROWS.length) * 0.18).toFixed(3)}%`
  const offsets: Record<Action, { x: string; y: string }> = {
    up: { x: '0%', y: `-${vertical}` },
    down: { x: '0%', y: vertical },
    left: { x: `-${horizontal}`, y: '0%' },
    right: { x: horizontal, y: '0%' },
  }

  return offsets[action]
}

function getBoardPoint(tile: TileId) {
  const { rowIndex, columnIndex } = getTileCoordinates(tile)

  return {
    x: columnIndex + 0.5,
    y: rowIndex + 0.5,
  }
}

function getMiniBoardMetrics(height: number) {
  const margin = Math.max(6, height * 0.0875)
  const cellSize = (height - margin * 2) / BOARD_ROWS.length
  const width = margin * 2 + cellSize * COLUMNS.length

  return {
    margin,
    cellSize,
    width,
    height,
  }
}

function buildPathSegments(path: TileId[]) {
  return path.slice(1).map((tile, index) => ({
    start: path[index],
    end: tile,
    startPoint: getBoardPoint(path[index]),
    endPoint: getBoardPoint(tile),
  }))
}

function getMiniPoint(tile: TileId, size = 80) {
  const { rowIndex, columnIndex } = getTileCoordinates(tile)
  const { margin, cellSize } = getMiniBoardMetrics(size)

  return {
    x: margin + columnIndex * cellSize + cellSize / 2,
    y: margin + rowIndex * cellSize + cellSize / 2,
  }
}

function getMiniTileFill(tile: TileId, goalTile?: TileId) {
  if (goalTile && tile === goalTile) {
    return '#dcfce7'
  }

  const tileType = getTileType(tile)

  if (tileType === 'S') {
    return '#cfe3ff'
  }

  if (tileType === 'T') {
    return '#ffd9d6'
  }

  return '#faf9f5'
}

function buildExecutionFrames(path: TileId[], actions: Action[]) {
  if (path.length === 0) {
    return []
  }

  const frames: ExecutionFrame[] = [
    {
      kind: 'execution',
      step: 0,
      phase: 'execution',
      message: `Execution begins at ${path[0]}.`,
      tile: path[0],
      from: null,
      action: null,
      path: [path[0]],
      mctsHistoryIndex: null,
    },
  ]

  for (let index = 1; index < path.length; index += 1) {
    frames.push({
      kind: 'execution',
      step: index,
      phase: 'execution',
      message: `Execution step ${index}: ${path[index - 1]} --${actions[index - 1] ?? 'up'}--> ${path[index]}.`,
      tile: path[index],
      from: path[index - 1],
      action: actions[index - 1] ?? null,
      path: path.slice(0, index + 1),
      mctsHistoryIndex: null,
    })
  }

  return frames
}

function buildExecutionFrame({
  timelineStep,
  tile,
  from,
  action,
  path,
  message,
  mctsHistoryIndex,
}: {
  timelineStep: number
  tile: TileId
  from: TileId | null
  action: Action | null
  path: TileId[]
  message: string
  mctsHistoryIndex: number | null
}) {
  return {
    kind: 'execution' as const,
    step: timelineStep,
    phase: 'execution' as const,
    message,
    tile,
    from,
    action,
    path,
    mctsHistoryIndex,
  }
}

function MiniBoardState({
  tile,
  path = [tile],
  emphasisTile,
  simulationPath = [],
  simulationIndicatorToTile = null,
  goalTile,
  size = 88,
  className = '',
}: {
  tile: TileId
  path?: TileId[]
  emphasisTile?: TileId
  simulationPath?: TileId[]
  simulationIndicatorToTile?: TileId | null
  goalTile?: TileId
  size?: number
  className?: string
}) {
  const { margin, cellSize, width, height } = getMiniBoardMetrics(size)
  const focusTile = emphasisTile ?? tile
  const segments = buildPathSegments(path)
  const simulationSegments = buildPathSegments(simulationPath)

  return (
    <svg
      className={className ? `mini-board-state ${className}` : 'mini-board-state'}
      viewBox={`0 0 ${width} ${height}`}
      style={{ width: `${width}px`, height: `${height}px` }}
      aria-hidden="true"
    >
      <rect x="1" y="1" width={width - 2} height={height - 2} rx="10" className="mini-border" />
      {BOARD_ROWS.map((row: (typeof BOARD_ROWS)[number]) =>
        COLUMNS.map((column: (typeof COLUMNS)[number]) => {
          const tileId = `${row}${column}` as TileId
          const { rowIndex, columnIndex } = getTileCoordinates(tileId)
          const x = margin + columnIndex * cellSize
          const y = margin + rowIndex * cellSize
          const isFocus = tileId === focusTile

          return (
            <Fragment key={tileId}>
              <rect
                x={x}
                y={y}
                width={cellSize}
                height={cellSize}
                rx="4"
                fill={getMiniTileFill(tileId, goalTile)}
                stroke="#90a0ba"
                strokeWidth="1"
              />
              <text
                x={x + cellSize / 2}
                y={y + cellSize / 2 + 3}
                textAnchor="middle"
                className={`mini-board-label ${isFocus ? 'mini-board-label-focus' : ''}`}
              >
                {tileId}
              </text>
            </Fragment>
          )
        }),
      )}
      {segments.map((segment, index) => {
        const start = getMiniPoint(segment.start, size)
        const end = getMiniPoint(segment.end, size)
        const progress = (index + 1) / Math.max(1, segments.length)

        return (
          <line
            key={`${segment.start}-${segment.end}-${index}`}
            x1={start.x}
            y1={start.y}
            x2={end.x}
            y2={end.y}
            className="mini-path-line"
            style={{
              opacity: String(0.35 + progress * 0.6),
              strokeWidth: String(1.6 + progress * 1.6),
            }}
          />
        )
      })}
      {path.map((pathTile, index) => {
        const point = getMiniPoint(pathTile, size)
        const progress = (index + 1) / Math.max(1, path.length)

        return (
          <circle
            key={`${pathTile}-${index}`}
            cx={point.x}
            cy={point.y}
            r={2.2 + progress * 1.8}
            className="mini-board-path-dot"
            style={{ opacity: String(0.3 + progress * 0.7) }}
          />
        )
      })}
      {simulationSegments.map((segment, index) => {
        const start = getMiniPoint(segment.start, size)
        const end = getMiniPoint(segment.end, size)
        const progress = (index + 1) / Math.max(1, simulationSegments.length)

        return (
          <line
            key={`simulation-${segment.start}-${segment.end}-${index}`}
            x1={start.x}
            y1={start.y}
            x2={end.x}
            y2={end.y}
            className="mini-board-sim-path-line"
            style={{
              opacity: String(0.24 + progress * 0.68),
              strokeWidth: String(1.8 + progress * 1.8),
            }}
          />
        )
      })}
      {simulationPath.map((pathTile, index) => {
        const point = getMiniPoint(pathTile, size)
        const progress = (index + 1) / Math.max(1, simulationPath.length)

        return (
          <circle
            key={`simulation-dot-${pathTile}-${index}`}
            cx={point.x}
            cy={point.y}
            r={2.1 + progress * 1.7}
            className="mini-board-sim-path-dot"
            style={{ opacity: String(0.28 + progress * 0.68) }}
          />
        )
      })}
      {simulationIndicatorToTile &&
        (() => {
          const start = getMiniPoint(tile, size)
          const end = getMiniPoint(simulationIndicatorToTile, size)
          const isStationary = tile === simulationIndicatorToTile

          return (
            <>
              {!isStationary && (
                <line
                  x1={start.x}
                  y1={start.y}
                  x2={end.x}
                  y2={end.y}
                  className="mini-board-sim-indicator-line"
                />
              )}
              <circle
                cx={end.x}
                cy={end.y}
                r={cellSize * 0.15}
                className="mini-board-sim-indicator-dot"
              />
            </>
          )
        })()}
      {(() => {
        const point = getMiniPoint(focusTile, size)
        return (
          <>
            <circle cx={point.x} cy={point.y} r={cellSize * 0.24} className="mini-board-focus-ring" />
            <circle cx={point.x} cy={point.y} r={cellSize * 0.11} className="mini-board-focus-dot" />
          </>
        )
      })()}
    </svg>
  )
}

type MCTSFlowNodeData = {
  mode: 'mcts'
  snapshot: MCTSTreeNodeSnapshot
  active: boolean
  transitionPath: TileId[]
  simulationPath: TileId[]
  simulationIndicatorToTile: TileId | null
  goalTile: TileId
  boardSize: number
  compact: boolean
}

type AStarRenderNodeStatus = AStarTreeNodeSnapshot['status'] | 'rejected'

type AStarRenderNodeSnapshot = {
  id: string
  tile: TileId
  parentId: string | null
  actionFromParent: Action | null
  depth: number
  g: number
  h: number
  f: number
  status: AStarRenderNodeStatus
  rejectionReason?: 'already-closed' | 'not-better'
}

type AStarFlowNodeData = {
  mode: 'astar'
  snapshot: AStarRenderNodeSnapshot
  previewPath: TileId[]
  goalTile: TileId
}

type SearchFlowNodeData = MCTSFlowNodeData | AStarFlowNodeData
type SearchFlowNodeType = Node<SearchFlowNodeData, 'searchNode'>

type MCTSTreeLayoutConfig = {
  nodeWidth: number
  nodeHeight: number
  boardSize: number
  nodesep: number
  ranksep: number
  margin: number
  minZoom: number
  fitViewPadding: number
  edgeType: 'straight'
  edgePathOffset: number
  showLabels: boolean
  compact: boolean
  activeEdgeWidth: number
  inactiveEdgeWidth: number
  markerSize: number
}

type MCTSTreeDiagram = {
  nodes: SearchFlowNodeType[]
  edges: Edge[]
  minZoom: number
  fitViewPadding: number
}

const SEARCH_NODE_WIDTH = 256
const SEARCH_NODE_HEIGHT = 188
const SEARCH_NODE_BOARD_SIZE = 110

function getMCTSTreeLayoutConfig(nodeCount: number): MCTSTreeLayoutConfig {
  if (nodeCount >= 52) {
    return {
      nodeWidth: 198,
      nodeHeight: 146,
      boardSize: 78,
      nodesep: 20,
      ranksep: 40,
      margin: 16,
      minZoom: 0.08,
      fitViewPadding: 0.07,
      edgeType: 'straight',
      edgePathOffset: 24,
      showLabels: false,
      compact: true,
      activeEdgeWidth: 3,
      inactiveEdgeWidth: 2.4,
      markerSize: 14,
    }
  }

  if (nodeCount >= 26) {
    return {
      nodeWidth: 224,
      nodeHeight: 166,
      boardSize: 92,
      nodesep: 28,
      ranksep: 54,
      margin: 20,
      minZoom: 0.12,
      fitViewPadding: 0.1,
      edgeType: 'straight',
      edgePathOffset: 28,
      showLabels: true,
      compact: true,
      activeEdgeWidth: 3,
      inactiveEdgeWidth: 2.4,
      markerSize: 16,
    }
  }

  return {
    nodeWidth: SEARCH_NODE_WIDTH,
    nodeHeight: SEARCH_NODE_HEIGHT,
    boardSize: SEARCH_NODE_BOARD_SIZE,
    nodesep: 46,
    ranksep: 86,
    margin: 28,
    minZoom: 0.18,
    fitViewPadding: 0.16,
    edgeType: 'straight',
    edgePathOffset: 32,
    showLabels: true,
    compact: false,
    activeEdgeWidth: 2.8,
    inactiveEdgeWidth: 2.2,
    markerSize: 18,
  }
}

function clampPlaybackMs(value: number) {
  if (!Number.isFinite(value)) {
    return 1
  }

  return Math.min(5000, Math.max(1, Math.round(value)))
}

function formatUcb(ucb: number | null) {
  if (ucb === null) {
    return 'root'
  }

  if (!Number.isFinite(ucb)) {
    return 'inf'
  }

  return ucb.toFixed(2)
}

function formatGraphNumber(value: number) {
  return Number.isInteger(value) ? String(value) : value.toFixed(2)
}

function formatAStarScore(value: number | null) {
  return value === null ? '-' : formatGraphNumber(value)
}

function formatHeuristicValue(value: number) {
  return formatGraphNumber(value)
}

function formatInputNumber(value: number) {
  return Number.isInteger(value) ? String(value) : value.toFixed(2)
}

function formatAdmissibilityViolation(violation: AStarAdmissibilityViolation) {
  return `${violation.tile}: h=${formatHeuristicValue(violation.heuristic)} but the exact cost to the goal is ${formatHeuristicValue(violation.trueCost)}.`
}

function formatConsistencyViolation(violation: AStarConsistencyViolation) {
  const rightHandSide = violation.stepCost + violation.neighborHeuristic
  return `${violation.tile} --${violation.action}--> ${violation.neighbor}: h=${formatHeuristicValue(violation.heuristic)} but c+h'=${formatHeuristicValue(rightHandSide)} (step cost ${formatHeuristicValue(violation.stepCost)}, neighbor h ${formatHeuristicValue(violation.neighborHeuristic)}).`
}

function SearchFlowNodeCard({ data }: NodeProps<SearchFlowNodeType>) {
  if (data.mode === 'mcts') {
    const node = data.snapshot

    return (
      <div
        className={`mcts-flow-node ${data.compact ? 'mcts-flow-node-compact' : ''} ${data.active ? 'mcts-flow-node-active' : ''}`}
      >
        <Handle type="target" position={Position.Top} className="mcts-flow-handle" />
        <div className="tree-node-header">
          <span className="tree-node-id">#{node.id}</span>
          <span className="tree-node-tile">({node.tile})</span>
        </div>
        <MiniBoardState
          tile={node.tile}
          path={data.transitionPath}
          emphasisTile={node.tile}
          simulationPath={data.simulationPath}
          simulationIndicatorToTile={data.simulationIndicatorToTile}
          goalTile={data.goalTile}
          size={data.boardSize}
        />
        <div className="tree-node-stats">
          <span>n={node.visits}</span>
          <span>W={formatGraphNumber(node.valueSum)}</span>
          <span>UCB={formatUcb(node.ucb)}</span>
        </div>
        <Handle type="source" position={Position.Bottom} className="mcts-flow-handle" />
      </div>
    )
  }

  return (
    <div
      className={`mcts-flow-node astar-flow-node astar-flow-node-${data.snapshot.status} ${data.snapshot.status === 'active' ? 'mcts-flow-node-active' : ''}`}
    >
      <Handle type="target" position={Position.Top} className="mcts-flow-handle" />
      <div className="tree-node-header">
        <span className="tree-node-id">{data.snapshot.tile}</span>
      </div>
      <MiniBoardState
        tile={data.snapshot.tile}
        path={data.previewPath}
        emphasisTile={data.snapshot.tile}
        goalTile={data.goalTile}
        size={SEARCH_NODE_BOARD_SIZE}
      />
      <div className="tree-node-stats">
        <span>g={formatAStarScore(data.snapshot.g)}</span>
        <span>h={formatGraphNumber(data.snapshot.h)}</span>
        <span>f={formatAStarScore(data.snapshot.f)}</span>
        <span>{data.snapshot.status}</span>
      </div>
      <Handle type="source" position={Position.Bottom} className="mcts-flow-handle" />
    </div>
  )
}

function buildMCTSTreeDiagram(
  frame: MCTSFrame,
  rolloutPathByNodeId: Map<number, TileId[]>,
  goalTile: TileId,
  graphIdPrefix = `mcts-${frame.decisionStep}`,
): MCTSTreeDiagram {
  const { tree, activeNodeId } = frame
  const graph = new dagre.graphlib.Graph()
  graph.setDefaultEdgeLabel(() => ({}))
  const layout = getMCTSTreeLayoutConfig(tree.length)
  graph.setGraph({
    rankdir: 'TB',
    nodesep: layout.nodesep,
    ranksep: layout.ranksep,
    marginx: layout.margin,
    marginy: layout.margin,
  })

  const orderedTree = [...tree].sort((left, right) => {
    const depthDiff = left.depth - right.depth
    if (depthDiff !== 0) {
      return depthDiff
    }

    const parentDiff = (left.parentId ?? -1) - (right.parentId ?? -1)
    if (parentDiff !== 0) {
      return parentDiff
    }

    return left.id - right.id
  })

  for (const node of orderedTree) {
    graph.setNode(String(node.id), {
      width: layout.nodeWidth,
      height: layout.nodeHeight,
    })
  }

  for (const node of orderedTree) {
    if (node.parentId !== null) {
      graph.setEdge(String(node.parentId), String(node.id))
    }
  }

  dagre.layout(graph)
  const nodeById = new Map(orderedTree.map((node) => [node.id, node] as const))
  const currentRolloutLeafId =
    frame.selectionPathIds.length > 0 ? frame.selectionPathIds[frame.selectionPathIds.length - 1] : null

  const nodes: SearchFlowNodeType[] = orderedTree.map((node) => {
    const position = graph.node(String(node.id))
    const parentTile = node.parentId === null ? null : nodeById.get(node.parentId)?.tile ?? null
    const simulationPath =
      rolloutPathByNodeId.get(node.id) ??
      (currentRolloutLeafId === node.id && frame.rolloutTiles.length > 0 ? frame.rolloutTiles : [])
    const simulationIndicatorToTile = simulationPath[1] ?? null

    return {
      id: String(node.id),
      type: 'searchNode',
      data: {
        mode: 'mcts',
        snapshot: node,
        active: node.id === activeNodeId,
        transitionPath: parentTile ? [parentTile, node.tile] : [node.tile],
        simulationPath,
        simulationIndicatorToTile,
        goalTile,
        boardSize: layout.boardSize,
        compact: layout.compact,
      },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
      draggable: false,
      selectable: false,
      position: {
        x: position.x - layout.nodeWidth / 2,
        y: position.y - layout.nodeHeight / 2,
      },
      width: layout.nodeWidth,
      height: layout.nodeHeight,
    }
  })

  const edges: Edge[] = orderedTree.flatMap((node) => {
    if (node.parentId === null || node.actionFromParent === null) {
      return []
    }

    const isActive = node.id === activeNodeId
    const stroke = isActive ? '#2563eb' : '#9fb3d8'
    return [
      {
        id: `${graphIdPrefix}-${node.parentId}-${node.id}`,
        source: String(node.parentId),
        target: String(node.id),
        type: layout.edgeType,
        animated: isActive,
        label: layout.showLabels ? describeAction(node.actionFromParent) : undefined,
        zIndex: 10,
        labelStyle: {
          fontSize: layout.compact ? 9 : 10,
          fontWeight: 700,
          fill: stroke,
        },
        labelBgPadding: [6, 3],
        labelBgBorderRadius: 10,
        labelBgStyle: {
          fill: '#ffffff',
          fillOpacity: 0.92,
          stroke: '#d9e1ee',
        },
        markerEnd: {
          id: `${graphIdPrefix}-marker-${node.parentId}-${node.id}`,
          type: MarkerType.ArrowClosed,
          width: layout.markerSize,
          height: layout.markerSize,
          color: stroke,
        },
        style: {
          stroke,
          strokeWidth: isActive ? layout.activeEdgeWidth : layout.inactiveEdgeWidth,
        },
      },
    ]
  })

  return {
    nodes,
    edges,
    minZoom: layout.minZoom,
    fitViewPadding: layout.fitViewPadding,
  }
}

function getLatestSearchFrame(frames: DemoFrame[], index: number) {
  for (let cursor = Math.min(index, frames.length - 1); cursor >= 0; cursor -= 1) {
    const frame = frames[cursor]
    if (isAStarFrame(frame) || isMCTSFrame(frame)) {
      return frame
    }
  }

  return null
}

function buildMCTSRolloutPathByNodeId(frames: MCTSFrame[]) {
  const rolloutPathMap = new Map<number, TileId[]>()

  for (const frame of frames) {
    if (frame.phase !== 'rollout' || frame.activeNodeId === null) {
      continue
    }

    rolloutPathMap.set(frame.activeNodeId, [...frame.rolloutTiles])
  }

  return rolloutPathMap
}

function getCommittedMCTSPath(frames: DemoFrame[], index: number, fallback: TileId) {
  const visibleFrames = frames.slice(0, index + 1)
  const latestExecutionFrame = [...visibleFrames]
    .reverse()
    .find((frame): frame is ExecutionFrame => isExecutionFrame(frame) && frame.mctsHistoryIndex !== null)

  if (latestExecutionFrame) {
    return latestExecutionFrame.path
  }

  const firstMCTSFrame = visibleFrames.find(isMCTSFrame) ?? frames.find(isMCTSFrame)
  if (!firstMCTSFrame) {
    return [fallback]
  }

  const rootTile = firstMCTSFrame.tree[0]?.tile ?? fallback
  return [rootTile]
}

function buildAStarTreePath(nodeId: string, nodeById: Map<string, AStarRenderNodeSnapshot>) {
  const path: TileId[] = []
  const seen = new Set<string>()
  let cursorId: string | null = nodeId

  while (cursorId !== null && !seen.has(cursorId)) {
    const node = nodeById.get(cursorId)
    if (!node) {
      break
    }
    path.unshift(node.tile)
    seen.add(cursorId)
    cursorId = node.parentId
  }

  return path
}

function buildAStarNavigationDiagram(frame: AStarFrame, goalTile: TileId) {
  const graph = new dagre.graphlib.Graph()
  graph.setDefaultEdgeLabel(() => ({}))
  graph.setGraph({
    rankdir: 'TB',
    nodesep: 42,
    ranksep: 84,
    marginx: 28,
    marginy: 28,
  })

  const treeNodeById = new Map(frame.tree.map((node) => [node.id, node] as const))
  const nodeByTile = new Map<TileId, AStarTreeNodeSnapshot>(
    frame.tree.map((node) => [node.tile, node] as const),
  )

  const uniqueTiles = new Set<TileId>(frame.tree.map((n) => n.tile))
  const rejectedChild = frame.rejectedChild
  if (rejectedChild) {
    uniqueTiles.add(rejectedChild.tile)
  }

  const renderNodeByTile = new Map<TileId, AStarRenderNodeSnapshot>()
  for (const tile of uniqueTiles) {
    const accepted = nodeByTile.get(tile)
    const parentNode = accepted ? (accepted.parentId !== null ? treeNodeById.get(accepted.parentId) : null) : null
    const parentTile = parentNode?.tile ?? null
    if (accepted) {
      renderNodeByTile.set(tile, {
        id: tile,
        tile,
        parentId: parentTile,
        actionFromParent: accepted.actionFromParent,
        depth: accepted.depth,
        g: accepted.g,
        h: accepted.h,
        f: accepted.f,
        status: accepted.status,
      })
    } else if (rejectedChild && rejectedChild.tile === tile) {
      const parentNodeForRejected = treeNodeById.get(rejectedChild.parentTreeNodeId)
      const parentTileForRejected = parentNodeForRejected?.tile ?? null
      renderNodeByTile.set(tile, {
        id: tile,
        tile,
        parentId: parentTileForRejected,
        actionFromParent: rejectedChild.actionFromParent,
        depth: (parentNodeForRejected?.depth ?? 0) + 1,
        g: rejectedChild.g,
        h: rejectedChild.h,
        f: rejectedChild.f,
        status: 'rejected',
        rejectionReason: rejectedChild.reason,
      })
    }
  }

  const orderedTiles = [...uniqueTiles].sort((a, b) => {
    const nodeA = renderNodeByTile.get(a)
    const nodeB = renderNodeByTile.get(b)
    if (!nodeA || !nodeB) return 0
    const depthDiff = nodeA.depth - nodeB.depth
    if (depthDiff !== 0) return depthDiff
    const parentDiff = (nodeA.parentId ?? '').localeCompare(nodeB.parentId ?? '')
    if (parentDiff !== 0) return parentDiff
    const fDiff = nodeA.f - nodeB.f
    if (fDiff !== 0) return fDiff
    return a.localeCompare(b)
  })

  for (const tile of orderedTiles) {
    graph.setNode(tile, {
      width: SEARCH_NODE_WIDTH,
      height: SEARCH_NODE_HEIGHT,
    })
  }

  type EdgeKey = { parentTile: TileId; childTile: TileId; action: Action; status: 'accepted' | 'rejected' }
  const edgesToAdd: EdgeKey[] = []
  for (const node of frame.tree) {
    if (node.parentId === null || node.actionFromParent === null) continue
    const parentNode = treeNodeById.get(node.parentId)
    if (!parentNode) continue
    edgesToAdd.push({
      parentTile: parentNode.tile,
      childTile: node.tile,
      action: node.actionFromParent,
      status: 'accepted',
    })
  }
  if (rejectedChild) {
    const parentNode = treeNodeById.get(rejectedChild.parentTreeNodeId)
    if (parentNode) {
      edgesToAdd.push({
        parentTile: parentNode.tile,
        childTile: rejectedChild.tile,
        action: rejectedChild.actionFromParent,
        status: 'rejected',
      })
    }
  }

  for (const { parentTile, childTile } of edgesToAdd) {
    graph.setEdge(parentTile, childTile)
  }

  dagre.layout(graph)
  const nodeById = new Map(orderedTiles.map((tile) => [tile, renderNodeByTile.get(tile)!] as const))

  const activePathEdgeSet = new Set<string>()
  let pathCursorId: number | null = frame.activeTreeNodeId
  const activeTiles: TileId[] = []
  while (pathCursorId !== null) {
    const node = treeNodeById.get(pathCursorId)
    if (!node) break
    activeTiles.unshift(node.tile)
    pathCursorId = node.parentId
  }
  for (let i = 0; i < activeTiles.length - 1; i++) {
    const parentTile = activeTiles[i]
    const childTile = activeTiles[i + 1]
    const edge = edgesToAdd.find((e) => e.parentTile === parentTile && e.childTile === childTile && e.status === 'accepted')
    if (edge) {
      activePathEdgeSet.add(`${parentTile}-${childTile}-${edge.action}-accepted`)
    }
  }

  const nodes: SearchFlowNodeType[] = orderedTiles.map((tile) => {
    const snapshot = renderNodeByTile.get(tile)!
    const position = graph.node(tile)
    return {
      id: tile,
      type: 'searchNode',
      data: {
        mode: 'astar',
        snapshot,
        previewPath: buildAStarTreePath(tile, nodeById),
        goalTile,
      },
      draggable: false,
      selectable: false,
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
      position: {
        x: position.x - SEARCH_NODE_WIDTH / 2,
        y: position.y - SEARCH_NODE_HEIGHT / 2,
      },
      width: SEARCH_NODE_WIDTH,
      height: SEARCH_NODE_HEIGHT,
    }
  })

  const edges: Edge[] = edgesToAdd.map(({ parentTile, childTile, action, status }) => {
    const edgeId = `${parentTile}-${childTile}-${action}-${status}`
    const isActivePath = status === 'accepted' && activePathEdgeSet.has(edgeId)
    const snapshot = renderNodeByTile.get(childTile)!
    const stroke =
      status === 'rejected'
        ? '#dc2626'
        : snapshot.status === 'active' || isActivePath
          ? '#2563eb'
          : snapshot.status === 'queued'
            ? '#d97706'
            : '#94a3b8'

    return {
      id: edgeId,
      source: parentTile,
      target: childTile,
      type: 'straight',
      animated: snapshot.status === 'active' || status === 'rejected',
      label: describeAction(action),
      labelStyle: {
        fontSize: 10,
        fontWeight: 700,
        fill: stroke,
      },
      labelBgPadding: [6, 3],
      labelBgBorderRadius: 10,
      labelBgStyle: {
        fill: '#ffffff',
        fillOpacity: 0.92,
        stroke: '#d9e1ee',
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 18,
        height: 18,
        color: stroke,
      },
      style: {
        stroke,
        strokeDasharray: status === 'rejected' ? '6 4' : undefined,
        strokeWidth: status === 'rejected' ? 2.2 : isActivePath ? 2.6 : 2,
      },
    }
  })

  return { nodes, edges }
}

function getFrameTile(frame: DemoFrame | null, fallback: TileId) {
  if (!frame) {
    return fallback
  }

  if (isExecutionFrame(frame)) {
    return frame.tile
  }

  if (isAStarFrame(frame)) {
    if (frame.pathPreview.length > 0) {
      return frame.pathPreview[frame.pathPreview.length - 1]
    }
    return frame.current ?? fallback
  }

  if (frame.rolloutTiles.length > 0) {
    return frame.rolloutTiles[frame.rolloutTiles.length - 1]
  }

  if (frame.activeNodeId !== null) {
    const activeNode = frame.tree.find((node) => node.id === frame.activeNodeId)
    if (activeNode) {
      return activeNode.tile
    }
  }

  return fallback
}

function formatTraceList<T>(
  values: T[],
  formatter: (value: T) => string = (value) => String(value),
  limit = 6,
) {
  if (values.length === 0) {
    return '(empty)'
  }

  const visibleValues = values.slice(0, limit).map(formatter)
  const remainder = values.length - visibleValues.length

  return remainder > 0
    ? `${visibleValues.join(', ')} ... (+${remainder})`
    : visibleValues.join(', ')
}

function formatAStarScorePreview(frame: AStarFrame) {
  const visibleRows = frame.scoreRows
    .filter((row) => row.f !== null)
    .sort((left, right) => (left.f ?? Number.POSITIVE_INFINITY) - (right.f ?? Number.POSITIVE_INFINITY))
    .slice(0, 4)
    .map((row) => `${row.tile}(g=${formatAStarScore(row.g)}, h=${formatGraphNumber(row.h)}, f=${formatAStarScore(row.f)})`)

  return visibleRows.length > 0 ? visibleRows.join(' | ') : '(none yet)'
}

function getAStarTraceLineId(frame: AStarFrame) {
  switch (frame.phase) {
    case 'init':
      return 'init'
    case 'loop':
      return 'loop'
    case 'expand':
      return 'expand'
    case 'consider-neighbor':
      return 'consider'
    case 'update-neighbor':
      return 'update'
    case 'update-queue':
      return 'update-queue'
    case 'skip-neighbor':
      return 'skip'
    case 'goal-found':
      return 'goal'
    case 'finished':
      return frame.reachedGoal ? 'goal' : 'finish'
    default:
      return 'loop'
  }
}

/** When the active line is a conditional body (then or else), return the parent line id (if/else) so both can be highlighted. */
function getAStarTraceParentLineId(frame: AStarFrame): string | undefined {
  switch (frame.phase) {
    case 'goal-found':
      return 'goal-check'
    case 'consider-neighbor':
      return 'goal-else'
    case 'skip-neighbor':
      return 'skip-check'
    case 'update-neighbor':
    case 'update-queue':
      return 'skip-else'
    default:
      return undefined
  }
}

function getMCTSTraceLineId(frame: MCTSFrame) {
  switch (frame.phase) {
    case 'init':
      return 'init'
    case 'loop':
      return 'loop'
    case 'selection':
      return 'selection'
    case 'expansion':
      return 'expansion'
    case 'expansion-else':
      return 'expansion-else'
    case 'rollout':
      return 'rollout'
    case 'backprop':
      return 'backprop'
    case 'iteration-end':
      return 'iteration-end'
    case 'finished':
      return 'finished'
    default:
      return 'loop'
  }
}

/** When the active line is a conditional body (expansion, expansion-else, or rollout), return the parent line id so both can be highlighted. */
function getMCTSTraceParentLineId(frame: MCTSFrame): string | undefined {
  switch (frame.phase) {
    case 'expansion':
      return 'expansion-check'
    case 'expansion-else':
      return 'expansion-check'
    case 'rollout':
      return 'expansion-else'
    default:
      return undefined
  }
}

function buildAStarTraceModel(frame: AStarFrame): AlgorithmTraceModel {
  return {
    algorithmLabel: 'A* Pseudocode',
    frameMessage: frame.message,
    activeLineId: getAStarTraceLineId(frame),
    activeParentLineId: getAStarTraceParentLineId(frame),
    lines: ASTAR_TRACE_LINES,
    values: [
      { label: 'Frame', value: String(frame.step) },
      { label: 'Step', value: String(frame.step) },
      { label: 'Phase', value: frame.phase },
      { label: 'Current', value: frame.current ?? '(none)' },
      { label: 'Action', value: frame.action ? describeAction(frame.action) : '(none)' },
      { label: 'Neighbor', value: frame.neighbor ?? '(none)' },
      { label: 'Open set', value: formatTraceList(frame.openSet) },
      { label: 'Closed set', value: formatTraceList(frame.closedSet) },
      { label: 'Path preview', value: formatTraceList(frame.pathPreview) },
      { label: 'Best scores', value: formatAStarScorePreview(frame) },
    ],
  }
}

function buildMCTSTraceModel(frame: MCTSFrame): AlgorithmTraceModel {
  return {
    algorithmLabel: 'MCTS Pseudocode',
    frameMessage: frame.message,
    activeLineId: getMCTSTraceLineId(frame),
    activeParentLineId: getMCTSTraceParentLineId(frame),
    lines: MCTS_TRACE_LINES,
    values: [
      { label: 'Frame', value: String(frame.step) },
      { label: 'Step', value: String(frame.step) },
      { label: 'Decision step', value: String(frame.decisionStep + 1) },
      { label: 'Iteration', value: String(frame.iteration) },
      { label: 'Phase', value: frame.phase },
      { label: 'Active node', value: frame.activeNodeId === null ? '(none)' : `#${frame.activeNodeId}` },
      {
        label: 'Selection path',
        value: formatTraceList(frame.selectionPathIds, (nodeId) => `#${nodeId}`),
      },
      { label: 'Rollout tiles', value: formatTraceList(frame.rolloutTiles) },
      { label: 'Best root action', value: frame.bestRootAction ? describeAction(frame.bestRootAction) : '(none yet)' },
      { label: 'Best root visits', value: String(frame.bestRootVisits) },
      { label: 'Tree nodes', value: String(frame.tree.length) },
    ],
  }
}

function AlgorithmTracePanel({
  trace,
  controls = [],
  variant = 'astar',
}: {
  trace: AlgorithmTraceModel
  controls?: AlgorithmTraceControl[]
  variant?: 'astar' | 'mcts'
}) {
  const controlsByLabel = new Map(controls.map((control) => [control.label, control] as const))

  return (
    <aside
      className={`algorithm-trace-panel${variant === 'mcts' ? ' algorithm-trace-panel-mcts' : ''}`}
      aria-label={`${trace.algorithmLabel} trace`}
    >
      <div className="algorithm-trace-header">
        <p className="algorithm-trace-eyebrow">Algorithm Trace</p>
        <h3>{trace.algorithmLabel}</h3>
        <p className="algorithm-trace-message">{trace.frameMessage}</p>
      </div>

      <ol
        className={`algorithm-trace-lines${trace.activeParentLineId ? ' algorithm-trace-lines-has-parent' : ''}`}
      >
        {trace.lines.map((line, index) => {
          const isActive = line.id === trace.activeLineId
          const isActiveParent = line.id === trace.activeParentLineId

          const isRolloutHighlight =
            variant === 'mcts' && line.id === 'rollout' && (isActive || isActiveParent)

          return (
            <li
              key={line.id}
              className={`algorithm-trace-line${isActive ? ' algorithm-trace-line-active' : ''}${isActiveParent ? ' algorithm-trace-line-active-parent' : ''}${isRolloutHighlight ? ' algorithm-trace-line-rollout' : ''}`}
              style={line.indent ? { marginLeft: `${line.indent * 16}px` } : undefined}
            >
              <span className="algorithm-trace-line-number">{index + 1}</span>
              <span className="algorithm-trace-line-text">{line.text}</span>
            </li>
          )
        })}
      </ol>

      <dl className="algorithm-trace-values">
        {trace.values.map((entry) => (
          <Fragment key={entry.label}>
            <dt>{entry.label}</dt>
            <dd>
              {controlsByLabel.has(entry.label) ? (
                <div
                  className="algorithm-trace-stepper"
                  aria-label={`${controlsByLabel.get(entry.label)!.label} navigation`}
                >
                  <button
                    className="algorithm-trace-stepper-button"
                    type="button"
                    onClick={controlsByLabel.get(entry.label)!.onPrevious}
                    disabled={!controlsByLabel.get(entry.label)!.canPrevious}
                    aria-label={`Previous ${controlsByLabel.get(entry.label)!.label.toLowerCase()}`}
                  >
                    {'<'}
                  </button>
                  <span className="algorithm-trace-stepper-value">
                    {controlsByLabel.get(entry.label)!.value}
                  </span>
                  <button
                    className="algorithm-trace-stepper-button"
                    type="button"
                    onClick={controlsByLabel.get(entry.label)!.onNext}
                    disabled={!controlsByLabel.get(entry.label)!.canNext}
                    aria-label={`Next ${controlsByLabel.get(entry.label)!.label.toLowerCase()}`}
                  >
                    {'>'}
                  </button>
                </div>
              ) : (
                entry.value
              )}
            </dd>
          </Fragment>
        ))}
      </dl>
    </aside>
  )
}

function App() {
  const [startTile, setStartTile] = useState<TileId>(() => getRandomStartTile())
  const [currentTile, setCurrentTile] = useState<TileId>(() => startTile)
  const [algorithm, setAlgorithm] = useState<AlgorithmKind>('astar')
  const [goalTile, setGoalTile] = useState<TileId>('2F')
  const [astarHeuristicExpression, setAStarHeuristicExpression] = useState(
    DEFAULT_ASTAR_HEURISTIC_EXPRESSION,
  )
  const [demoFrames, setDemoFrames] = useState<DemoFrame[]>([])
  const [demoIndex, setDemoIndex] = useState(0)
  const [isAutoPlay, setIsAutoPlay] = useState(false)
  const [playbackMs, setPlaybackMs] = useState(550)
  const [mctsIterations, setMctsIterations] = useState(40)
  const [mctsExplorationC, setMctsExplorationC] = useState(Math.SQRT2)
  const [mctsGamma, setMctsGamma] = useState(0.9)
  const [mctsHorizon, setMctsHorizon] = useState(100)
  const [mctsGoalReward, setMctsGoalReward] = useState(10)
  const [mctsTrapReward, setMctsTrapReward] = useState(-1)
  const [mctsStuckReward, setMctsStuckReward] = useState(0)
  const [mctsNormalReward, setMctsNormalReward] = useState(0)
  const [mctsGraphHistory, setMctsGraphHistory] = useState<MCTSGraphHistoryEntry[]>([])
  const [selectedHistoricalMCTSGraphIndex, setSelectedHistoricalMCTSGraphIndex] = useState<number | null>(null)
  const [tokenAnimation, setTokenAnimation] = useState<TokenAnimation | null>(null)
  const [boardCellSize, setBoardCellSize] = useState<number | null>(null)
  const tokenAnimationCounterRef = useRef(0)
  const tokenAnimationTimerRef = useRef<number | null>(null)
  const boardGridShellRef = useRef<HTMLDivElement | null>(null)
  const boardGridRef = useRef<HTMLDivElement | null>(null)

  const allTiles = useMemo(() => getAllTiles(), [])
  const astarHeuristicAnalysis = useMemo(
    () => analyzeAStarHeuristic(astarHeuristicExpression, goalTile),
    [astarHeuristicExpression, goalTile],
  )
  const activeFrame = demoFrames[demoIndex] ?? null
  const latestSearchFrame = useMemo(
    () => getLatestSearchFrame(demoFrames, demoIndex),
    [demoFrames, demoIndex],
  )
  const displayedAStarFrame = isAStarFrame(latestSearchFrame) ? latestSearchFrame : null
  const currentMCTSFrame = isMCTSFrame(latestSearchFrame) ? latestSearchFrame : null
  const activeMCTSGraphIndex =
    isMCTSFrame(activeFrame) ? activeFrame.decisionStep : isExecutionFrame(activeFrame) ? activeFrame.mctsHistoryIndex : null
  const displayedHistoricalMCTSHistoryEntry =
    selectedHistoricalMCTSGraphIndex === null
      ? null
      : mctsGraphHistory[selectedHistoricalMCTSGraphIndex] ?? null
  const isViewingMCTSPlayback =
    currentMCTSFrame !== null || (isExecutionFrame(activeFrame) && activeFrame.mctsHistoryIndex !== null)
  const frameVisibleTile = getFrameTile(activeFrame, currentTile)

  const astarOpenSet = isAStarFrame(activeFrame) ? new Set(activeFrame.openSet) : null
  const astarClosedSet = isAStarFrame(activeFrame) ? new Set(activeFrame.closedSet) : null
  const astarPathTiles = isAStarFrame(activeFrame) ? new Set(activeFrame.pathPreview) : null
  const astarPathSegments = isAStarFrame(activeFrame) ? buildPathSegments(activeFrame.pathPreview) : []
  const mctsCommittedPath = useMemo(
    () => getCommittedMCTSPath(demoFrames, demoIndex, currentTile),
    [currentTile, demoFrames, demoIndex],
  )
  const mctsCommittedTileSet = useMemo(() => new Set(mctsCommittedPath), [mctsCommittedPath])
  const mctsCommittedSegments = useMemo(() => buildPathSegments(mctsCommittedPath), [mctsCommittedPath])
  const visibleTile = isViewingMCTSPlayback
    ? mctsCommittedPath[mctsCommittedPath.length - 1] ?? currentTile
    : frameVisibleTile
  const currentType = getTileType(visibleTile)
  const restingTokenPosition = getTokenPosition(visibleTile)
  const astarHasDistinctNeighbor =
    isAStarFrame(activeFrame) &&
    activeFrame.current !== null &&
    activeFrame.neighbor !== null &&
    activeFrame.neighbor !== activeFrame.current
  const astarScoreMap = useMemo(
    () =>
      displayedAStarFrame
        ? new Map(displayedAStarFrame.scoreRows.map((row) => [row.tile, row] as const))
        : new Map<TileId, AStarFrame['scoreRows'][number]>(),
    [displayedAStarFrame],
  )
  const displayedMCTSFrame =
    isMCTSFrame(activeFrame) && activeMCTSGraphIndex === selectedHistoricalMCTSGraphIndex
      ? activeFrame
      : displayedHistoricalMCTSHistoryEntry?.frame ?? currentMCTSFrame
  const astarTrace = useMemo(
    () => (displayedAStarFrame ? buildAStarTraceModel(displayedAStarFrame) : null),
    [displayedAStarFrame],
  )
  const mctsTrace = useMemo(
    () => (displayedMCTSFrame ? buildMCTSTraceModel(displayedMCTSFrame) : null),
    [displayedMCTSFrame],
  )
  const displayedMCTSRolloutPathByNodeId = useMemo(() => {
    if (isMCTSFrame(activeFrame) && activeMCTSGraphIndex === selectedHistoricalMCTSGraphIndex) {
      const currentStepFrames = demoFrames
        .slice(0, demoIndex + 1)
        .filter(
          (frame): frame is MCTSFrame =>
            isMCTSFrame(frame) && frame.decisionStep === activeFrame.decisionStep,
        )

      return buildMCTSRolloutPathByNodeId(currentStepFrames)
    }

    return displayedHistoricalMCTSHistoryEntry?.rolloutPathByNodeId ?? new Map<number, TileId[]>()
  }, [activeFrame, activeMCTSGraphIndex, demoFrames, demoIndex, displayedHistoricalMCTSHistoryEntry, selectedHistoricalMCTSGraphIndex])
  const currentMCTSTreeDiagram = useMemo(
    () =>
      displayedMCTSFrame
        ? buildMCTSTreeDiagram(
            displayedMCTSFrame,
            displayedMCTSRolloutPathByNodeId,
            goalTile,
            `mcts-current-${displayedMCTSFrame.decisionStep}`,
          )
        : {
            nodes: [] as SearchFlowNodeType[],
            edges: [] as Edge[],
            minZoom: 0.18,
            fitViewPadding: 0.16,
          },
    [displayedMCTSFrame, displayedMCTSRolloutPathByNodeId, goalTile],
  )
  const astarGraphDiagram = useMemo(
    () =>
      displayedAStarFrame
        ? buildAStarNavigationDiagram(displayedAStarFrame, goalTile)
        : {
            nodes: [] as SearchFlowNodeType[],
            edges: [] as Edge[],
          },
    [displayedAStarFrame, goalTile],
  )
  const searchNodeTypes = useMemo(() => ({ searchNode: SearchFlowNodeCard }), [])
  const canViewPreviousHistoricalGraph =
    selectedHistoricalMCTSGraphIndex !== null && selectedHistoricalMCTSGraphIndex > 0
  const canViewNextHistoricalGraph =
    selectedHistoricalMCTSGraphIndex !== null &&
    selectedHistoricalMCTSGraphIndex < mctsGraphHistory.length - 1
  const goToPreviousFrame = useCallback(() => {
    setIsAutoPlay(false)
    setDemoIndex((index) => Math.max(0, index - 1))
  }, [])
  const goToNextFrame = useCallback(() => {
    setIsAutoPlay(false)
    setDemoIndex((index) => Math.min(index + 1, demoFrames.length - 1))
  }, [demoFrames.length])
  const clearPlaybackState = useCallback(() => {
    setDemoFrames([])
    setDemoIndex(0)
    setIsAutoPlay(false)
    setMctsGraphHistory([])
    setSelectedHistoricalMCTSGraphIndex(null)
  }, [])

  const clearTokenAnimation = useCallback(() => {
    if (tokenAnimationTimerRef.current !== null) {
      window.clearTimeout(tokenAnimationTimerRef.current)
      tokenAnimationTimerRef.current = null
    }
    setTokenAnimation(null)
  }, [])

  const animateTokenMove = useCallback(
    (from: TileId, to: TileId, action: Action) => {
      clearTokenAnimation()
      tokenAnimationCounterRef.current += 1

      const start = getTokenPosition(from)
      const end = getTokenPosition(to)
      const bump = getTokenBumpOffset(action)
      const stationary = from === to
      const duration = stationary ? 280 : 360

      setTokenAnimation({
        key: tokenAnimationCounterRef.current,
        className: stationary ? 'avatar-bump' : 'avatar-slide',
        style: {
          left: start.x,
          top: start.y,
          '--start-x': start.x,
          '--start-y': start.y,
          '--end-x': end.x,
          '--end-y': end.y,
          '--bump-x': bump.x,
          '--bump-y': bump.y,
        } as CSSProperties,
      })

      tokenAnimationTimerRef.current = window.setTimeout(() => {
        tokenAnimationTimerRef.current = null
        setTokenAnimation(null)
      }, duration + 40)
    },
    [clearTokenAnimation],
  )

  useEffect(
    () => () => {
      if (tokenAnimationTimerRef.current !== null) {
        window.clearTimeout(tokenAnimationTimerRef.current)
      }
    },
    [],
  )

  useEffect(() => {
    const boardGridShell = boardGridShellRef.current
    const boardGrid = boardGridRef.current

    if (!boardGridShell || !boardGrid) {
      return
    }

    const updateBoardCellSize = () => {
      const labelColumnSize =
        Number.parseFloat(window.getComputedStyle(boardGrid).getPropertyValue('--label-col-size')) || 32
      const availableCellWidth = boardGridShell.clientWidth - labelColumnSize

      if (availableCellWidth <= 0) {
        return
      }

      const nextCellSize = Math.min(140, Math.max(38, availableCellWidth / COLUMNS.length))
      setBoardCellSize((current) =>
        current !== null && Math.abs(current - nextCellSize) < 0.5 ? current : nextCellSize,
      )
    }

    updateBoardCellSize()

    const resizeObserver = new ResizeObserver(() => {
      updateBoardCellSize()
    })
    resizeObserver.observe(boardGridShell)

    return () => {
      resizeObserver.disconnect()
    }
  }, [])

  useEffect(() => {
    if (mctsGraphHistory.length === 0) {
      setSelectedHistoricalMCTSGraphIndex(null)
      return
    }

    if (activeMCTSGraphIndex !== null) {
      setSelectedHistoricalMCTSGraphIndex(activeMCTSGraphIndex)
      return
    }

    setSelectedHistoricalMCTSGraphIndex((current) => {
      if (current === null) {
        return 0
      }

      return Math.min(mctsGraphHistory.length - 1, Math.max(0, current))
    })
  }, [activeMCTSGraphIndex, mctsGraphHistory])

  const handleReset = useCallback(() => {
    clearTokenAnimation()
    setCurrentTile(startTile)
    clearPlaybackState()
  }, [clearPlaybackState, clearTokenAnimation, startTile])

  const handleStartTileChange = useCallback(
    (tile: TileId) => {
      clearTokenAnimation()
      setStartTile(tile)
      setCurrentTile(tile)
      clearPlaybackState()
    },
    [clearPlaybackState, clearTokenAnimation],
  )

  const performStep = useCallback(
    (action: Action) => {
      const sampledTransition = sampleTransition(currentTile, action, Math.random, goalTile)

      animateTokenMove(currentTile, sampledTransition.destination, action)
      setStartTile(sampledTransition.destination)
      setCurrentTile(sampledTransition.destination)
    },
    [animateTokenMove, currentTile, goalTile],
  )

  useEffect(() => {
    const keyToAction: Partial<Record<string, Action>> = {
      ArrowUp: 'up',
      ArrowDown: 'down',
      ArrowLeft: 'left',
      ArrowRight: 'right',
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      const tagName = target?.tagName

      if (tagName === 'INPUT' || tagName === 'TEXTAREA' || target?.isContentEditable) {
        return
      }

      const action = keyToAction[event.key]

      if (!action) {
        return
      }

      event.preventDefault()
      performStep(action)
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [performStep])

  const runAlgorithm = useCallback(() => {
    clearTokenAnimation()
    setIsAutoPlay(true)
    if (algorithm === 'astar') {
      if (astarHeuristicAnalysis.error) {
        setIsAutoPlay(false)
        return
      }

      const result = runAStarDemo({
        start: currentTile,
        goal: goalTile,
        heuristicExpression: astarHeuristicExpression,
      })
      const executionFrames = buildExecutionFrames(result.path, result.actions).map((frame, index) => ({
        ...frame,
        step: result.frames.length + index,
      }))
      setDemoFrames([...result.frames, ...executionFrames])
      setDemoIndex(0)
      setMctsGraphHistory([])
      setSelectedHistoricalMCTSGraphIndex(null)
      return
    }

    const nextDemoFrames: DemoFrame[] = []
    const nextGraphHistory: MCTSGraphHistoryEntry[] = []
    const executionPath: TileId[] = [currentTile]
    let planningTile = currentTile
    let timelineStep = 0
    let decisionStep = 0
    let incomingAction: Action | null = null

    while (true) {
      const result = runMCTSDemo({
        start: planningTile,
        goal: goalTile,
        decisionStep,
        iterations: mctsIterations,
        explorationConstant: mctsExplorationC,
        gamma: mctsGamma,
        rolloutHorizon: mctsHorizon,
        goalReward: mctsGoalReward,
        trapReward: mctsTrapReward,
        stuckReward: mctsStuckReward,
        normalReward: mctsNormalReward,
        rng: Math.random,
      })

      const searchFrames = result.frames.map((frame) => ({
        ...frame,
        step: timelineStep++,
      }))
      nextDemoFrames.push(...searchFrames)

      const finalFrame = searchFrames[searchFrames.length - 1]!
      const historyIndex = nextGraphHistory.length
      nextGraphHistory.push({
        decisionStep,
        rootTile: planningTile,
        resultingTile: result.nextTile,
        incomingAction,
        selectedAction: result.selectedAction,
        bestRootVisits: result.selectedActionVisits,
        frame: finalFrame,
        rolloutPathByNodeId: buildMCTSRolloutPathByNodeId(result.frames),
      })

      if (!result.selectedAction) {
        break
      }

      executionPath.push(result.nextTile)
      nextDemoFrames.push(
        buildExecutionFrame({
          timelineStep: timelineStep++,
          tile: result.nextTile,
          from: planningTile,
          action: result.selectedAction,
          path: [...executionPath],
          message: `Execution step ${executionPath.length - 1}: ${planningTile} --${result.selectedAction}--> ${result.nextTile}. ${result.transitionExplanation ?? ''}`.trim(),
          mctsHistoryIndex: historyIndex,
        }),
      )

      if (result.terminated) {
        break
      }

      incomingAction = result.selectedAction
      planningTile = result.nextTile
      decisionStep += 1
    }

    setDemoFrames(nextDemoFrames)
    setDemoIndex(0)
    setMctsGraphHistory(nextGraphHistory)
    setSelectedHistoricalMCTSGraphIndex(nextGraphHistory.length > 0 ? 0 : null)
  }, [
    algorithm,
    clearTokenAnimation,
    currentTile,
    goalTile,
    astarHeuristicAnalysis.error,
    astarHeuristicExpression,
    mctsExplorationC,
    mctsGamma,
    mctsGoalReward,
    mctsHorizon,
    mctsIterations,
    mctsNormalReward,
    mctsStuckReward,
    mctsTrapReward,
  ])
  const goToPreviousHistoricalGraph = useCallback(() => {
    setIsAutoPlay(false)
    setSelectedHistoricalMCTSGraphIndex((current) => {
      if (current === null) {
        return null
      }
      return Math.max(0, current - 1)
    })
  }, [])

  const goToNextHistoricalGraph = useCallback(() => {
    setIsAutoPlay(false)
    setSelectedHistoricalMCTSGraphIndex((current) => {
      if (current === null) {
        return mctsGraphHistory.length > 0 ? 0 : null
      }
      return Math.min(mctsGraphHistory.length - 1, current + 1)
    })
  }, [mctsGraphHistory.length])

  useEffect(() => {
    if (!isAutoPlay || demoFrames.length === 0) {
      return
    }

    if (demoIndex >= demoFrames.length - 1) {
      return
    }

    const timerId = window.setTimeout(() => {
      setDemoIndex((index) => {
        const nextIndex = Math.min(index + 1, demoFrames.length - 1)
        if (nextIndex >= demoFrames.length - 1) {
          setIsAutoPlay(false)
        }
        return nextIndex
      })
    }, playbackMs)

    return () => window.clearTimeout(timerId)
  }, [demoFrames.length, demoIndex, isAutoPlay, playbackMs])

  const renderedTokenAnimation = activeFrame ? null : tokenAnimation
  const boardAvatarClassName = renderedTokenAnimation?.className ?? 'avatar-rest'
  const boardAvatarStyle = renderedTokenAnimation?.style ?? {
    left: restingTokenPosition.x,
    top: restingTokenPosition.y,
  }

  const canStepBack = demoIndex > 0
  const canStepForward = demoFrames.length > 0 && demoIndex < demoFrames.length - 1

  return (
    <main className="app-shell">
      <div className="layout">
        <section className="hero card">
          <p className="eyebrow">Text RPG visualizer</p>
          <h1>Dungeon Crawler</h1>
          <ul className="hero-rules">
            <li>3×6 grid (rows 1–3, columns A–F). Pick the starting location from the controls.</li>
            <li>Arrow keys move the starting location one step at a time.</li>
            <li>
              <strong>White</strong>: move as chosen. <strong>Blue</strong>: 50% stay, else move. <strong>Red/Green</strong>: absorbing (never leave).
            </li>
            <li>
              <strong>Goal</strong>: reaching the goal tile is absorbing (you win and stay).
            </li>
            <li>Reset returns you to the selected starting location. Wall hits keep you in place.</li>
          </ul>
        </section>

        <div className="card board-card">
          <div className="section-heading">
            <h2>Room Map</h2>
            <p>
              Displayed tile: <strong>{visibleTile}</strong> ({currentType})
            </p>
          </div>

          <div className="board-grid-shell" ref={boardGridShellRef}>
            <div
              className="board-grid"
              ref={boardGridRef}
              style={
                {
                  '--board-columns': COLUMNS.length,
                  '--board-rows': BOARD_ROWS.length,
                  ...(boardCellSize === null ? {} : { '--cell-size': `${boardCellSize}px` }),
                } as CSSProperties
              }
              aria-label="Dungeon board"
            >
            <div className="corner-label" />
            {COLUMNS.map((column: (typeof COLUMNS)[number]) => (
              <div className="axis-label column-label" key={column}>
                {column}
              </div>
            ))}

            {BOARD_ROWS.map((row: (typeof BOARD_ROWS)[number]) => (
              <Fragment key={row}>
                <div className="axis-label row-label" key={`row-${row}`}>
                  {row}
                </div>
                {COLUMNS.map((column: (typeof COLUMNS)[number]) => {
                  const tileId = `${row}${column}` as TileId
                  const isCurrent = tileId === visibleTile
                  const isGoal = tileId === goalTile
                  const isOpen = astarOpenSet?.has(tileId) ?? false
                  const isClosed = astarClosedSet?.has(tileId) ?? false
                  const isAStarPath = astarPathTiles?.has(tileId) ?? false
                  const isMCTSPath = isViewingMCTSPlayback && mctsCommittedTileSet.has(tileId)
                  const scoreRow = astarScoreMap.get(tileId)

                  return (
                    <div
                      className={`board-cell ${tileClassName(tileId)} ${isCurrent ? 'current-cell' : ''} ${isGoal ? 'goal-cell' : ''} ${isOpen ? 'open-cell' : ''} ${isClosed ? 'closed-cell' : ''} ${isAStarPath ? 'path-preview-cell' : ''} ${isMCTSPath ? 'committed-path-cell' : ''}`}
                      key={tileId}
                    >
                      <span className="cell-id">{tileId}</span>
                      {scoreRow && (scoreRow.g !== null || scoreRow.f !== null) && (
                        <span className="cell-score">
                          g:{formatAStarScore(scoreRow.g)} h:{formatGraphNumber(scoreRow.h)} f:{formatAStarScore(scoreRow.f)}
                        </span>
                      )}
                    </div>
                  )
                })}
              </Fragment>
            ))}

              <div className="board-overlay" aria-hidden="true">
                <svg className="board-visual-layer" viewBox={`0 0 ${COLUMNS.length} ${BOARD_ROWS.length}`}>
                {isAStarFrame(activeFrame) &&
                  Array.from(astarClosedSet ?? []).map((tile) => {
                    const point = getBoardPoint(tile)
                    return (
                      <g key={`closed-${tile}`}>
                        <circle cx={point.x} cy={point.y} r="0.08" className="astar-closed-dot" />
                        <circle cx={point.x} cy={point.y} r="0.24" className="astar-closed-ring" />
                      </g>
                    )
                  })}
                {isAStarFrame(activeFrame) &&
                  Array.from(astarOpenSet ?? []).map((tile) => {
                    const point = getBoardPoint(tile)
                    return (
                      <g key={`open-${tile}`}>
                        <circle cx={point.x} cy={point.y} r="0.17" className="astar-open-ring" />
                        <circle cx={point.x} cy={point.y} r="0.07" className="astar-open-dot" />
                      </g>
                    )
                  })}
                {isAStarFrame(activeFrame) &&
                  astarPathSegments.map((segment, index) => {
                    const progress = (index + 1) / Math.max(1, astarPathSegments.length)
                    return (
                      <line
                        key={`astar-path-${segment.start}-${segment.end}-${index}`}
                        x1={segment.startPoint.x}
                        y1={segment.startPoint.y}
                        x2={segment.endPoint.x}
                        y2={segment.endPoint.y}
                        className="astar-path-segment"
                        style={{
                          opacity: String(0.28 + progress * 0.72),
                          strokeWidth: String(4.1 + progress * 1),
                        }}
                      />
                    )
                  })}
                {isAStarFrame(activeFrame) &&
                  activeFrame.current &&
                  (() => {
                    const point = getBoardPoint(activeFrame.current)
                    return (
                      <>
                        <circle cx={point.x} cy={point.y} r="0.32" className="astar-current-ring" />
                        <circle cx={point.x} cy={point.y} r="0.14" className="astar-current-dot" />
                      </>
                    )
                  })()}
                {astarHasDistinctNeighbor &&
                  (() => {
                    const start = getBoardPoint(activeFrame.current!)
                    const end = getBoardPoint(activeFrame.neighbor!)
                    return (
                      <>
                        <line
                          x1={start.x}
                          y1={start.y}
                          x2={end.x}
                          y2={end.y}
                          className="astar-neighbor-edge"
                        />
                        <circle cx={end.x} cy={end.y} r="0.23" className="astar-neighbor-ring" />
                      </>
                    )
                  })()}
                {isViewingMCTSPlayback &&
                  mctsCommittedSegments.map((segment, index) => {
                    const proximityToFocus =
                      (index + 1) / Math.max(1, mctsCommittedSegments.length)
                    return (
                      <line
                        key={`mcts-committed-${segment.start}-${segment.end}-${index}`}
                        x1={segment.startPoint.x}
                        y1={segment.startPoint.y}
                        x2={segment.endPoint.x}
                        y2={segment.endPoint.y}
                        className="mcts-committed-segment"
                        style={{
                          opacity: String(0.42 + proximityToFocus * 0.5),
                          strokeWidth: String(4.4 + proximityToFocus * 1.2),
                        }}
                      />
                    )
                  })}
                </svg>
                <span
                  key={renderedTokenAnimation?.key ?? `rest-${visibleTile}`}
                  className={`board-avatar ${boardAvatarClassName}`}
                  style={boardAvatarStyle}
                />
              </div>
            </div>
          </div>

          <div className="legend">
            <span>
              <strong>N</strong>: normal
            </span>
            <span>
              <strong>S</strong>: stuck
            </span>
            <span>
              <strong>T</strong>: trap
            </span>
            <span>
              <strong>G</strong>: goal
            </span>
            <span>
              <strong>Arrows</strong>: move starting location
            </span>
            <span>
              <strong>Start</strong>: {startTile}
            </span>
            <span>
              <strong>Goal</strong>: {goalTile}
            </span>
          </div>

        </div>

        <section className="card settings-card">
          <div className="section-heading">
            <h2>Controls</h2>
            <p>Compute timeline first, then execution timeline.</p>
          </div>

          <div className="control-grid">
            <label>
              Algorithm
              <select
                value={algorithm}
                onChange={(event) => setAlgorithm(event.target.value as AlgorithmKind)}
              >
                <option value="astar">A*</option>
                <option value="mcts">MCTS</option>
              </select>
            </label>

            <label>
              Starting location
              <select
                value={startTile}
                onChange={(event) => handleStartTileChange(event.target.value as TileId)}
              >
                {allTiles.map((tile: TileId) => (
                  <option key={tile} value={tile}>
                    {tile}
                  </option>
                ))}
              </select>
            </label>

            <label>
              Goal tile
              <select value={goalTile} onChange={(event) => setGoalTile(event.target.value as TileId)}>
                {allTiles.map((tile: TileId) => (
                  <option key={tile} value={tile}>
                    {tile}
                  </option>
                ))}
              </select>
            </label>
          </div>

          {algorithm === 'astar' && (
            <div className="heuristic-panel">
              <label className="heuristic-editor">
                Heuristic expression
                <textarea
                  className={`heuristic-input ${astarHeuristicAnalysis.error ? 'heuristic-input-invalid' : ''}`}
                  value={astarHeuristicExpression}
                  onChange={(event) => setAStarHeuristicExpression(event.target.value)}
                  rows={3}
                  spellCheck={false}
                />
              </label>
              <p className="heuristic-hint">
                Use <code>x1</code>, <code>y1</code>, <code>x2</code>, <code>y2</code> with{' '}
                <code>+</code>, <code>-</code>, <code>*</code>, <code>/</code>, <code>**</code>, and parentheses.
              </p>
              <div className="heuristic-status-row">
                <span
                  className={`heuristic-status-tag ${
                    astarHeuristicAnalysis.admissible ? 'heuristic-status-tag-true' : 'heuristic-status-tag-false'
                  }`}
                >
                  Admissible: {astarHeuristicAnalysis.admissible ? 'True' : 'False'}
                </span>
                <span
                  className={`heuristic-status-tag ${
                    astarHeuristicAnalysis.consistent ? 'heuristic-status-tag-true' : 'heuristic-status-tag-false'
                  }`}
                >
                  Consistent: {astarHeuristicAnalysis.consistent ? 'True' : 'False'}
                </span>
              </div>
              {astarHeuristicAnalysis.error ? (
                <p className="heuristic-error">{astarHeuristicAnalysis.error}</p>
              ) : (
                (astarHeuristicAnalysis.admissible === false || astarHeuristicAnalysis.consistent === false) && (
                  <div className="heuristic-counterexample-grid">
                    {astarHeuristicAnalysis.admissible === false &&
                      astarHeuristicAnalysis.admissibilityViolations.length > 0 && (
                        <div className="heuristic-counterexample-card">
                          <h3>Not admissible</h3>
                          <ul className="heuristic-counterexample-list">
                            {astarHeuristicAnalysis.admissibilityViolations.map((violation) => (
                              <li key={`admissible-${violation.tile}`}>
                                {formatAdmissibilityViolation(violation)}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    {astarHeuristicAnalysis.consistent === false &&
                      astarHeuristicAnalysis.consistencyViolations.length > 0 && (
                        <div className="heuristic-counterexample-card">
                          <h3>Not consistent</h3>
                          <ul className="heuristic-counterexample-list">
                            {astarHeuristicAnalysis.consistencyViolations.map((violation) => (
                              <li key={`consistent-${violation.tile}-${violation.action}-${violation.neighbor}`}>
                                {formatConsistencyViolation(violation)}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                  </div>
                )
              )}
            </div>
          )}

          {algorithm === 'mcts' && (
            <div className="control-grid">
              <label>
                UCT c
                <input
                  type="number"
                  min="0"
                  step="0.1"
                  value={formatInputNumber(mctsExplorationC)}
                  onChange={(event) => setMctsExplorationC(Number(event.target.value) || 0)}
                />
              </label>
              <label>
                Iterations / step
                <input
                  type="number"
                  min="1"
                  step="1"
                  value={mctsIterations}
                  onChange={(event) => setMctsIterations(Math.max(1, Math.floor(Number(event.target.value) || 1)))}
                />
              </label>
              <label>
                Horizon
                <input
                  type="number"
                  min="1"
                  step="1"
                  value={mctsHorizon}
                  onChange={(event) => setMctsHorizon(Math.max(1, Math.floor(Number(event.target.value) || 1)))}
                />
              </label>
              <label>
                Gamma
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.05"
                  value={formatInputNumber(mctsGamma)}
                  onChange={(event) =>
                    setMctsGamma(Math.min(1, Math.max(0, Number(event.target.value) || 0)))
                  }
                />
              </label>
              <label>
                Goal reward
                <input
                  type="number"
                  value={formatInputNumber(mctsGoalReward)}
                  onChange={(event) => setMctsGoalReward(Number(event.target.value) || 0)}
                />
              </label>
              <label>
                Trap reward
                <input
                  type="number"
                  value={formatInputNumber(mctsTrapReward)}
                  onChange={(event) => setMctsTrapReward(Number(event.target.value) || 0)}
                />
              </label>
              <label>
                Stuck reward
                <input
                  type="number"
                  value={formatInputNumber(mctsStuckReward)}
                  onChange={(event) => setMctsStuckReward(Number(event.target.value) || 0)}
                />
              </label>
              <label>
                Normal reward
                <input
                  type="number"
                  value={formatInputNumber(mctsNormalReward)}
                  onChange={(event) => setMctsNormalReward(Number(event.target.value) || 0)}
                />
              </label>
            </div>
          )}

          <div className="control-row">
            <button
              className="primary-button"
              type="button"
              onClick={runAlgorithm}
              disabled={algorithm === 'astar' && astarHeuristicAnalysis.error !== null}
            >
              Start
            </button>
            <button
              className="secondary-button"
              type="button"
              onClick={handleReset}
            >
              Reset
            </button>
            <button
              className="secondary-button"
              type="button"
              onClick={() => {
                setIsAutoPlay(false)
                setDemoIndex(0)
              }}
              disabled={demoFrames.length === 0}
            >
              Rewind
            </button>
            <button
              className="secondary-button"
              type="button"
              onClick={() => setIsAutoPlay((value) => !value)}
              disabled={demoFrames.length === 0}
            >
              {isAutoPlay ? 'Auto-play: On' : 'Auto-play: Off'}
            </button>
          </div>

          <label className="speed-row">
            Simulation auto-play delay: {playbackMs} ms/frame
            <input
              type="number"
              min="1"
              max="5000"
              step="1"
              value={playbackMs}
              onChange={(event) => setPlaybackMs(clampPlaybackMs(Number(event.target.value)))}
            />
            <input
              type="range"
              min="1"
              max="5000"
              step="1"
              value={playbackMs}
              onChange={(event) => setPlaybackMs(clampPlaybackMs(Number(event.target.value)))}
            />
          </label>

        </section>

        {displayedAStarFrame && (
          <section className="card full-span-card">
            <div className="tree-panel trace-tree-panel">
              <div className="tree-panel-layout">
                <div className="tree-flow-shell">
                  <div className="mcts-flow astar-flow">
                    <ReactFlow
                      nodes={astarGraphDiagram.nodes}
                      edges={astarGraphDiagram.edges}
                      nodeTypes={searchNodeTypes}
                      fitView
                      fitViewOptions={{ padding: 0.18 }}
                      nodesDraggable={false}
                      nodesConnectable={false}
                      elementsSelectable={false}
                      zoomOnDoubleClick={false}
                      minZoom={0.3}
                      maxZoom={1.6}
                      proOptions={{ hideAttribution: true }}
                    >
                      <Background gap={24} color="#d9e1ee" />
                      <Controls showInteractive={false} position="top-right" />
                    </ReactFlow>
                  </div>
                </div>
                {astarTrace && (
                  <AlgorithmTracePanel
                    trace={astarTrace}
                    controls={[
                      {
                        label: 'Frame',
                        value: `${demoIndex + 1} / ${demoFrames.length}`,
                        onPrevious: goToPreviousFrame,
                        onNext: goToNextFrame,
                        canPrevious: canStepBack,
                        canNext: canStepForward,
                      },
                    ]}
                  />
                )}
              </div>
            </div>
          </section>
        )}

        {currentMCTSFrame && (
          <section className="card full-span-card">
            <div className="tree-panel trace-tree-panel">
              <div className="tree-panel-layout">
                <div className="tree-flow-shell">
                  <div className="mcts-flow">
                    <ReactFlow
                      nodes={currentMCTSTreeDiagram.nodes}
                      edges={currentMCTSTreeDiagram.edges}
                      nodeTypes={searchNodeTypes}
                      fitView
                      fitViewOptions={{ padding: currentMCTSTreeDiagram.fitViewPadding }}
                      nodesDraggable={false}
                      nodesConnectable={false}
                      elementsSelectable={false}
                      zoomOnDoubleClick={false}
                      minZoom={currentMCTSTreeDiagram.minZoom}
                      maxZoom={1.5}
                      proOptions={{ hideAttribution: true }}
                    >
                      <Background gap={24} color="#d9e1ee" />
                      <Controls showInteractive={false} position="top-right" />
                    </ReactFlow>
                  </div>
                </div>
                {mctsTrace && (
                  <AlgorithmTracePanel
                    trace={mctsTrace}
                    variant="mcts"
                    controls={[
                      {
                        label: 'Frame',
                        value: `${demoIndex + 1} / ${demoFrames.length}`,
                        onPrevious: goToPreviousFrame,
                        onNext: goToNextFrame,
                        canPrevious: canStepBack,
                        canNext: canStepForward,
                      },
                      ...(mctsGraphHistory.length > 0
                        ? [
                            {
                              label: 'Decision step',
                              value: `${(selectedHistoricalMCTSGraphIndex ?? 0) + 1}`,
                              onPrevious: goToPreviousHistoricalGraph,
                              onNext: goToNextHistoricalGraph,
                              canPrevious: canViewPreviousHistoricalGraph,
                              canNext: canViewNextHistoricalGraph,
                            },
                          ]
                        : []),
                    ]}
                  />
                )}
              </div>
            </div>
          </section>
        )}
      </div>
    </main>
  )
}

export default App
