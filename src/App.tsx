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

type ExecutionFrame = {
  kind: 'execution'
  step: number
  phase: 'execution'
  message: string
  tile: TileId
  from: TileId | null
  action: Action | null
  path: TileId[]
}

type TokenAnimation = {
  key: number
  className: 'avatar-slide' | 'avatar-bump'
  style: CSSProperties
}

type DemoFrame = AStarFrame | MCTSFrame | ExecutionFrame

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
    })
  }

  return frames
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

const SEARCH_NODE_WIDTH = 256
const SEARCH_NODE_HEIGHT = 188
const SEARCH_NODE_BOARD_SIZE = 110

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

function SearchFlowNodeCard({ data }: NodeProps<SearchFlowNodeType>) {
  if (data.mode === 'mcts') {
    const node = data.snapshot

    return (
      <div className={`mcts-flow-node ${data.active ? 'mcts-flow-node-active' : ''}`}>
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
          size={SEARCH_NODE_BOARD_SIZE}
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
        <span>h={data.snapshot.h}</span>
        <span>f={formatAStarScore(data.snapshot.f)}</span>
        <span>{data.snapshot.status}</span>
      </div>
      <Handle type="source" position={Position.Bottom} className="mcts-flow-handle" />
    </div>
  )
}

function buildMCTSTreeDiagram(frame: MCTSFrame, rolloutPathByNodeId: Map<number, TileId[]>, goalTile: TileId) {
  const { tree, activeNodeId } = frame
  const graph = new dagre.graphlib.Graph()
  graph.setDefaultEdgeLabel(() => ({}))
  graph.setGraph({
    rankdir: 'TB',
    nodesep: 46,
    ranksep: 86,
    marginx: 28,
    marginy: 28,
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
      width: SEARCH_NODE_WIDTH,
      height: SEARCH_NODE_HEIGHT,
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
      },
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top,
      draggable: false,
      selectable: false,
      position: {
        x: position.x - SEARCH_NODE_WIDTH / 2,
        y: position.y - SEARCH_NODE_HEIGHT / 2,
      },
      width: SEARCH_NODE_WIDTH,
      height: SEARCH_NODE_HEIGHT,
    }
  })

  const edges: Edge[] = orderedTree.flatMap((node) => {
    if (node.parentId === null) {
      return []
    }

    const isActive = node.id === activeNodeId
    return [
      {
        id: `${node.parentId}-${node.id}`,
        source: String(node.parentId),
        target: String(node.id),
        type: 'smoothstep',
        animated: isActive,
        markerEnd: {
          type: MarkerType.ArrowClosed,
          width: 18,
          height: 18,
          color: isActive ? '#2563eb' : '#9fb3d8',
        },
        style: {
          stroke: isActive ? '#2563eb' : '#9fb3d8',
          strokeWidth: isActive ? 2.8 : 2.2,
        },
      },
    ]
  })

  return { nodes, edges }
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

  const renderTree: AStarRenderNodeSnapshot[] = frame.tree.map((node) => ({
    id: String(node.id),
    tile: node.tile,
    parentId: node.parentId === null ? null : String(node.parentId),
    actionFromParent: node.actionFromParent,
    depth: node.depth,
    g: node.g,
    h: node.h,
    f: node.f,
    status: node.status,
  }))

  const rejectedChild = frame.rejectedChild
  if (rejectedChild) {
    const currentNode = frame.tree.find((node) => node.id === rejectedChild.parentTreeNodeId)
    renderTree.push({
      id: `rejected-${frame.step}`,
      tile: rejectedChild.tile,
      parentId: String(rejectedChild.parentTreeNodeId),
      actionFromParent: rejectedChild.actionFromParent,
      depth: (currentNode?.depth ?? 0) + 1,
      g: rejectedChild.g,
      h: rejectedChild.h,
      f: rejectedChild.f,
      status: 'rejected',
      rejectionReason: rejectedChild.reason,
    })
  }

  const orderedTree = renderTree.sort((left, right) => {
    const depthDiff = left.depth - right.depth
    if (depthDiff !== 0) {
      return depthDiff
    }

    const leftParent = left.parentId ?? ''
    const rightParent = right.parentId ?? ''
    if (leftParent !== rightParent) {
      return leftParent.localeCompare(rightParent)
    }

    const fDiff = left.f - right.f
    if (fDiff !== 0) {
      return fDiff
    }

    return left.id.localeCompare(right.id)
  })

  for (const node of orderedTree) {
    graph.setNode(node.id, {
      width: SEARCH_NODE_WIDTH,
      height: SEARCH_NODE_HEIGHT,
    })
  }

  for (const node of orderedTree) {
    if (node.parentId !== null) {
      graph.setEdge(node.parentId, node.id)
    }
  }

  dagre.layout(graph)
  const nodeById = new Map(orderedTree.map((node) => [node.id, node] as const))
  const activePathEdgeSet = new Set<string>()
  let pathCursorId = frame.activeTreeNodeId === null ? null : String(frame.activeTreeNodeId)

  while (pathCursorId !== null) {
    const node = nodeById.get(pathCursorId)
    if (!node || node.parentId === null) {
      break
    }
    activePathEdgeSet.add(`${node.parentId}-${node.id}`)
    pathCursorId = node.parentId
  }

  const nodes: SearchFlowNodeType[] = orderedTree.map((node) => {
    const position = graph.node(node.id)

    return {
      id: node.id,
      type: 'searchNode',
      data: {
        mode: 'astar',
        snapshot: node,
        previewPath: buildAStarTreePath(node.id, nodeById),
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

  const edges: Edge[] = orderedTree.flatMap((node) => {
    if (node.parentId === null) {
      return []
    }

    const edgeId = `${node.parentId}-${node.id}`
    const isActivePath = activePathEdgeSet.has(edgeId)
    const stroke =
      node.status === 'rejected'
        ? '#dc2626'
        : node.status === 'active' || isActivePath
        ? '#2563eb'
        : node.status === 'queued'
          ? '#d97706'
          : '#94a3b8'

    return [
      {
        id: edgeId,
        source: String(node.parentId),
        target: String(node.id),
        type: 'smoothstep',
        animated: node.status === 'active' || node.status === 'rejected',
        label: node.actionFromParent ?? '',
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
          strokeDasharray: node.status === 'rejected' ? '6 4' : undefined,
          strokeWidth: node.status === 'rejected' ? 2.2 : isActivePath ? 2.6 : 2,
        },
      },
    ]
  })

  return { nodes, edges }
}

function getSelectionTiles(frame: MCTSFrame) {
  const nodeById = new Map(frame.tree.map((node) => [node.id, node]))

  return frame.selectionPathIds.flatMap((nodeId) => {
    const node = nodeById.get(nodeId)
    return node ? [node.tile] : []
  })
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

function App() {
  const [currentTile, setCurrentTile] = useState<TileId>(() => getRandomStartTile())
  const [algorithm, setAlgorithm] = useState<AlgorithmKind>('astar')
  const [goalTile, setGoalTile] = useState<TileId>('2F')
  const [demoFrames, setDemoFrames] = useState<DemoFrame[]>([])
  const [demoIndex, setDemoIndex] = useState(0)
  const [isAutoPlay, setIsAutoPlay] = useState(false)
  const [playbackMs, setPlaybackMs] = useState(550)
  const [mctsIterations, setMctsIterations] = useState(40)
  const [mctsExplorationC, setMctsExplorationC] = useState(Math.SQRT2)
  const [mctsGamma, setMctsGamma] = useState(0.9)
  const [mctsHorizon, setMctsHorizon] = useState(100)
  const [tokenAnimation, setTokenAnimation] = useState<TokenAnimation | null>(null)
  const tokenAnimationCounterRef = useRef(0)
  const tokenAnimationTimerRef = useRef<number | null>(null)

  const allTiles = useMemo(() => getAllTiles(), [])
  const activeFrame = demoFrames[demoIndex] ?? null
  const latestSearchFrame = useMemo(
    () => getLatestSearchFrame(demoFrames, demoIndex),
    [demoFrames, demoIndex],
  )
  const displayedAStarFrame = isAStarFrame(latestSearchFrame) ? latestSearchFrame : null
  const displayedMCTSFrame = isMCTSFrame(latestSearchFrame) ? latestSearchFrame : null
  const visibleTile = getFrameTile(activeFrame, currentTile)
  const currentType = getTileType(visibleTile)
  const restingTokenPosition = getTokenPosition(visibleTile)

  const astarOpenSet = isAStarFrame(activeFrame) ? new Set(activeFrame.openSet) : null
  const astarClosedSet = isAStarFrame(activeFrame) ? new Set(activeFrame.closedSet) : null
  const mctsRolloutTiles = isMCTSFrame(activeFrame) ? new Set(activeFrame.rolloutTiles) : null
  const astarPathTiles = isAStarFrame(activeFrame) ? new Set(activeFrame.pathPreview) : null
  const astarPathSegments = isAStarFrame(activeFrame) ? buildPathSegments(activeFrame.pathPreview) : []
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
  const mctsSelectionTiles = isMCTSFrame(activeFrame) ? getSelectionTiles(activeFrame) : []
  const mctsSelectionTileSet = isMCTSFrame(activeFrame) ? new Set(mctsSelectionTiles) : null
  const mctsSelectionSegments = isMCTSFrame(activeFrame) ? buildPathSegments(mctsSelectionTiles) : []
  const mctsRolloutSegments = isMCTSFrame(activeFrame) ? buildPathSegments(activeFrame.rolloutTiles) : []
  const mctsRolloutPathByNodeId = useMemo(() => {
    const rolloutPathMap = new Map<number, TileId[]>()

    for (const frame of demoFrames.slice(0, demoIndex + 1)) {
      if (!isMCTSFrame(frame) || frame.phase !== 'rollout' || frame.activeNodeId === null) {
        continue
      }

      rolloutPathMap.set(frame.activeNodeId, [...frame.rolloutTiles])
    }

    return rolloutPathMap
  }, [demoFrames, demoIndex])
  const mctsTreeDiagram = useMemo(
    () =>
      displayedMCTSFrame
        ? buildMCTSTreeDiagram(displayedMCTSFrame, mctsRolloutPathByNodeId, goalTile)
        : {
            nodes: [] as SearchFlowNodeType[],
            edges: [] as Edge[],
          },
    [displayedMCTSFrame, goalTile, mctsRolloutPathByNodeId],
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
  const displayedMCTSSelectionTiles = displayedMCTSFrame ? getSelectionTiles(displayedMCTSFrame) : []
  const mctsPlaybackTiles = displayedMCTSFrame
    ? displayedMCTSFrame.phase === 'selection'
      ? displayedMCTSSelectionTiles
      : displayedMCTSFrame.rolloutTiles.length > 0
        ? displayedMCTSFrame.rolloutTiles
        : displayedMCTSSelectionTiles
    : []
  const mctsPlaybackFocusTile =
    mctsPlaybackTiles[mctsPlaybackTiles.length - 1] ?? (displayedMCTSFrame ? visibleTile : currentTile)
  const astarCurrentScores =
    displayedAStarFrame && displayedAStarFrame.current
      ? astarScoreMap.get(displayedAStarFrame.current) ?? null
      : null

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

  const handleReset = useCallback(() => {
    clearTokenAnimation()
    setCurrentTile(getRandomStartTile())
    setDemoFrames([])
    setDemoIndex(0)
    setIsAutoPlay(false)
  }, [clearTokenAnimation])

  const performStep = useCallback(
    (action: Action) => {
      const sampledTransition = sampleTransition(currentTile, action, Math.random, goalTile)

      animateTokenMove(currentTile, sampledTransition.destination, action)
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

      if (event.key === 'r' || event.key === 'R') {
        event.preventDefault()
        handleReset()
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
  }, [handleReset, performStep])

  const runAlgorithm = useCallback(() => {
    clearTokenAnimation()
    setIsAutoPlay(true)
    if (algorithm === 'astar') {
      const result = runAStarDemo({ start: currentTile, goal: goalTile })
      const executionFrames = buildExecutionFrames(result.path, result.actions).map((frame, index) => ({
        ...frame,
        step: result.frames.length + index,
      }))
      setDemoFrames([...result.frames, ...executionFrames])
      setDemoIndex(0)
      return
    }

    const result = runMCTSDemo({
      start: currentTile,
      goal: goalTile,
      iterations: mctsIterations,
      explorationConstant: mctsExplorationC,
      gamma: mctsGamma,
      rolloutHorizon: mctsHorizon,
    })
    const executionFrames = buildExecutionFrames(result.recommendedPath, result.recommendedActions).map(
      (frame, index) => ({
        ...frame,
        step: result.frames.length + index,
      }),
    )
    setDemoFrames([...result.frames, ...executionFrames])
    setDemoIndex(0)
  }, [algorithm, clearTokenAnimation, currentTile, goalTile, mctsExplorationC, mctsGamma, mctsHorizon, mctsIterations])

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
      <section className="hero card">
        <p className="eyebrow">Text RPG visualizer</p>
        <h1>Dungeon Crawler</h1>
        <ul className="hero-rules">
          <li>The room is a 3x6 grid with rows 1-3 and columns A-F.</li>
          <li>You start at time 0 on a uniformly random tile.</li>
          <li>
            In this interactive version, you choose an action each step: up, down,
            left, or right.
          </li>
          <li>
            Normal tiles (<strong>N</strong>) follow the chosen action.
          </li>
          <li>
            Stuck tiles (<strong>S</strong>) have a 50% chance to keep you in place;
            otherwise they follow the chosen action.
          </li>
          <li>
            Trap tiles (<strong>T</strong>) are absorbing, so once entered you never
            leave.
          </li>
          <li>
            If a move hits a wall, including the special walls around <strong>2A</strong> and
            the corridor walls in the right half, you stay in the same tile for that time step.
          </li>
          <li>
            Your observation is only the type of tile you are currently standing on:
            <strong> N</strong>, <strong>S</strong>, or <strong>T</strong>.
          </li>
        </ul>
      </section>

      <section className="layout">
        <div className="card board-card">
          <div className="section-heading">
            <h2>Room Map</h2>
            <p>
              Displayed tile: <strong>{visibleTile}</strong> ({currentType})
            </p>
          </div>

          <div
            className="board-grid"
            style={
              {
                '--board-columns': COLUMNS.length,
                '--board-rows': BOARD_ROWS.length,
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
                  const tileType = getTileType(tileId)
                  const isCurrent = tileId === visibleTile
                  const isGoal = tileId === goalTile
                  const isOpen = astarOpenSet?.has(tileId) ?? false
                  const isClosed = astarClosedSet?.has(tileId) ?? false
                  const isRollout = mctsRolloutTiles?.has(tileId) ?? false
                  const isAStarPath = astarPathTiles?.has(tileId) ?? false
                  const isSelectionPath = mctsSelectionTileSet?.has(tileId) ?? false
                  const scoreRow = astarScoreMap.get(tileId)

                  return (
                    <div
                      className={`board-cell ${tileClassName(tileId)} ${isCurrent ? 'current-cell' : ''} ${isGoal ? 'goal-cell' : ''} ${isOpen ? 'open-cell' : ''} ${isClosed ? 'closed-cell' : ''} ${isRollout ? 'rollout-cell' : ''} ${isAStarPath ? 'path-preview-cell' : ''} ${isSelectionPath ? 'selection-path-cell' : ''}`}
                      key={tileId}
                    >
                      <span className="cell-id">{tileId}</span>
                      <span className="cell-type">{isGoal ? 'G' : tileType}</span>
                      {scoreRow && (scoreRow.g !== null || scoreRow.f !== null) && (
                        <span className="cell-score">
                          g:{scoreRow.g ?? '-'} h:{scoreRow.h} f:{scoreRow.f ?? '-'}
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
                          strokeWidth: String(0.05 + progress * 0.05),
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
                {isMCTSFrame(activeFrame) &&
                  mctsSelectionSegments.map((segment, index) => (
                    <line
                      key={`selection-${segment.start}-${segment.end}-${index}`}
                      x1={segment.startPoint.x}
                      y1={segment.startPoint.y}
                      x2={segment.endPoint.x}
                      y2={segment.endPoint.y}
                      className="mcts-selection-segment"
                    />
                  ))}
                {isMCTSFrame(activeFrame) &&
                  mctsRolloutSegments.map((segment, index) => {
                    const progress = (index + 1) / Math.max(1, mctsRolloutSegments.length)
                    return (
                      <line
                        key={`rollout-${segment.start}-${segment.end}-${index}`}
                        x1={segment.startPoint.x}
                        y1={segment.startPoint.y}
                        x2={segment.endPoint.x}
                        y2={segment.endPoint.y}
                        className="mcts-rollout-segment"
                        style={{ opacity: String(0.3 + progress * 0.7) }}
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
              <strong>Arrows</strong>: move and step
            </span>
            <span>
              <strong>R</strong>: reset
            </span>
            <span>
              <strong>Goal</strong>: {goalTile}
            </span>
          </div>

        </div>

        <div className="sidebar">
          <section className="card">
            <div className="section-heading">
              <h2>Algorithm Demo</h2>
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

            {algorithm === 'mcts' ? (
              <div className="control-grid">
                <label>
                  UCT c
                  <input
                    type="number"
                    min="0"
                    step="0.1"
                    value={mctsExplorationC}
                    onChange={(event) => setMctsExplorationC(Number(event.target.value) || 0)}
                  />
                </label>
                <label>
                  Iterations
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
                    value={mctsGamma}
                    onChange={(event) =>
                      setMctsGamma(Math.min(1, Math.max(0, Number(event.target.value) || 0)))
                    }
                  />
                </label>
              </div>
            ) : (
              <p className="explanation-text">
                A* uses expected turn cost for edge weights. A normal move costs 1 turn, while
                leaving a stuck tile in the chosen direction costs 2 expected turns when the move
                succeeds with probability 0.5.
              </p>
            )}

            <div className="control-row">
              <button className="primary-button" type="button" onClick={runAlgorithm}>
                Start
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
                onClick={() => {
                  setIsAutoPlay(false)
                  setDemoIndex((index) => Math.max(0, index - 1))
                }}
                disabled={!canStepBack}
              >
                Step Back
              </button>
              <button
                className="secondary-button"
                type="button"
                onClick={() => {
                  setIsAutoPlay(false)
                  setDemoIndex((index) => Math.min(index + 1, demoFrames.length - 1))
                }}
                disabled={!canStepForward}
              >
                Step Forward
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
                min="120"
                max="1200"
                step="20"
                value={playbackMs}
                onChange={(event) =>
                  setPlaybackMs(
                    Math.min(1200, Math.max(120, Number(event.target.value) || 120)),
                  )
                }
              />
              <input
                type="range"
                min="120"
                max="1200"
                step="20"
                value={playbackMs}
                onChange={(event) => setPlaybackMs(Number(event.target.value))}
              />
            </label>

            {activeFrame ? (
              <div className="probability-block">
                <h3>
                  Frame {demoIndex + 1} / {demoFrames.length}
                </h3>
                <p className="explanation-text">{activeFrame.message}</p>
                <p className="explanation-text">
                  Phase: <strong>{activeFrame.phase}</strong>
                </p>
              </div>
            ) : (
              <p className="empty-state">Press Start to begin stepping through the algorithm.</p>
            )}

          </section>

        </div>

        {displayedAStarFrame && (
          <section className="card full-span-card">
            <div className="mcts-visualization-header">
              <div className="section-heading">
                <h2>A* Navigation Graph</h2>
                <p>
                  This tree grows as A* discovers better states, with each new node added as a leaf
                  beneath the active state that generated it.
                </p>
              </div>
              <div className="mcts-tree-summary">
                <span>
                  <strong>Phase</strong>: {displayedAStarFrame.phase}
                </span>
                <span>
                  <strong>Current</strong>: {displayedAStarFrame.current ?? '(none)'}
                </span>
                <span>
                  <strong>Queued leaves</strong>: {displayedAStarFrame.openSet.length}
                </span>
                <span>
                  <strong>Pruned</strong>: {displayedAStarFrame.closedSet.length}
                </span>
              </div>
            </div>

            <div className="tree-panel">
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

            <div className="mcts-rollout-panel astar-detail-panel">
              <div className="mcts-rollout-copy">
                <h4>Live A* Metrics</h4>
                <p className="explanation-text">
                  Blue marks the active expansion, orange marks queued leaf nodes, and gray marks
                  branches that were pruned or already expanded.
                </p>
                <div className="mcts-metric-grid">
                  <div className="mcts-metric-card">
                    <span className="mcts-metric-label">Current state</span>
                    <strong>{displayedAStarFrame.current ?? '(none)'}</strong>
                  </div>
                  <div className="mcts-metric-card">
                    <span className="mcts-metric-label">Queued leaves</span>
                    <strong>{displayedAStarFrame.openSet.length}</strong>
                  </div>
                  <div className="mcts-metric-card">
                    <span className="mcts-metric-label">Expected turns g</span>
                    <strong>{formatAStarScore(astarCurrentScores?.g ?? null)}</strong>
                  </div>
                  <div className="mcts-metric-card">
                    <span className="mcts-metric-label">Heuristic h</span>
                    <strong>{astarCurrentScores?.h ?? '-'}</strong>
                  </div>
                  <div className="mcts-metric-card">
                    <span className="mcts-metric-label">Expected total f</span>
                    <strong>{formatAStarScore(astarCurrentScores?.f ?? null)}</strong>
                  </div>
                  <div className="mcts-metric-card">
                    <span className="mcts-metric-label">Neighbor under test</span>
                    <strong>{displayedAStarFrame.neighbor ?? '(none)'}</strong>
                  </div>
                </div>
              </div>
              <MiniBoardState
                tile={displayedAStarFrame.current ?? visibleTile}
                path={displayedAStarFrame.pathPreview.length > 0 ? displayedAStarFrame.pathPreview : [visibleTile]}
                emphasisTile={displayedAStarFrame.current ?? visibleTile}
                goalTile={goalTile}
                size={148}
                className="mcts-rollout-board"
              />
            </div>
          </section>
        )}

        {displayedMCTSFrame && (
          <section className="card full-span-card">
            <div className="mcts-visualization-header">
              <div className="section-heading">
                <h2>MCTS Tree View</h2>
                <p>
                  The full search tree is laid out by depth, with the active node highlighted and
                  enough spacing to make rollout structure easier to scan.
                </p>
              </div>
              <div className="mcts-tree-summary">
                <span>
                  <strong>Phase</strong>: {displayedMCTSFrame.phase}
                </span>
                <span>
                  <strong>Iteration</strong>: {displayedMCTSFrame.iteration}
                </span>
                <span>
                  <strong>Best root action</strong>: {displayedMCTSFrame.bestRootAction ?? '(none yet)'}
                </span>
                <span>
                  <strong>UCB c</strong>: {displayedMCTSFrame.explorationConstant.toFixed(3)}
                </span>
                <span>
                  <strong>Gamma</strong>: {displayedMCTSFrame.gamma.toFixed(3)}
                </span>
              </div>
            </div>

            <div className="tree-panel">
              <div className="mcts-flow">
                <ReactFlow
                  nodes={mctsTreeDiagram.nodes}
                  edges={mctsTreeDiagram.edges}
                  nodeTypes={searchNodeTypes}
                  fitView
                  fitViewOptions={{ padding: 0.18 }}
                  nodesDraggable={false}
                  nodesConnectable={false}
                  elementsSelectable={false}
                  zoomOnDoubleClick={false}
                  minZoom={0.25}
                  maxZoom={1.5}
                  proOptions={{ hideAttribution: true }}
                >
                  <Background gap={24} color="#d9e1ee" />
                  <Controls showInteractive={false} position="top-right" />
                </ReactFlow>
              </div>
            </div>

            <div className="mcts-rollout-panel">
              <div className="mcts-rollout-copy">
                <h4>Live Simulation Board</h4>
                <p className="explanation-text">
                  This board replays the currently visible selection or rollout path frame by frame.
                </p>
                <div className="mcts-metric-grid">
                  <div className="mcts-metric-card">
                    <span className="mcts-metric-label">Phase</span>
                    <strong>{displayedMCTSFrame.phase}</strong>
                  </div>
                  <div className="mcts-metric-card">
                    <span className="mcts-metric-label">Simulations run</span>
                    <strong>{displayedMCTSFrame.iteration}</strong>
                  </div>
                  <div className="mcts-metric-card">
                    <span className="mcts-metric-label">Current focus</span>
                    <strong>{mctsPlaybackFocusTile}</strong>
                  </div>
                  <div className="mcts-metric-card">
                    <span className="mcts-metric-label">Displayed path length</span>
                    <strong>{mctsPlaybackTiles.length}</strong>
                  </div>
                </div>
              </div>
              <MiniBoardState
                tile={mctsPlaybackFocusTile}
                path={mctsPlaybackTiles}
                emphasisTile={mctsPlaybackFocusTile}
                goalTile={goalTile}
                size={148}
                className="mcts-rollout-board"
              />
            </div>
          </section>
        )}
      </section>
    </main>
  )
}

export default App
