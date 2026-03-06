import {
  ACTIONS,
  COLUMNS,
  deterministicTransition,
  expectedTurnsForDestination,
  getAllTiles,
  type Action,
  type TileId,
} from './game'

const ALLOWED_IDENTIFIERS = new Set(['x1', 'y1', 'x2', 'y2'])
const FLOAT_TOLERANCE = 1e-9

type HeuristicIdentifier = 'x1' | 'y1' | 'x2' | 'y2'

type Token =
  | { type: 'number'; value: number }
  | { type: 'identifier'; value: HeuristicIdentifier }
  | { type: 'operator'; value: '+' | '-' | '*' | '/' | '**' }
  | { type: 'paren'; value: '(' | ')' }

type ExpressionNode =
  | { kind: 'number'; value: number }
  | { kind: 'identifier'; value: HeuristicIdentifier }
  | { kind: 'unary'; operator: '+' | '-'; operand: ExpressionNode }
  | {
      kind: 'binary'
      operator: '+' | '-' | '*' | '/' | '**'
      left: ExpressionNode
      right: ExpressionNode
    }

type HeuristicCoordinates = {
  x: number
  y: number
}

type CompiledHeuristic = {
  expression: string
  evaluate: (source: TileId, target: TileId) => number
}

type ReverseEdge = {
  from: TileId
  cost: number
}

export const DEFAULT_ASTAR_HEURISTIC_EXPRESSION = '((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5'

export type AStarAdmissibilityViolation = {
  tile: TileId
  heuristic: number
  trueCost: number
  excess: number
}

export type AStarConsistencyViolation = {
  tile: TileId
  action: Action
  neighbor: TileId
  heuristic: number
  stepCost: number
  neighborHeuristic: number
  excess: number
}

export type AStarHeuristicAnalysis = {
  expression: string
  error: string | null
  admissible: boolean | null
  consistent: boolean | null
  admissibilityViolations: AStarAdmissibilityViolation[]
  consistencyViolations: AStarConsistencyViolation[]
}

function parseTileCoordinates(tile: TileId): HeuristicCoordinates {
  return {
    x: COLUMNS.indexOf(tile[1] as (typeof COLUMNS)[number]),
    y: Number(tile[0]),
  }
}

function tokenizeExpression(expression: string): Token[] {
  const tokens: Token[] = []
  let index = 0

  while (index < expression.length) {
    const character = expression[index]

    if (/\s/.test(character)) {
      index += 1
      continue
    }

    const numberMatch = expression.slice(index).match(/^(?:\d+(?:\.\d*)?|\.\d+)/)
    if (numberMatch) {
      tokens.push({
        type: 'number',
        value: Number(numberMatch[0]),
      })
      index += numberMatch[0].length
      continue
    }

    const identifierMatch = expression.slice(index).match(/^[A-Za-z][A-Za-z0-9]*/)
    if (identifierMatch) {
      const identifier = identifierMatch[0]
      if (!ALLOWED_IDENTIFIERS.has(identifier)) {
        throw new Error(`Unsupported identifier "${identifier}". Use only x1, y1, x2, y2.`)
      }
      tokens.push({
        type: 'identifier',
        value: identifier as HeuristicIdentifier,
      })
      index += identifier.length
      continue
    }

    if (expression.startsWith('**', index)) {
      tokens.push({ type: 'operator', value: '**' })
      index += 2
      continue
    }

    if (character === '+' || character === '-' || character === '*' || character === '/') {
      tokens.push({
        type: 'operator',
        value: character,
      })
      index += 1
      continue
    }

    if (character === '(' || character === ')') {
      tokens.push({
        type: 'paren',
        value: character,
      })
      index += 1
      continue
    }

    throw new Error(`Unsupported character "${character}".`)
  }

  return tokens
}

class ExpressionParser {
  private index = 0

  constructor(private readonly tokens: Token[]) {}

  parse() {
    const node = this.parseExpression()
    if (this.index < this.tokens.length) {
      throw new Error('Unexpected token at the end of the expression.')
    }
    return node
  }

  private parseExpression(): ExpressionNode {
    let node = this.parseMultiplicative()

    while (true) {
      const operator = this.matchOperator('+', '-')
      if (!operator) {
        return node
      }
      node = {
        kind: 'binary',
        operator,
        left: node,
        right: this.parseMultiplicative(),
      }
    }
  }

  private parseMultiplicative(): ExpressionNode {
    let node = this.parseUnary()

    while (true) {
      const operator = this.matchOperator('*', '/')
      if (!operator) {
        return node
      }
      node = {
        kind: 'binary',
        operator,
        left: node,
        right: this.parseUnary(),
      }
    }
  }

  private parseUnary(): ExpressionNode {
    const operator = this.matchOperator('+', '-')
    if (operator) {
      return {
        kind: 'unary',
        operator,
        operand: this.parseUnary(),
      }
    }

    return this.parsePower()
  }

  private parsePower(): ExpressionNode {
    let node = this.parsePrimary()

    if (this.matchSpecificOperator('**')) {
      node = {
        kind: 'binary',
        operator: '**',
        left: node,
        right: this.parseUnary(),
      }
    }

    return node
  }

  private parsePrimary(): ExpressionNode {
    const token = this.tokens[this.index]
    if (!token) {
      throw new Error('Incomplete expression.')
    }

    if (token.type === 'number') {
      this.index += 1
      return { kind: 'number', value: token.value }
    }

    if (token.type === 'identifier') {
      this.index += 1
      return { kind: 'identifier', value: token.value }
    }

    if (token.type === 'paren' && token.value === '(') {
      this.index += 1
      const node = this.parseExpression()
      const closingToken = this.tokens[this.index]
      if (!closingToken || closingToken.type !== 'paren' || closingToken.value !== ')') {
        throw new Error('Missing closing parenthesis.')
      }
      this.index += 1
      return node
    }

    throw new Error('Expected a number, variable, or parenthesized expression.')
  }

  private matchOperator<T extends '+' | '-' | '*' | '/'>(
    ...operators: T[]
  ): T | null {
    const token = this.tokens[this.index]
    if (token?.type !== 'operator') {
      return null
    }
    if (!operators.includes(token.value as T)) {
      return null
    }
    this.index += 1
    return token.value as T
  }

  private matchSpecificOperator(operator: '**') {
    const token = this.tokens[this.index]
    if (token?.type !== 'operator' || token.value !== operator) {
      return false
    }
    this.index += 1
    return true
  }
}

function evaluateExpression(
  node: ExpressionNode,
  variables: Record<HeuristicIdentifier, number>,
): number {
  switch (node.kind) {
    case 'number':
      return node.value
    case 'identifier':
      return variables[node.value]
    case 'unary': {
      const operand = evaluateExpression(node.operand, variables)
      return node.operator === '-' ? -operand : operand
    }
    case 'binary': {
      const left = evaluateExpression(node.left, variables)
      const right = evaluateExpression(node.right, variables)

      switch (node.operator) {
        case '+':
          return left + right
        case '-':
          return left - right
        case '*':
          return left * right
        case '/':
          return left / right
        case '**':
          return left ** right
      }
    }
  }
}

function compileHeuristic(expression: string): CompiledHeuristic {
  const normalizedExpression = expression.trim()

  if (!normalizedExpression) {
    throw new Error('Enter a heuristic expression.')
  }

  const tokens = tokenizeExpression(normalizedExpression)
  if (tokens.length === 0) {
    throw new Error('Enter a heuristic expression.')
  }

  const parser = new ExpressionParser(tokens)
  const ast = parser.parse()

  return {
    expression: normalizedExpression,
    evaluate: (source: TileId, target: TileId) => {
      const sourceCoordinates = parseTileCoordinates(source)
      const targetCoordinates = parseTileCoordinates(target)
      const value = evaluateExpression(ast, {
        x1: sourceCoordinates.x,
        y1: sourceCoordinates.y,
        x2: targetCoordinates.x,
        y2: targetCoordinates.y,
      })

      if (!Number.isFinite(value)) {
        throw new Error('The heuristic must evaluate to a finite number for every tile.')
      }

      return value
    },
  }
}

function buildReverseEdgeMap(goal: TileId) {
  const reverseEdges = new Map<TileId, ReverseEdge[]>()

  for (const tile of getAllTiles()) {
    for (const action of ACTIONS) {
      const neighbor = deterministicTransition(tile, action, goal).destination
      const cost = expectedTurnsForDestination(tile, action, neighbor, goal)
      const existingEdges = reverseEdges.get(neighbor) ?? []
      existingEdges.push({ from: tile, cost })
      reverseEdges.set(neighbor, existingEdges)
    }
  }

  return reverseEdges
}

function computeExactGoalCosts(goal: TileId) {
  const distances: Partial<Record<TileId, number>> = { [goal]: 0 }
  const visited = new Set<TileId>()
  const reverseEdges = buildReverseEdgeMap(goal)
  const allTiles = getAllTiles()

  while (visited.size < allTiles.length) {
    let currentTile: TileId | null = null
    let currentDistance = Number.POSITIVE_INFINITY

    for (const tile of allTiles) {
      if (visited.has(tile)) {
        continue
      }
      const distance = distances[tile] ?? Number.POSITIVE_INFINITY
      if (distance < currentDistance) {
        currentTile = tile
        currentDistance = distance
      }
    }

    if (currentTile === null || !Number.isFinite(currentDistance)) {
      break
    }

    visited.add(currentTile)

    for (const edge of reverseEdges.get(currentTile) ?? []) {
      if (visited.has(edge.from)) {
        continue
      }

      const candidateDistance = currentDistance + edge.cost
      if (candidateDistance < (distances[edge.from] ?? Number.POSITIVE_INFINITY)) {
        distances[edge.from] = candidateDistance
      }
    }
  }

  return distances
}

export function analyzeAStarHeuristic(expression: string, goal: TileId): AStarHeuristicAnalysis {
  let compiled: CompiledHeuristic

  try {
    compiled = compileHeuristic(expression)
  } catch (error) {
    return {
      expression: expression.trim(),
      error: error instanceof Error ? error.message : 'Invalid heuristic expression.',
      admissible: null,
      consistent: null,
      admissibilityViolations: [],
      consistencyViolations: [],
    }
  }

  const heuristicByTile: Partial<Record<TileId, number>> = {}

  try {
    for (const tile of getAllTiles()) {
      heuristicByTile[tile] = compiled.evaluate(tile, goal)
    }
  } catch (error) {
    return {
      expression: compiled.expression,
      error: error instanceof Error ? error.message : 'Invalid heuristic expression.',
      admissible: null,
      consistent: null,
      admissibilityViolations: [],
      consistencyViolations: [],
    }
  }

  const exactGoalCosts = computeExactGoalCosts(goal)
  const admissibilityViolations = getAllTiles()
    .flatMap((tile) => {
      const heuristic = heuristicByTile[tile]
      const trueCost = exactGoalCosts[tile]
      if (heuristic === undefined || trueCost === undefined || !Number.isFinite(trueCost)) {
        return []
      }
      const excess = heuristic - trueCost
      if (excess <= FLOAT_TOLERANCE) {
        return []
      }
      return [
        {
          tile,
          heuristic,
          trueCost,
          excess,
        } satisfies AStarAdmissibilityViolation,
      ]
    })
    .sort((left, right) => right.excess - left.excess)
    .slice(0, 3)

  const consistencyViolations = getAllTiles()
    .flatMap((tile) =>
      ACTIONS.flatMap((action) => {
        const neighbor = deterministicTransition(tile, action, goal).destination
        const stepCost = expectedTurnsForDestination(tile, action, neighbor, goal)
        const heuristic = heuristicByTile[tile]
        const neighborHeuristic = heuristicByTile[neighbor]

        if (heuristic === undefined || neighborHeuristic === undefined) {
          return []
        }

        const excess = heuristic - (stepCost + neighborHeuristic)
        if (excess <= FLOAT_TOLERANCE) {
          return []
        }

        return [
          {
            tile,
            action,
            neighbor,
            heuristic,
            stepCost,
            neighborHeuristic,
            excess,
          } satisfies AStarConsistencyViolation,
        ]
      }),
    )
    .sort((left, right) => right.excess - left.excess)
    .slice(0, 3)

  return {
    expression: compiled.expression,
    error: null,
    admissible: admissibilityViolations.length === 0,
    consistent: consistencyViolations.length === 0,
    admissibilityViolations,
    consistencyViolations,
  }
}

export function getAStarHeuristicDefinition(expression: string, goal: TileId) {
  const analysis = analyzeAStarHeuristic(expression, goal)
  const fallbackHeuristic = compileHeuristic(DEFAULT_ASTAR_HEURISTIC_EXPRESSION)

  if (analysis.error) {
    return {
      analysis,
      evaluator: fallbackHeuristic.evaluate,
      expression: fallbackHeuristic.expression,
    }
  }

  const compiled = compileHeuristic(expression)
  return {
    analysis,
    evaluator: compiled.evaluate,
    expression: compiled.expression,
  }
}
