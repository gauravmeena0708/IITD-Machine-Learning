"use strict";

(function initHavannahRules(globalScope) {
  const DIRECTIONS = ["up", "top-left", "bottom-left", "down"];

  function toKey(x, y) {
    return `${x},${y}`;
  }

  function isValid(x, y, dim) {
    return x >= 0 && x < dim && y >= 0 && y < dim;
  }

  function getNeighbours(dim, vertex) {
    const [i, j] = vertex;
    const siz = Math.floor(dim / 2);
    const neighbours = [];

    if (i > 0) neighbours.push([i - 1, j]);
    if (i < dim - 1) neighbours.push([i + 1, j]);
    if (j > 0) neighbours.push([i, j - 1]);
    if (j < dim - 1) neighbours.push([i, j + 1]);
    if (i > 0 && j <= siz && j > 0) neighbours.push([i - 1, j - 1]);
    if (i > 0 && j >= siz && j < dim - 1) neighbours.push([i - 1, j + 1]);
    if (i < dim - 1 && j < siz) neighbours.push([i + 1, j + 1]);
    if (i < dim - 1 && j > siz) neighbours.push([i + 1, j - 1]);

    return neighbours;
  }

  function moveCoordinates(direction, half) {
    switch (direction) {
      case "up":
        return [-1, 0];
      case "down":
        return [1, 0];
      case "top-left":
        if (half <= 0) return [-1, -1];
        return [0, -1];
      case "top-right":
        if (half === 0) return [-1, 1];
        if (half < 0) return [0, 1];
        return [-1, 1];
      case "bottom-left":
        if (half <= 0) return [0, -1];
        return [1, -1];
      case "bottom-right":
        if (half === 0) return [0, 1];
        if (half < 0) return [1, 1];
        return [0, 1];
      default:
        return null;
    }
  }

  function threeForwardMoves(direction) {
    switch (direction) {
      case "up":
        return ["top-left", "up", "top-right"];
      case "down":
        return ["down", "bottom-left", "bottom-right"];
      case "top-left":
        return ["bottom-left", "top-left", "up"];
      case "top-right":
        return ["top-right", "up", "bottom-right"];
      case "bottom-left":
        return ["bottom-left", "down", "top-left"];
      case "bottom-right":
        return ["bottom-right", "down", "top-right"];
      default:
        return [];
    }
  }

  function bfsReachable(boardMask, start) {
    const dim = boardMask.length;
    const queue = [[start[0], start[1]]];
    const visited = new Set([toKey(start[0], start[1])]);
    let head = 0;

    while (head < queue.length) {
      const [x, y] = queue[head++];
      const neighbours = getNeighbours(dim, [x, y]);
      for (const [nx, ny] of neighbours) {
        if (!isValid(nx, ny, dim) || !boardMask[nx][ny]) continue;
        const key = toKey(nx, ny);
        if (visited.has(key)) continue;
        visited.add(key);
        queue.push([nx, ny]);
      }
    }

    return visited;
  }

  function getCorner(vertex, dim) {
    const [i, j] = vertex;
    const mid = Math.floor(dim / 2);
    if (i === 0 && j === 0) return 0;
    if (i === 0 && j === mid) return 1;
    if (i === 0 && j === dim - 1) return 2;
    if (i === mid && j === dim - 1) return 3;
    if (i === dim - 1 && j === mid) return 4;
    if (i === mid && j === 0) return 5;
    return -1;
  }

  function getEdge(vertex, dim) {
    const [i, j] = vertex;
    const siz = Math.floor(dim / 2);
    if (j === 0 && i > 0 && i < siz) return 0;
    if (i === 0 && j > 0 && j < siz) return 1;
    if (i === 0 && j > siz && j < dim - 1) return 2;
    if (j === dim - 1 && i > 0 && i < siz) return 3;
    if (i > siz && i < dim - 1 && i + j === 3 * siz) return 4;
    if (i > siz && i < dim - 1 && i - j === siz) return 5;
    return -1;
  }

  function getAllCorners(dim) {
    const mid = Math.floor(dim / 2);
    return [
      [0, 0],
      [0, mid],
      [0, dim - 1],
      [mid, dim - 1],
      [dim - 1, mid],
      [mid, 0],
    ];
  }

  function getAllEdges(dim) {
    const siz = Math.floor((dim + 1) / 2);
    const edges = [[], [], [], [], [], []];

    for (let i = 1; i <= siz - 2; i++) edges[0].push([0, i]);
    for (let i = siz; i <= dim - 2; i++) edges[1].push([0, i]);
    for (let i = 1; i <= siz - 2; i++) edges[2].push([i, dim - 1]);
    for (let i = 1; i <= siz - 2; i++) edges[3].push([siz - 1 + i, dim - 1 - i]);
    for (let i = 1; i <= siz - 2; i++) edges[4].push([siz - 1 + i, i]);
    for (let i = 1; i <= siz - 2; i++) edges[5].push([i, 0]);

    return edges;
  }

  function checkForkAndBridge(boardMask, move) {
    const visited = bfsReachable(boardMask, move);
    const dim = boardMask.length;
    const edges = getAllEdges(dim).map((edge) =>
      edge.map(([x, y]) => toKey(x, y))
    );
    let touchedEdges = 0;
    for (const edge of edges) {
      if (edge.some((cell) => visited.has(cell))) {
        touchedEdges += 1;
      }
      if (touchedEdges >= 3) {
        return [true, "fork"];
      }
    }

    const cornerKeys = new Set(
      getAllCorners(dim).map(([x, y]) => toKey(x, y))
    );
    const reachableCorners = [...cornerKeys].filter((corner) =>
      visited.has(corner)
    ).length;
    if (reachableCorners >= 2) {
      return [true, "bridge"];
    }

    return [false, null];
  }

  function checkRing(boardMask, move) {
    const dim = boardMask.length;
    const siz = Math.floor(dim / 2);
    const neighbours = getNeighbours(dim, move).filter(
      ([x, y]) => boardMask[x][y]
    );
    if (neighbours.length < 2) {
      return false;
    }

    const visited = new Set();
    let frontier = [];
    for (const direction of DIRECTIONS) {
      const step = moveCoordinates(direction, Math.sign(move[1] - siz));
      if (!step) continue;
      const nx = move[0] + step[0];
      const ny = move[1] + step[1];
      if (!isValid(nx, ny, dim) || !boardMask[nx][ny]) continue;
      frontier.push([[nx, ny], direction]);
      visited.add(`${nx},${ny},${direction}`);
    }

    let ringLength = 1;
    while (frontier.length > 0) {
      const next = [];
      for (const [[x, y], prevDirection] of frontier) {
        const half = Math.sign(y - siz);
        const forwardDirections = threeForwardMoves(prevDirection);
        for (const direction of forwardDirections) {
          const step = moveCoordinates(direction, half);
          if (!step) continue;
          const nx = x + step[0];
          const ny = y + step[1];
          if (!isValid(nx, ny, dim) || !boardMask[nx][ny]) continue;
          if (nx === move[0] && ny === move[1] && ringLength >= 5) {
            return true;
          }
          const key = `${nx},${ny},${direction}`;
          if (visited.has(key)) continue;
          visited.add(key);
          next.push([[nx, ny], direction]);
        }
      }
      frontier = next;
      ringLength += 1;
    }
    return false;
  }

  function checkWin(board, move, playerNum) {
    const boardMask = board.map((row) =>
      row.map((cell) => cell === playerNum)
    );

    if (checkRing(boardMask, move)) {
      return { win: true, way: "ring" };
    }

    const [win, way] = checkForkAndBridge(boardMask, move);
    if (win) {
      return { win: true, way };
    }

    return { win: false, way: null };
  }

  function getValidActions(board) {
    const moves = [];
    for (let i = 0; i < board.length; i++) {
      for (let j = 0; j < board[i].length; j++) {
        if (board[i][j] === 0) {
          moves.push([i, j]);
        }
      }
    }
    return moves;
  }

  const HavannahRules = Object.freeze({
    checkRing,
    checkWin,
    getAllCorners,
    getCorner,
    getEdge,
    getAllEdges,
    getNeighbours,
    getValidActions,
    isValid,
    moveCoordinates,
    threeForwardMoves,
    bfsReachable,
    checkForkAndBridge,
  });

  if (typeof module !== "undefined" && module.exports) {
    module.exports = HavannahRules;
  } else {
    globalScope.Havannah = HavannahRules;
  }
})(typeof window !== "undefined" ? window : globalThis);
