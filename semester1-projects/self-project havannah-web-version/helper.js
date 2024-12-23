// Helper function to check if coordinates are valid
function isValid(x, y, dim) {
  return 0 <= x && x < dim && 0 <= y && y < dim;
}

// Helper function to get all valid moves
function getValidActions(board) {
  let validMoves = [];
  for (let i = 0; i < board.length; i++) {
    for (let j = 0; j < board[i].length; j++) {
      // Only consider cells that are empty (0) and not blocked (3)
      if (board[i][j] === 0) {
        validMoves.push([i, j]);
      }
    }
  }
  return validMoves;
}

// Neighbor calculation
function getNeighbours(dim, vertex) {
  let [i, j] = vertex;
  const siz = Math.floor(dim / 2); // Size to differentiate between the two halves
  let neighbours = [];

  // Horizontal and vertical neighbors
  if (i > 0) neighbours.push([i - 1, j]);
  if (i < dim - 1) neighbours.push([i + 1, j]);
  if (j > 0) neighbours.push([i, j - 1]);
  if (j < dim - 1) neighbours.push([i, j + 1]);

  // Diagonal neighbors for hexagonal structure
  if (i > 0 && j <= siz && j > 0) neighbours.push([i - 1, j - 1]);
  if (i > 0 && j >= siz && j < dim - 1) neighbours.push([i - 1, j + 1]);
  if (i < dim - 1 && j < siz) neighbours.push([i + 1, j + 1]);
  if (i < dim - 1 && j > siz) neighbours.push([i + 1, j - 1]);

  return neighbours;
}

// BFS to find all reachable points for the current player
function bfsReachable(board, start) {
  const dim = board.length;
  const playerNum = board[start[0]][start[1]]; // Capture the current player's number
  const queue = [start];
  const visited = new Set([`${start[0]},${start[1]}`]);

  while (queue.length > 0) {
    const [x, y] = queue.shift();
    const neighbors = getNeighbours(dim, [x, y]);

    for (const [nx, ny] of neighbors) {
      if (
        isValid(nx, ny, dim) &&
        !visited.has(`${nx},${ny}`) &&
        board[nx][ny] !== 3 && // Exclude blocked tiles
        board[nx][ny] === playerNum // Same player tiles
      ) {
        queue.push([nx, ny]);
        visited.add(`${nx},${ny}`);
      }
    }
  }

  return visited;
}

// Fixing edges identification
function getAllEdges(dim) {
  const siz = Math.floor((dim + 1) / 2);
  const sides = [
    Array.from({ length: siz - 2 }, (_, i) => `${0},${i + 1}`), // Top edge (excluding corners)
    Array.from({ length: dim - siz - 1 }, (_, i) => `${i + 1},${dim - 1}`), // Right edge
    Array.from({ length: siz - 2 }, (_, i) => `${dim - 1},${siz + i}`), // Bottom edge (excluding corners)
    Array.from({ length: siz - 2 }, (_, i) => `${siz - 1 + i},${0}`), // Left edge (excluding corners)
  ];
  return sides.map((edge) => new Set(edge));
}

// Helper function to check if a side has reachable points
function sideHasReachablePoints(side, visited) {
  for (const point of visited) {
    if (side.has(point)) return true; // Check if any point in the visited set belongs to this edge
  }
  return false;
}

// Updated checkFork function to ensure only edges are considered
function checkFork(board, move) {
  const visited = bfsReachable(board, move); // Get all reachable points from the current move
  const edges = getAllEdges(board.length); // Only consider valid edges

  // Count how many distinct edges are reachable
  const reachableEdges = edges.filter((edge) =>
    sideHasReachablePoints(edge, visited)
  );

  // A fork is detected if 3 or more edges are connected
  return reachableEdges.length >= 3;
}

// Function to check both fork and bridge in a single move
function checkForkAndBridge(board, move) {
  const visited = bfsReachable(board, move);
  const corners = new Set(getAllCorners(board.length));
  const edges = getAllEdges(board.length);

  // Check for fork
  const reachableEdges = edges.map((side) =>
    sideHasReachablePoints(side, visited)
  );
  if (reachableEdges.filter(Boolean).length >= 3) {
    return [true, "fork"];
  }

  // Check for bridge
  const reachableCorners = [...visited].filter((point) =>
    corners.has(point)
  ).length;
  if (reachableCorners >= 2) {
    return [true, "bridge"];
  }

  return [false, null];
}

// Function to check if the player has won by either fork or bridge
function checkWin(board, move, playerNum) {
  // Convert the board to a boolean array where the player's positions are `true`
  board = board.map((row) => row.map((cell) => cell === playerNum));

  // Check if a ring is formed
  if (checkRing(board, move)) {
    return [true, "ring"];
  }

  // Check for fork or bridge
  const [win, way] = checkForkAndBridge(board, move);
  if (win) {
    return [true, way];
  }

  return [false, null];
}

function checkRing(board, move) {
  /*
      Check whether a ring is formed by the move.
  
      Parameters:
      - board (Array[Array[bool]]): Game board with true values at the positions of the player and false elsewhere
      - move (Array[int, int]): Position of the move. Must have already been played (marked on the board)
  
      Returns:
      - bool: True if a ring is formed by the move, False otherwise
    */

  const dim = board.length; // Get the dimension of the board
  const siz = Math.floor(dim / 2); // Determine the half of the board
  const initMove = move;
  const directions = ["up", "top-left", "bottom-left", "down"];
  const visited = new Set();

  // Get neighbors of the move
  let neighbours = getNeighbours(dim, move);
  neighbours = neighbours.map(([x, y]) => board[x][y]);

  // If less than 2 valid neighbors, return false
  if (neighbours.filter(Boolean).length < 2) {
    return false;
  }

  // Start exploration in 4 contiguous directions
  let exploration = [];
  for (let direction of directions) {
    let [x, y] = move;
    let half = Math.sign(move[1] - siz); // 0 for mid, -1 for left, 1 for right
    const [dx, dy] = moveCoordinates(direction, half);
    const nx = x + dx,
      ny = y + dy;

    if (isValid(nx, ny, dim) && board[nx][ny]) {
      exploration.push([[nx, ny], direction]);
      visited.add(`${nx},${ny},${direction}`);
    }
  }

  let ringLength = 1;
  // Continue exploration in forward directions
  while (exploration.length !== 0) {
    let newExp = [];

    for (const [currentMove, prevDirection] of exploration) {
      let [x, y] = currentMove;
      let half = Math.sign(y - siz);
      const newDirections = threeForwardMoves(prevDirection);

      for (let direction of newDirections) {
        const [dx, dy] = moveCoordinates(direction, half);
        const nx = x + dx,
          ny = y + dy;

        if (
          isValid(nx, ny, dim) &&
          board[nx][ny] &&
          !visited.has(`${nx},${ny},${direction}`)
        ) {
          if (initMove[0] === nx && initMove[1] === ny && ringLength >= 5) {
            return true; // Ring detected
          }
          newExp.push([[nx, ny], direction]);
          visited.add(`${nx},${ny},${direction}`);
        }
      }
    }

    exploration = newExp;
    ringLength += 1;
  }

  return false;
}

function getAllCorners(dim) {
  /*
      Returns the coordinates of the corner vertices on the hexagonal board.
  
      Parameters:
      - dim (int): The dimension of the board
  
      Returns:
      - Array: List of corner coordinates as strings in the format "x,y"
    */

  return [
    [0, 0], // Top-left corner
    [0, Math.floor(dim / 2)], // Top-middle corner
    [0, dim - 1], // Top-right corner
    [Math.floor(dim / 2), dim - 1], // Middle-right corner
    [dim - 1, Math.floor(dim / 2)], // Bottom-middle corner
    [Math.floor(dim / 2), 0], // Middle-left corner
  ].map(([x, y]) => `${x},${y}`);
}

function getEdge(vertex, dim) {
  /*
      Returns the edge number on which the vertex lies.
  
      Parameters:
      - vertex (Array): The coordinates of the vertex [i, j]
      - dim (int): The dimension of the board
  
      Returns:
      - int: Edge number (0 to 5), or -1 if the vertex is not on an edge.
    */

  let [i, j] = vertex;
  const mid = Math.floor(dim / 2);

  if (j === 0 && i > 0 && i < mid) return 0; // Left edge
  if (i === 0 && j > 0 && j < mid) return 1; // Top-left edge
  if (i === 0 && j > mid && j < dim - 1) return 2; // Top-right edge
  if (j === dim - 1 && i > 0 && i < mid) return 3; // Right edge
  if (i > mid && i < dim - 1 && i + j === 3 * mid) return 4; // Bottom-right edge
  if (i > mid && i < dim - 1 && i - j === mid) return 5; // Bottom-left edge

  return -1; // If the vertex does not lie on any edge
}

function getCorner(vertex, dim) {
  /*
      Returns the corner number on which the vertex lies.
  
      Parameters:
      - vertex (Array): The coordinates of the vertex [i, j]
      - dim (int): The dimension of the board
  
      Returns:
      - int: Corner number (0 to 5), or -1 if the vertex is not on a corner.
    */

  let [i, j] = vertex;
  const mid = Math.floor(dim / 2);

  if (i === 0 && j === 0) return 0; // Top-left corner
  if (i === 0 && j === mid) return 1; // Top middle corner
  if (i === 0 && j === dim - 1) return 2; // Top-right corner
  if (i === mid && j === dim - 1) return 3; // Right middle corner
  if (i === dim - 1 && j === mid) return 4; // Bottom middle corner
  if (i === mid && j === 0) return 5; // Left middle corner

  return -1; // If the vertex does not lie on any corner
}

function moveCoordinates(direction, half) {
  /*
      Returns the coordinates of the move in the given direction.
  
      Parameters:
      - direction (string): The direction to which the move is to be made (e.g., 'up', 'down', 'top-left', 'top-right', etc.)
      - half (int): The half of the board (0 for middle, -1 for left half, 1 for right half)
  
      Returns:
      - Array: The x and y deltas for the direction
    */

  if (direction === "up") {
    return [-1, 0];
  } else if (direction === "down") {
    return [1, 0];
  } else if (direction === "top-left") {
    return half === 0 || half < 0 ? [-1, -1] : [0, -1];
  } else if (direction === "top-right") {
    return half === 0 || half > 0 ? [-1, 1] : [0, 1];
  } else if (direction === "bottom-left") {
    return half === 0 || half < 0 ? [0, -1] : [1, -1];
  } else if (direction === "bottom-right") {
    return half === 0 || half > 0 ? [0, 1] : [1, 1];
  }

  return [0, 0]; // Default case: No movement
}

function threeForwardMoves(direction) {
  /**
   * Returns the 3 forward moves from the current direction.
   * @param {string} direction - The direction of the last move.
   * @returns {Array} - List of 3 forward moves from the current direction.
   */
  if (direction === "up") {
    return ["top-left", "up", "top-right"];
  } else if (direction === "down") {
    return ["bottom-left", "down", "bottom-right"];
  } else if (direction === "top-left") {
    return ["bottom-left", "top-left", "up"];
  } else if (direction === "top-right") {
    return ["top-right", "up", "bottom-right"];
  } else if (direction === "bottom-left") {
    return ["bottom-left", "down", "top-left"];
  } else if (direction === "bottom-right") {
    return ["bottom-right", "down", "top-right"];
  }
  return [];
}
