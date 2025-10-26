let canvas = document.getElementById("gameCanvas");
let context = canvas.getContext("2d");
let layers;
let currentPlayer = 0; // 0 -> Player 1, 1 -> Player 2
let playerTime = [300, 300]; // 5 minutes for each player in seconds
let gameOver = false;
let board;
let intervalId;
let hexSize;
let aiPlayer2 = null; // AI for Player 2
let isPlayer2AI = false; // To check if Player 2 is AI

document.getElementById("startGameBtn").addEventListener("click", startGame);

function startGame() {
  layers = parseInt(document.getElementById("sizeSelect").value);
  adjustCanvasSize(layers); // Adjust canvas size dynamically based on board size
  board = createInitialBoard(layers);
  currentPlayer = 0;
  gameOver = false;
  playerTime = [300, 300];
  document.getElementById("winner").textContent = "";
  document.getElementById("currentTurn").textContent = "Current Turn: Player 1";
  drawBoard(board);
  if (intervalId) clearInterval(intervalId);
  intervalId = setInterval(updateTimer, 1000); // Update timer every second

  // Determine if Player 2 is AI or human/AI2
  let player2Type = document.getElementById("player2Type").value;
  if (player2Type === "human") {
    aiPlayer2 = null; // Player 2 is human
    isPlayer2AI = false;
  } else if (player2Type === "ai") {
    aiPlayer2 = new AIPlayer(2); // Initialize AI for Player 2
    isPlayer2AI = true;
  } else if (player2Type === "ai2") {
    aiPlayer2 = new AIPlayer2(2); // Initialize AI2 for Player 2
    isPlayer2AI = true;
  }
}

let particles = []; // Array to store sparkles for celebration effect

function adjustCanvasSize(layers) {
  const screenHeight = window.innerHeight; // Get the screen height
  const maxCanvasSize = screenHeight - 100; // Max canvas size should be 200px less than screen height
  const canvasPadding = 10; // Extra padding around the canvas
  hexSize = Math.floor(maxCanvasSize / (3 * layers)); // Dynamically calculate hex size based on layers and max canvas size
  const canvasSize = hexSize * 3 * layers + canvasPadding; // Adjust canvas size dynamically

  canvas.width = canvasSize;
  canvas.height = canvasSize;
}

function createInitialBoard(layers, numBlockedTiles) {
  const board = [];

  // Initialize the board with empty cells (0 represents empty)
  for (let i = 0; i < 2 * layers - 1; i++) {
    const row = [];
    for (let j = 0; j < 2 * layers - 1; j++) {
      row.push(0); // Initialize all tiles to 0 (empty)
    }
    board.push(row);
  }

  // Logic to block tiles on the lower triangular side, similar to the Python code
  for (let i = layers; i < 2 * layers - 1; i++) {
    for (let j = 0; j < i - layers + 1; j++) {
      // Block tiles in the lower triangular region
      board[i][j] = 3;
      board[i][2 * layers - 2 - j] = 3;
    }
  }

  // Randomly block additional tiles if specified
  if (numBlockedTiles > 0) {
    for (let i = 0; i < numBlockedTiles; i++) {
      let row, col;
      do {
        row = Math.floor(Math.random() * (2 * layers - 1));
        col = Math.floor(Math.random() * (2 * layers - 1));
      } while (board[row][col] !== 0); // Ensure that we don't overwrite an already blocked tile
      board[row][col] = 3; // Block the randomly selected tile
    }
  }
  return board;
}

function drawBoard(board) {
  context.clearRect(0, 0, canvas.width, canvas.height);
  for (let j = 0; j < 2 * layers - 1; j++) {
    const colSize = j < layers ? layers + j : layers + 2 * layers - 2 - j;
    for (let i = 0; i < colSize; i++) {
      const hexCoords = calculateHexagon(i, j);
      const color =
        board[i][j] === 1 ? "yellow" : board[i][j] === 2 ? "red" : "white";
      drawHexagon(hexCoords, color, i, j);
    }
  }
  // Draw particles (sparkles)
  drawParticles();
}

function calculateHexagon(i, j) {
  const sqrt3 = Math.sqrt(3);
  const offsetX = (j * hexSize * 3) / 2;
  const offsetY = ((Math.abs(j - layers + 1) + 2 * i) * hexSize * sqrt3) / 2;
  return [
    { x: hexSize / 2 + offsetX, y: offsetY },
    { x: (hexSize * 3) / 2 + offsetX, y: offsetY },
    { x: hexSize * 2 + offsetX, y: (hexSize * sqrt3) / 2 + offsetY },
    { x: (hexSize * 3) / 2 + offsetX, y: hexSize * sqrt3 + offsetY },
    { x: hexSize / 2 + offsetX, y: hexSize * sqrt3 + offsetY },
    { x: offsetX, y: (hexSize * sqrt3) / 2 + offsetY },
  ];
}

function drawHexagon(coords, fillColor, row, col) {
  context.beginPath();
  context.moveTo(coords[0].x, coords[0].y);
  coords.forEach((point) => context.lineTo(point.x, point.y));
  context.closePath();
  context.fillStyle = fillColor || "white";
  context.fill();
  context.stroke();

  const centroid = calculateCentroid(coords);
  context.fillStyle = "black";
  context.font = "12px Arial";
  context.fillText(`(${row},${col})`, centroid.x - 10, centroid.y + 4);
}

function calculateCentroid(coords) {
  let centroidX = 0,
    centroidY = 0;
  coords.forEach((point) => {
    centroidX += point.x;
    centroidY += point.y;
  });
  return { x: centroidX / coords.length, y: centroidY / coords.length };
}

function updateTimer() {
  if (gameOver) return;
  playerTime[currentPlayer]--;
  if (playerTime[currentPlayer] <= 0) {
    gameOver = true;
    document.getElementById("currentTurn").textContent = `Game Over! Player ${
      currentPlayer + 1
    } ran out of time.`;
    clearInterval(intervalId);
    return;
  }
  updatePlayerTime();
}

function updatePlayerTime() {
  const player1Min = Math.floor(playerTime[0] / 60);
  const player1Sec = playerTime[0] % 60;
  const player2Min = Math.floor(playerTime[1] / 60);
  const player2Sec = playerTime[1] % 60;
  document.getElementById("player1Time").textContent = `${player1Min}:${
    player1Sec < 10 ? "0" : ""
  }${player1Sec}`;
  document.getElementById("player2Time").textContent = `${player2Min}:${
    player2Sec < 10 ? "0" : ""
  }${player2Sec}`;
}

canvas.addEventListener("click", (event) => {
  if (gameOver || (currentPlayer === 1 && isPlayer2AI)) return; // Skip human clicks if it's AI's turn

  const x = event.offsetX;
  const y = event.offsetY;

  const [row, col] = getHexagonAtCoords(x, y);
  if (row !== null && col !== null && board[row][col] === 0) {
    makeMove(row, col);
  }
});

function getHexagonAtCoords(x, y) {
  for (let r = 0; r < 2 * layers - 1; r++) {
    for (let c = 0; c < 2 * layers - 1; c++) {
      const hexCoords = calculateHexagon(r, c);
      if (isPointInHexagon(x, y, hexCoords)) {
        return [r, c];
      }
    }
  }
  return [null, null];
}

function isPointInHexagon(x, y, coords) {
  let inside = false;
  for (let i = 0; i < coords.length; i++) {
    const j = (i + 1) % coords.length;
    const xi = coords[i].x,
      yi = coords[i].y;
    const xj = coords[j].x,
      yj = coords[j].y;

    const intersect =
      yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function switchPlayer() {
  currentPlayer = 1 - currentPlayer;
  document.getElementById("currentTurn").textContent = `Current Turn: Player ${
    currentPlayer + 1
  }`;
  if (currentPlayer === 1 && isPlayer2AI && !gameOver) {
    handleAITurn(); // AI makes a move
  }
}

function makeMove(row, col) {
  board[row][col] = currentPlayer + 1; // Player 1 is Yellow, Player 2 is Red
  drawBoard(board);

  const [win, structure] = checkWin(board, [row, col], currentPlayer + 1);
  if (win) {
    endGame(currentPlayer + 1, structure);
  } else {
    switchPlayer();
  }
}

function handleAITurn() {
  setTimeout(() => {
    if (aiPlayer2) {
      const aiMove = aiPlayer2.getMove(board); // Use the selected AI for Player 2
      makeMove(aiMove[0], aiMove[1]);
    }
  }, 500); // Slight delay to mimic AI thinking
}

function endGame(winner, structure) {
  gameOver = true;
  const winnerColor = winner === 1 ? "yellow" : "red";
  canvas.style.border = `5px solid ${winnerColor}`; // Change canvas border to winning player's color
  document.getElementById(
    "currentTurn"
  ).textContent = `Game Over! Player ${winner} wins by ${structure}!`;

  // Trigger the celebration (sparkle effect)
  triggerCelebration();
  alert(`Player ${winner} won!!`); // Alert the winner
}

// Sparkle/celebration effect
function triggerCelebration() {
  for (let i = 0; i < 100; i++) {
    // Create 100 particles
    particles.push(createParticle());
  }
  animateParticles();
}

function createParticle() {
  return {
    x: canvas.width / 2,
    y: canvas.height / 2,
    radius: Math.random() * 4 + 2, // Random radius
    color: `hsl(${Math.random() * 360}, 100%, 50%)`, // Random color
    speedX: Math.random() * 5 - 2.5, // Random speed X
    speedY: Math.random() * 5 - 2.5, // Random speed Y
    life: 100, // Lifespan of the particle
  };
}

function drawParticles() {
  particles.forEach((particle, index) => {
    context.beginPath();
    context.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
    context.fillStyle = particle.color;
    context.fill();
    context.closePath();

    // Update particle position and reduce life
    particle.x += particle.speedX;
    particle.y += particle.speedY;
    particle.life--;

    // Remove particle if life is over
    if (particle.life <= 0) {
      particles.splice(index, 1);
    }
  });
}

function animateParticles() {
  if (particles.length > 0) {
    drawBoard(board); // Redraw the board
    requestAnimationFrame(animateParticles);
  }
}

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

function getAllEdges(dim) {
  const siz = Math.floor((dim + 1) / 2);
  const sides = [
    Array.from({ length: siz }, (_, i) => `${0},${i}`), // Top edge
    Array.from({ length: siz }, (_, i) => `${i},${dim - 1}`), // Right edge
    Array.from({ length: siz }, (_, i) => `${dim - 1},${siz + i - 1}`), // Bottom edge
    Array.from({ length: siz }, (_, i) => `${siz - 1 + i},${0}`), // Left edge
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

function checkFork(board, move) {
  const dim = board.length;
  const visited = bfsReachable(board, move);
  const edges = getAllEdges(dim);
  const move_i = move[0],
    move_j = move[1];

  // Manually check if the move itself is on any edge
  let edgesTouchedByMove = 0;
  for (const edge of edges) {
    if (edge.has(`${move_i},${move_j}`)) {
      edgesTouchedByMove++;
    }
  }

  // Count reachable edges, excluding those already counted by the move itself
  let reachableEdges = 0;
  for (const edge of edges) {
    if (edge.has(`${move_i},${move_j}`)) continue; // Skip if the edge was already counted
    if (sideHasReachablePoints(edge, visited)) {
      reachableEdges++;
    }
  }

  // A fork is detected if the move touches at least 3 edges 
  // or if the move touches fewer than 3 edges but the total (including reachable edges) is 3 or more
  return edgesTouchedByMove >= 3 || edgesTouchedByMove + reachableEdges >= 3;
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

// Add this function to render the game board in the table
function renderGameTable(board) {
  const gameTable = document.getElementById("gameTable");
  gameTable.innerHTML = ""; // Clear previous table content

  board.forEach((row, rowIndex) => {
    const tr = document.createElement("tr");
    row.forEach((cell, colIndex) => {
      const td = document.createElement("td");
      if (cell === 1) {
        td.textContent = "Y"; // Player 1 (Yellow)
        td.style.backgroundColor = "yellow";
      } else if (cell === 2) {
        td.textContent = "R"; // Player 2 (Red)
        td.style.backgroundColor = "red";
      } else if (cell === 3) {
        td.textContent = "B"; // Blocked tiles
        td.style.backgroundColor = "gray";
      } else {
        td.textContent = ""; // Empty cell
      }
      tr.appendChild(td);
    });
    gameTable.appendChild(tr);
  });
}

// Update the startGame function to render the initial board
function startGame() {
  layers = parseInt(document.getElementById("sizeSelect").value);
  adjustCanvasSize(layers); // Adjust canvas size dynamically based on board size
  board = createInitialBoard(layers);
  currentPlayer = 0;
  gameOver = false;
  playerTime = [300, 300];
  document.getElementById("winner").textContent = "";
  document.getElementById("currentTurn").textContent = "Current Turn: Player 1";
  drawBoard(board);
  renderGameTable(board); // Render the initial board in the table
  if (intervalId) clearInterval(intervalId);
  intervalId = setInterval(updateTimer, 1000); // Update timer every second

  // Determine if Player 2 is AI or human/AI2
  let player2Type = document.getElementById("player2Type").value;
  if (player2Type === "human") {
    aiPlayer2 = null; // Player 2 is human
    isPlayer2AI = false;
  } else if (player2Type === "ai") {
    aiPlayer2 = new AIPlayer(2); // Initialize AI for Player 2
    isPlayer2AI = true;
  } else if (player2Type === "ai2") {
    aiPlayer2 = new AIPlayer2(2); // Initialize AI2 for Player 2
    isPlayer2AI = true;
  }
}

// Update the makeMove function to also render the table after each move
function makeMove(row, col) {
  board[row][col] = currentPlayer + 1; // Player 1 is Yellow, Player 2 is Red
  drawBoard(board);
  renderGameTable(board); // Update the table after each move

  const [win, structure] = checkWin(board, [row, col], currentPlayer + 1);
  if (win) {
    endGame(currentPlayer + 1, structure);
  } else {
    switchPlayer();
  }
}

// Add this function to check if there are any valid moves left for any player
function checkForTie(board) {
  const validMoves = getValidActions(board);
  return validMoves.length === 0; // Return true if no valid moves are left
}

// Update the makeMove function to check for a tie after each move
function makeMove(row, col) {
  board[row][col] = currentPlayer + 1; // Player 1 is Yellow, Player 2 is Red
  drawBoard(board);
  renderGameTable(board); // Update the table after each move

  // Check if the current move results in a win
  const [win, structure] = checkWin(board, [row, col], currentPlayer + 1);
  if (win) {
    endGame(currentPlayer + 1, structure);
  } else {
    // Check if there are no valid moves left for either player
    if (checkForTie(board)) {
      endGame(null, "tie");
    } else {
      switchPlayer();
    }
  }
}

// Update the endGame function to handle tie cases
function endGame(winner, structure) {
  gameOver = true;
  clearInterval(intervalId); // Stop the timer

  if (winner) {
    const winnerColor = winner === 1 ? "yellow" : "red";
    canvas.style.border = `5px solid ${winnerColor}`; // Change canvas border to winning player's color
    document.getElementById(
      "currentTurn"
    ).textContent = `Game Over! Player ${winner} wins by ${structure}!`;
    gtag("send", "event", "Game", `Win player ${winner}`);
    alert(`Player ${winner} won!!`); // Alert the winner
    if (typeof gtag === "function") {
      gtag("send", "event", "Win", {
        event_category: "Game",
        event_label: `Level 1 ${winner}`,
        value: `Player ${winner} won`,
      });
      gtag("event", "Win", {
        event_category: "Game",
        event_label: `Level 1 ${winner}`,
        value: `Player ${winner} won`,
      });

      gtag("event", "Game Outcome", {
        Game_Name: "Havannah",
        Winner: winner,
        "Won BY": structure,
      });
    } else {
      console.warn("Google Analytics (gtag) is not defined.");
    }
  } else if (structure === "tie") {
    document.getElementById(
      "currentTurn"
    ).textContent = `Game Over! It's a tie!`;
    alert("It's a tie! No valid moves left.");
  }

  // Trigger the celebration (sparkle effect) if there's a winner
  if (winner) {
    triggerCelebration();
  }
}
