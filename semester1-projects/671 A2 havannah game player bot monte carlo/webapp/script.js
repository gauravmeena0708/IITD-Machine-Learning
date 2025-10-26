"use strict";

const canvas = document.getElementById("gameCanvas");
const context = canvas.getContext("2d");
const sizeSelect = document.getElementById("sizeSelect");
const player2TypeSelect = document.getElementById("player2Type");
const difficultySelect = document.getElementById("difficultySelect");
const startGameBtn = document.getElementById("startGameBtn");
const currentTurnEl = document.getElementById("currentTurn");
const winnerEl = document.getElementById("winner");
const player1TimeEl = document.getElementById("player1Time");
const player2TimeEl = document.getElementById("player2Time");
const gameTable = document.getElementById("gameTable");

Object.assign(window, {
  getValidActions: Havannah.getValidActions,
  getEdge: Havannah.getEdge,
  getCorner: Havannah.getCorner,
  getNeighbours: Havannah.getNeighbours,
  checkWin(boardState, move, playerNum) {
    const { win, way } = Havannah.checkWin(boardState, move, playerNum);
    return [win, way];
  },
});

class RandomAI {
  constructor(playerNumber) {
    this.playerNumber = playerNumber;
  }

  getMove(state) {
    const moves = Havannah.getValidActions(state);
    if (!moves.length) return null;
    return moves[Math.floor(Math.random() * moves.length)];
  }
}

let layers = parseInt(sizeSelect.value, 10) || 6;
let hexSize = 36;
let board = [];
let currentPlayer = 0; // 0 => Player 1, 1 => Player 2
let playerTime = [300, 300];
let timerInterval = null;
let gameOver = false;
let aiPlayer2 = null;
let isPlayer2AI = false;
let particles = [];

startGameBtn.addEventListener("click", startGame);
canvas.addEventListener("click", handleCanvasClick);

function startGame() {
  layers = parseInt(sizeSelect.value, 10) || layers;
  adjustCanvasSize(layers);
  board = createInitialBoard(layers);
  currentPlayer = 0;
  playerTime = [300, 300];
  gameOver = false;
  particles = [];
  winnerEl.textContent = "";
  canvas.style.border = "2px solid #000";

  updateTimerLabels();
  updateCurrentTurn();
  drawBoard();
  renderGameTable(board);

  if (timerInterval) clearInterval(timerInterval);
  timerInterval = setInterval(updateTimer, 1000);

  const selection = player2TypeSelect.value;
  if (selection === "human") {
    aiPlayer2 = null;
    isPlayer2AI = false;
  } else if (selection === "computer") {
    isPlayer2AI = true;
    const difficulty = difficultySelect.value;
    if (difficulty === "easy") {
      aiPlayer2 = new RandomAI(2);
    } else if (difficulty === "medium") {
      aiPlayer2 = new AIPlayer(2);
    } else {
      aiPlayer2 = new AIPlayer2(2);
    }
  }
}

function adjustCanvasSize(layerCount) {
  const screenHeight = window.innerHeight || 800;
  const padding = 120;
  const maxCanvasSize = Math.max(screenHeight - padding, 360);
  hexSize = Math.floor(maxCanvasSize / (3 * layerCount));
  hexSize = Math.max(hexSize, 24);
  const canvasSize = hexSize * 3 * layerCount;
  canvas.width = canvasSize;
  canvas.height = canvasSize;
}

function createInitialBoard(layerCount) {
  const size = 2 * layerCount - 1;
  const newBoard = [];
  for (let i = 0; i < size; i++) {
    const row = [];
    for (let j = 0; j < size; j++) {
      row.push(0);
    }
    newBoard.push(row);
  }

  for (let i = layerCount; i < size; i++) {
    for (let j = 0; j < i - layerCount + 1; j++) {
      newBoard[i][j] = 3;
      newBoard[i][2 * layerCount - 2 - j] = 3;
    }
  }
  return newBoard;
}

function drawBoard() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  const limit = 2 * layers - 1;
  for (let j = 0; j < limit; j++) {
    const colSize = j < layers ? layers + j : layers + (limit - 1 - j);
    for (let i = 0; i < colSize; i++) {
      const hexCoords = calculateHexagon(i, j);
      drawHexagon(hexCoords, board[i][j], i, j);
    }
  }
  drawParticles();
}

function calculateHexagon(i, j) {
  const sqrt3 = Math.sqrt(3);
  const offsetX = (j * hexSize * 3) / 2;
  const offsetY =
    ((Math.abs(j - layers + 1) + 2 * i) * hexSize * sqrt3) / 2;
  return [
    { x: hexSize / 2 + offsetX, y: offsetY },
    { x: (hexSize * 3) / 2 + offsetX, y: offsetY },
    { x: hexSize * 2 + offsetX, y: (hexSize * sqrt3) / 2 + offsetY },
    { x: (hexSize * 3) / 2 + offsetX, y: hexSize * sqrt3 + offsetY },
    { x: hexSize / 2 + offsetX, y: hexSize * sqrt3 + offsetY },
    { x: offsetX, y: (hexSize * sqrt3) / 2 + offsetY },
  ];
}

function drawHexagon(coords, value, row, col) {
  context.beginPath();
  context.moveTo(coords[0].x, coords[0].y);
  for (let k = 1; k < coords.length; k++) {
    context.lineTo(coords[k].x, coords[k].y);
  }
  context.closePath();

  let fill = "#ffffff";
  if (value === 1) fill = "#facc15";
  else if (value === 2) fill = "#ef4444";
  else if (value === 3) fill = "#334155";

  context.fillStyle = fill;
  context.strokeStyle = "#111827";
  context.lineWidth = 1;
  context.fill();
  context.stroke();

  const centroid = calculateCentroid(coords);
  context.fillStyle = "#111827";
  context.font = `${Math.max(hexSize * 0.35, 10)}px Arial`;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(`(${row},${col})`, centroid.x, centroid.y);
}

function calculateCentroid(coords) {
  let x = 0;
  let y = 0;
  for (const point of coords) {
    x += point.x;
    y += point.y;
  }
  return { x: x / coords.length, y: y / coords.length };
}

function handleCanvasClick(event) {
  if (gameOver) return;
  if (currentPlayer === 1 && isPlayer2AI) return;

  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const [row, col] = getHexagonAtCoords(x, y);
  if (row === null || col === null) return;
  if (board[row][col] !== 0) return;

  makeMove(row, col);
}

function getHexagonAtCoords(x, y) {
  const limit = 2 * layers - 1;
  for (let j = 0; j < limit; j++) {
    const colSize = j < layers ? layers + j : layers + (limit - 1 - j);
    for (let i = 0; i < colSize; i++) {
      const hexCoords = calculateHexagon(i, j);
      if (isPointInHexagon(x, y, hexCoords)) {
        return [i, j];
      }
    }
  }
  return [null, null];
}

function isPointInHexagon(x, y, coords) {
  let inside = false;
  for (let i = 0, j = coords.length - 1; i < coords.length; j = i++) {
    const xi = coords[i].x;
    const yi = coords[i].y;
    const xj = coords[j].x;
    const yj = coords[j].y;
    const intersect =
      yi > y !== yj > y &&
      x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function makeMove(row, col) {
  board[row][col] = currentPlayer + 1;
  drawBoard();
  renderGameTable(board);

  const { win, way } = Havannah.checkWin(board, [row, col], currentPlayer + 1);
  if (win) {
    endGame(currentPlayer + 1, way);
    return;
  }

  if (Havannah.getValidActions(board).length === 0) {
    endGame(null, "tie");
    return;
  }

  currentPlayer = currentPlayer === 0 ? 1 : 0;
  updateCurrentTurn();
  if (!gameOver && currentPlayer === 1 && isPlayer2AI) {
    handleAITurn();
  }
}

function handleAITurn() {
  setTimeout(() => {
    if (!aiPlayer2 || gameOver) return;
    const move = aiPlayer2.getMove(board);
    if (!move) {
      endGame(null, "tie");
      return;
    }
    makeMove(move[0], move[1]);
  }, 400);
}

function updateTimer() {
  if (gameOver) return;
  playerTime[currentPlayer] -= 1;
  if (playerTime[currentPlayer] <= 0) {
    playerTime[currentPlayer] = 0;
    endGame(currentPlayer === 0 ? 2 : 1, "timeout");
    return;
  }
  updateTimerLabels();
}

function updateTimerLabels() {
  const [p1, p2] = playerTime.map((time) => {
    const minutes = Math.floor(time / 60);
    const seconds = time % 60;
    return `${minutes}:${seconds < 10 ? "0" : ""}${seconds}`;
  });
  player1TimeEl.textContent = p1;
  player2TimeEl.textContent = p2;
}

function updateCurrentTurn() {
  if (gameOver) return;
  currentTurnEl.textContent = `Current Turn: Player ${currentPlayer + 1}`;
}

function endGame(winner, structure) {
  gameOver = true;
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }

  let message;
  if (winner === 1 || winner === 2) {
    const color = winner === 1 ? "gold" : "#ef4444";
    canvas.style.border = `5px solid ${color}`;
    message = `Player ${winner} wins${structure ? ` by ${structure}` : ""}!`;
  } else {
    canvas.style.border = "5px solid #334155";
    message = "It's a tie!";
  }

  currentTurnEl.textContent = message;
  winnerEl.textContent = message;
  triggerCelebration();
  alert(message);
}

function renderGameTable(currentBoard) {
  gameTable.innerHTML = "";
  currentBoard.forEach((row) => {
    const tr = document.createElement("tr");
    row.forEach((cell) => {
      const td = document.createElement("td");
      if (cell === 1) {
        td.textContent = "Y";
        td.style.backgroundColor = "#fde68a";
      } else if (cell === 2) {
        td.textContent = "R";
        td.style.backgroundColor = "#fca5a5";
      } else if (cell === 3) {
        td.textContent = "X";
        td.style.backgroundColor = "#cbd5f5";
      }
      tr.appendChild(td);
    });
    gameTable.appendChild(tr);
  });
}

function triggerCelebration() {
  particles = [];
  for (let i = 0; i < 120; i++) {
    particles.push(createParticle());
  }
  animateParticles();
}

function createParticle() {
  return {
    x: canvas.width / 2,
    y: canvas.height / 2,
    radius: Math.random() * 3 + 2,
    color: `hsl(${Math.random() * 360}, 100%, 60%)`,
    speedX: Math.random() * 4 - 2,
    speedY: Math.random() * 4 - 2,
    life: Math.floor(Math.random() * 60 + 30),
  };
}

function drawParticles() {
  for (let i = particles.length - 1; i >= 0; i--) {
    const particle = particles[i];
    context.beginPath();
    context.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
    context.fillStyle = particle.color;
    context.fill();
    particle.x += particle.speedX;
    particle.y += particle.speedY;
    particle.life -= 1;
    if (particle.life <= 0) {
      particles.splice(i, 1);
    }
  }
}

function animateParticles() {
  if (particles.length === 0) return;
  drawBoard();
  requestAnimationFrame(animateParticles);
}

startGame();
