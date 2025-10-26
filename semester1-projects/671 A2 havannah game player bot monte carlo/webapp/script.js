"use strict";

const canvas = document.getElementById("gameCanvas");
const context = canvas.getContext("2d");
const sizeSelect = document.getElementById("sizeSelect");
const modeSelect = document.getElementById("modeSelect");
const tutorialControls = document.getElementById("tutorialControls");
const tutorialSelect = document.getElementById("tutorialSelect");
const tutorialObjectiveEl = document.getElementById("tutorialObjective");
const player2TypeSelect = document.getElementById("player2Type");
const difficultySelect = document.getElementById("difficultySelect");
const startGameBtn = document.getElementById("startGameBtn");
const hintBtn = document.getElementById("hintBtn");
const clearHintBtn = document.getElementById("clearHintBtn");
const helperMessageEl = document.getElementById("helperMessage");
const currentTurnEl = document.getElementById("currentTurn");
const winnerEl = document.getElementById("winner");
const player1TimeEl = document.getElementById("player1Time");
const player2TimeEl = document.getElementById("player2Time");
const analysisToggle = document.getElementById("analysisToggle");
const analysisSummary = document.getElementById("analysisSummary");
const gameTable = document.getElementById("gameTable");
const moveHistoryList = document.getElementById("moveHistory");
const opponentControls = document.getElementById("opponentControls");
const difficultyControls = document.getElementById("difficultyControls");
const statsRoundsEl = document.getElementById("statsRounds");
const statsP1WinsEl = document.getElementById("statsP1Wins");
const statsP2WinsEl = document.getElementById("statsP2Wins");
const statsTiesEl = document.getElementById("statsTies");
const statsFastestEl = document.getElementById("statsFastest");
const resetProgressBtn = document.getElementById("resetProgressBtn");
const achievementBadges = {
  bridge: document.querySelector('[data-achievement="bridge"]'),
  fork: document.querySelector('[data-achievement="fork"]'),
  ring: document.querySelector('[data-achievement="ring"]'),
  streak: document.querySelector('[data-achievement="streak"]'),
};

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
let hintMove = null;
let lastMove = null;
let moveHistory = [];
let turnCounter = 0;
let consecutiveWins = 0;
let currentMode = modeSelect.value || "standard";
let tutorialScenario = null;
let analysisEnabled = false;
let analysisData = null;

const stats = {
  rounds: 0,
  p1Wins: 0,
  p2Wins: 0,
  ties: 0,
  fastest: null,
};

const achievements = {
  bridge: false,
  fork: false,
  ring: false,
  streak: false,
};

const STORAGE_KEY = "havannah_webapp2_progress_v1";

const TUTORIAL_SCENARIOS = {
  "bridge-basics": {
    id: "bridge-basics",
    name: "Bridge Basics",
    layers: 4,
    objective:
      "Complete the bridge by linking the two highlighted corners on the left spine. You only need one more stone!",
    helperText:
      "Corners count as part of the bridge. Look for a single move that creates an unbroken path between them.",
    placements: {
      1: [
        [0, 0],
        [1, 0],
        [3, 0],
      ],
      2: [
        [1, 2],
        [2, 3],
      ],
    },
    hint: [2, 0],
    startingPlayer: 1,
  },
  "fork-basics": {
    id: "fork-basics",
    name: "Fork Foundations",
    layers: 4,
    objective:
      "Extend your group to reach a third edge. A fork is any connection to three distinct edges.",
    helperText:
      "You already touch the northwest and west edges. Use the top row to branch out toward a new edge.",
    placements: {
      1: [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 0],
        [1, 1],
      ],
      2: [
        [2, 3],
        [2, 4],
      ],
    },
    hint: [0, 4],
    startingPlayer: 1,
  },
  "ring-setup": {
    id: "ring-setup",
    name: "Ring Awareness",
    layers: 4,
    objective:
      "Close the loop to trap the red stone. Rings enclose at least one cell (even if it belongs to the opponent).",
    helperText:
      "Visualise a loop of your stones around the red stone. Only one gap remains open.",
    placements: {
      1: [
        [1, 2],
        [1, 4],
        [2, 2],
        [2, 5],
        [3, 3],
        [3, 4],
      ],
      2: [
        [2, 3],
      ],
    },
    hint: [1, 3],
    startingPlayer: 1,
  },
};

const EDGE_NAMES = [
  "West",
  "North-West",
  "North-East",
  "East",
  "South-East",
  "South-West",
];

const CORNER_NAMES = [
  "Top-left",
  "Top",
  "Top-right",
  "Right",
  "Bottom",
  "Left",
];

startGameBtn.addEventListener("click", startGame);
canvas.addEventListener("click", handleCanvasClick);
hintBtn.addEventListener("click", offerHint);
clearHintBtn.addEventListener("click", () => clearHint());
player2TypeSelect.addEventListener("change", () => {
  syncDifficultyAvailability();
  setHelperMessage("");
});
resetProgressBtn.addEventListener("click", resetProgress);
modeSelect.addEventListener("change", () => {
  updateModeUI();
  startGame();
});
tutorialSelect.addEventListener("change", () => {
  if (modeSelect.value === "tutorial") {
    startGame();
  }
});
analysisToggle.addEventListener("change", () => {
  analysisEnabled = analysisToggle.checked;
  resetAnalysis();
  if (analysisEnabled) {
    setHelperMessage(
      currentMode === "tutorial" && tutorialScenario
        ? (tutorialScenario.helperText || tutorialScenario.objective || "") +
            " — analysis overlay active."
        : "Analysis mode on. Click any stone to inspect its connected group."
    );
  } else if (currentMode === "tutorial" && tutorialScenario) {
    setHelperMessage(
      tutorialScenario.helperText || tutorialScenario.objective || ""
    );
  } else {
    setHelperMessage("");
  }
  drawBoard();
  renderGameTable(board);
});

loadPersistentProgress();
renderStats();
renderAchievements();
updateModeUI();

function startGame() {
  const requestedLayers = parseInt(sizeSelect.value, 10) || layers;
  currentMode = modeSelect.value;
  const isTutorial = currentMode === "tutorial";
  tutorialScenario = isTutorial
    ? TUTORIAL_SCENARIOS[tutorialSelect.value] ||
      TUTORIAL_SCENARIOS["bridge-basics"]
    : null;
  if (isTutorial && tutorialScenario) {
    layers = tutorialScenario.layers;
  } else {
    layers = requestedLayers;
  }
  adjustCanvasSize(layers);
  board = createInitialBoard(layers);
  currentPlayer = 0;
  playerTime = isTutorial ? [Infinity, Infinity] : [300, 300];
  gameOver = false;
  particles = [];
  hintMove = null;
  lastMove = null;
  moveHistory = [];
  turnCounter = 0;
  resetAnalysis();
  winnerEl.textContent = "";
  canvas.style.border = "2px solid #000";
  setHelperMessage("");

  renderMoveHistory();
  renderStats();
  renderAchievements();
  updateTimerLabels();
  updateCurrentTurn();
  drawBoard();
  renderGameTable(board);

  if (timerInterval) clearInterval(timerInterval);
  timerInterval = isTutorial ? null : setInterval(updateTimer, 1000);

  syncDifficultyAvailability();
  if (isTutorial) {
    isPlayer2AI = false;
    aiPlayer2 = null;
    currentPlayer = (tutorialScenario.startingPlayer || 1) - 1;
    applyTutorialScenario(tutorialScenario);
  } else {
    const opponent = player2TypeSelect.value;
    isPlayer2AI = opponent === "computer";
    aiPlayer2 = isPlayer2AI
      ? createAIForDifficulty(2, difficultySelect.value)
      : null;
    setHelperMessage("");
    tutorialObjectiveEl.textContent = "";
  }

  updateControls();
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
  const isHint =
    hintMove && hintMove[0] === row && hintMove[1] === col;
  const isLast =
    lastMove && lastMove[0] === row && lastMove[1] === col;
  const key = `${row},${col}`;

  if (isLast) {
    context.save();
    context.lineWidth = 4;
    context.strokeStyle = "rgba(34,197,94,0.85)";
    context.stroke();
    context.restore();
  }

  if (isHint) {
    context.save();
    context.lineWidth = 4;
    context.strokeStyle = "rgba(59,130,246,0.9)";
    context.stroke();
    context.restore();
  }

  if (
    analysisData &&
    analysisData.visited &&
    analysisData.visited.has(key)
  ) {
    context.save();
    context.lineWidth = analysisData.originKey === key ? 4 : 3;
    context.strokeStyle =
      analysisData.originKey === key
        ? "rgba(99,102,241,0.85)"
        : "rgba(59,130,246,0.55)";
    context.stroke();
    context.restore();
  }

  if (
    analysisData &&
    analysisData.cornerKeys &&
    analysisData.cornerKeys.has(key)
  ) {
    context.save();
    context.beginPath();
    context.arc(
      centroid.x,
      centroid.y,
      Math.max(6, hexSize * 0.18),
      0,
      Math.PI * 2
    );
    context.lineWidth = 2.5;
    context.strokeStyle = "rgba(14,116,144,0.9)";
    context.stroke();
    context.restore();
  }

  context.fillStyle = value === 3 ? "#f8fafc" : "#111827";
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
  const cellValue = board[row][col];

  if (
    analysisEnabled &&
    cellValue !== 0 &&
    cellValue !== 3
  ) {
    computeAnalysis(row, col);
    return;
  }

  if (cellValue !== 0) return;

  resetAnalysis();

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
  clearHint({ silent: true, redraw: false });
  board[row][col] = currentPlayer + 1;
  lastMove = [row, col];
  turnCounter += 1;

  const moveRecord = { player: currentPlayer + 1, row, col };
  moveHistory.push(moveRecord);

  drawBoard();
  renderGameTable(board);

  const { win, way } = Havannah.checkWin(board, [row, col], currentPlayer + 1);
  if (win) {
    moveRecord.result = way || "structure";
    renderMoveHistory();
    endGame(currentPlayer + 1, way);
    return;
  }

  if (Havannah.getValidActions(board).length === 0) {
    renderMoveHistory();
    endGame(null, "tie");
    return;
  }

  if (currentMode === "tutorial") {
    board[row][col] = 0;
    moveHistory.pop();
    turnCounter = Math.max(0, turnCounter - 1);
    drawBoard();
    renderGameTable(board);
    renderMoveHistory();
    updateCurrentTurn();
    setHelperMessage(
      "Not quite! Check the lesson objective and try again from the original position."
    );
    return;
  }

  renderMoveHistory();
  currentPlayer = currentPlayer === 0 ? 1 : 0;
  updateCurrentTurn();

  if (!gameOver && currentPlayer === 1 && isPlayer2AI) {
    handleAITurn();
  }
}

function handleAITurn() {
  updateControls();
  setTimeout(() => {
    if (!aiPlayer2 || gameOver) return;
    const boardClone = board.map((row) => row.slice());
    const move = aiPlayer2.getMove(boardClone);
    if (!move) {
      endGame(null, "tie");
      return;
    }
    makeMove(move[0], move[1]);
  }, 400);
}

function updateTimer() {
  if (currentMode === "tutorial") return;
  if (gameOver) return;
  playerTime[currentPlayer] -= 1;
  if (playerTime[currentPlayer] <= 0) {
    playerTime[currentPlayer] = 0;
    const winner = currentPlayer === 0 ? 2 : 1;
    endGame(winner, "timeout");
    return;
  }
  updateTimerLabels();
}

function updateTimerLabels() {
  const [p1, p2] = playerTime.map((time) => {
    if (!Number.isFinite(time)) {
      return "∞";
    }
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
  updateControls();
}

function endGame(winner, structure) {
  gameOver = true;
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }

  recordOutcome({ winner, structure });

  let message;
  if (winner === 1 || winner === 2) {
    const color = winner === 1 ? "gold" : "#ef4444";
    canvas.style.border = `5px solid ${color}`;
    message = `Player ${winner} wins${structure && structure !== "timeout" ? ` by ${structure}` : ""}!`;
    const lastEntry = moveHistory[moveHistory.length - 1];
    if (lastEntry && !lastEntry.result) {
      lastEntry.result = structure === "timeout" ? "timeout" : "win";
    }
  } else {
    canvas.style.border = "5px solid #334155";
    message = structure === "tie" ? "It's a tie!" : "Round ended.";
  }

  if (structure === "timeout") {
    moveHistory.push({
      summary: `Player ${winner === 1 ? 2 : 1} ran out of time.`,
    });
  } else if (!winner && structure === "tie") {
    moveHistory.push({
      summary: "Board locked up – declared a tie.",
    });
  }

  renderMoveHistory();

  currentTurnEl.textContent = message;
  winnerEl.textContent = message;
  renderMoveHistory();
  updateControls();
  triggerCelebration();
  alert(message);
}

function renderGameTable(currentBoard) {
  gameTable.innerHTML = "";
  currentBoard.forEach((row, rowIdx) => {
    const tr = document.createElement("tr");
    row.forEach((cell, colIdx) => {
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
      } else {
        td.textContent = "";
        td.style.backgroundColor = "";
      }
      if (lastMove && lastMove[0] === rowIdx && lastMove[1] === colIdx) {
        td.style.outline = "3px solid rgba(34,197,94,0.85)";
      } else if (analysisData && analysisData.visited && analysisData.visited.has(`${rowIdx},${colIdx}`)) {
        td.style.outline =
          analysisData.originKey === `${rowIdx},${colIdx}`
            ? "3px solid rgba(99,102,241,0.8)"
            : "2px solid rgba(59,130,246,0.5)";
      }
      if (
        analysisData &&
        analysisData.cornerKeys &&
        analysisData.cornerKeys.has(`${rowIdx},${colIdx}`)
      ) {
        td.style.boxShadow = "inset 0 0 0 3px rgba(14,116,144,0.6)";
      } else {
        td.style.boxShadow = "";
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

function offerHint() {
  if (gameOver) {
    setHelperMessage("Game over. Start a new round for more hints.");
    return;
  }
  const isTutorial = currentMode === "tutorial";
  if (!isTutorial && currentPlayer !== 0) {
    setHelperMessage("Wait for your turn before asking for a hint.");
    return;
  }

  let suggestion = null;
  if (isTutorial && tutorialScenario) {
    if (tutorialScenario.hint) {
      suggestion = tutorialScenario.hint.slice();
    } else {
      const advisor = createAIForDifficulty(1, "medium");
      const boardClone = board.map((row) => row.slice());
      suggestion = advisor.getMove(boardClone);
    }
  } else {
    const advisor = createAIForDifficulty(
      1,
      difficultySelect.value === "easy" ? "medium" : difficultySelect.value
    );
    const boardClone = board.map((row) => row.slice());
    suggestion = advisor.getMove(boardClone);
  }

  if (!suggestion) {
    setHelperMessage("No hint available – board might be full.");
    return;
  }

  hintMove = suggestion;
  drawBoard();
  const message = isTutorial
    ? `Lesson hint: place at (${suggestion[0]}, ${suggestion[1]}).`
    : `Hint: try placing at (${suggestion[0]}, ${suggestion[1]}).`;
  setHelperMessage(message);
  updateControls();
}

function createAIForDifficulty(playerNumber, difficulty) {
  if (difficulty === "easy") {
    return new RandomAI(playerNumber);
  }
  if (difficulty === "hard") {
    return new AIPlayer2(playerNumber, 260);
  }
  return new AIPlayer(playerNumber, 70);
}

function clearHint(options = {}) {
  const { silent = false, redraw = true } = options;
  if (!hintMove && silent) {
    return;
  }
  hintMove = null;
  if (!silent) {
    if (currentMode === "tutorial" && tutorialScenario) {
      setHelperMessage(
        tutorialScenario.helperText || tutorialScenario.objective || ""
      );
    } else {
      setHelperMessage("");
    }
  }
  if (redraw) {
    drawBoard();
  }
  updateControls();
}

function updateControls() {
  const isTutorial = currentMode === "tutorial";
  const isHumanTurn = !gameOver && currentPlayer === 0;
  hintBtn.disabled = !isHumanTurn && !isTutorial;
  clearHintBtn.disabled = !hintMove;
  player2TypeSelect.disabled = isTutorial;
  difficultySelect.disabled =
    isTutorial || player2TypeSelect.value === "human";
}

function syncDifficultyAvailability() {
  if (modeSelect.value === "tutorial") {
    difficultySelect.disabled = true;
    return;
  }
  difficultySelect.disabled = player2TypeSelect.value === "human";
}

function setHelperMessage(message) {
  helperMessageEl.textContent = message;
}

function updateModeUI() {
  const isTutorial = modeSelect.value === "tutorial";
  if (tutorialControls) {
    tutorialControls.style.display = isTutorial ? "block" : "none";
  }
  if (opponentControls) {
    opponentControls.style.display = isTutorial ? "none" : "block";
  }
  if (difficultyControls) {
    difficultyControls.style.display = isTutorial ? "none" : "block";
  }

  hintBtn.textContent = isTutorial
    ? "Show Lesson Hint"
    : "Hint For Player 1";

  if (!isTutorial) {
    tutorialObjectiveEl.textContent = "";
  }
}

function recordOutcome({ winner, structure }) {
  if (currentMode === "tutorial") return;
  if (turnCounter === 0) return;

  stats.rounds += 1;

  if (winner === 1) {
    stats.p1Wins += 1;
    consecutiveWins += 1;
  } else {
    consecutiveWins = 0;
  }

  if (winner === 2) {
    stats.p2Wins += 1;
  }

  if (!winner) {
    stats.ties += 1;
  }

  if (
    winner === 1 &&
    structure &&
    ["bridge", "fork", "ring"].includes(structure)
  ) {
    achievements[structure] = true;
  }

  if (winner === 1 && consecutiveWins >= 3) {
    achievements.streak = true;
  }

  if (winner && structure !== "timeout") {
    if (!stats.fastest || turnCounter < stats.fastest.turns) {
      stats.fastest = {
        player: winner,
        structure: structure || "standard",
        turns: turnCounter,
      };
    }
  }

  renderStats();
  renderAchievements();
  savePersistentProgress();
}

function formatStructureName(structure) {
  if (!structure || structure === "standard") return "Standard";
  return structure
    .split("-")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function renderStats() {
  statsRoundsEl.textContent = stats.rounds;
  statsP1WinsEl.textContent = stats.p1Wins;
  statsP2WinsEl.textContent = stats.p2Wins;
  statsTiesEl.textContent = stats.ties;

  if (stats.fastest) {
    const { player, structure, turns } = stats.fastest;
    statsFastestEl.textContent = `P${player} • ${turns} moves • ${formatStructureName(
      structure
    )}`;
  } else {
    statsFastestEl.textContent = "—";
  }
}

function renderAchievements() {
  Object.entries(achievementBadges).forEach(([key, badge]) => {
    if (!badge) return;
    if (achievements[key]) {
      badge.textContent = "Unlocked";
      badge.classList.remove("bg-secondary", "text-dark");
      badge.classList.add("bg-success", "text-white");
    } else {
      badge.textContent = "Locked";
      badge.classList.remove("bg-success", "text-white");
      badge.classList.add("bg-secondary", "text-dark");
    }
  });
}

function renderMoveHistory() {
  moveHistoryList.innerHTML = "";
  moveHistory.forEach((entry) => {
    const li = document.createElement("li");
    li.className = "list-group-item d-flex justify-content-between align-items-center";

    if (entry.summary) {
      li.textContent = entry.summary;
    } else {
      const coord = `(${entry.row}, ${entry.col})`;
      const text = `Player ${entry.player} → ${coord}`;
      li.textContent = text;
      if (entry.result) {
        const badge = document.createElement("span");
        badge.className = "badge bg-primary rounded-pill";
        badge.textContent = entry.result;
        li.appendChild(badge);
      }
    }

    moveHistoryList.appendChild(li);
  });
}

function resetAnalysis() {
  analysisData = null;
  if (analysisSummary) {
    analysisSummary.style.display = "none";
    analysisSummary.textContent = "";
  }
}

function computeAnalysis(row, col) {
  const player = board[row][col];
  if (player !== 1 && player !== 2) return;

  const dim = board.length;
  const mask = board.map((r) => r.map((cell) => cell === player));
  const visitedSet = Havannah.bfsReachable(mask, [row, col]);
  const visited =
    visitedSet instanceof Set ? visitedSet : new Set(visitedSet);

  const cornerCoords = Havannah.getAllCorners(dim);
  const cornerKeys = cornerCoords.map(([x, y]) => `${x},${y}`);
  const touchedCorners = [];
  cornerKeys.forEach((key, idx) => {
    if (visited.has(key)) {
      touchedCorners.push({
        key,
        name: CORNER_NAMES[idx] || `Corner ${idx + 1}`,
      });
    }
  });

  const edges = Havannah.getAllEdges(dim);
  const touchedEdges = [];
  edges.forEach((edgeCoords, idx) => {
    if (
      edgeCoords.some(([x, y]) => visited.has(`${x},${y}`))
    ) {
      touchedEdges.push({
        index: idx,
        name: EDGE_NAMES[idx] || `Edge ${idx + 1}`,
      });
    }
  });

  const originKey = `${row},${col}`;
  analysisData = {
    player,
    origin: [row, col],
    visited,
    originKey,
    cornerKeys: new Set(touchedCorners.map((corner) => corner.key)),
    touchedCorners,
    touchedEdges,
  };

  if (analysisSummary) {
    const edgeLabel =
      touchedEdges.length > 0
        ? touchedEdges.map((edge) => edge.name).join(", ")
        : "None";
    const cornerLabel =
      touchedCorners.length > 0
        ? touchedCorners.map((corner) => corner.name).join(", ")
        : "None";
    const notes = [];
    if (touchedEdges.length >= 3) {
      notes.push("fork threat");
    }
    if (touchedCorners.length >= 2) {
      notes.push("bridge threat");
    }
    const noteLabel = notes.length ? ` — ${notes.join(" & ")}` : "";
    analysisSummary.innerHTML = `
      <strong>Player ${player}</strong> group size: ${visited.size}.<br>
      Edges reached: ${edgeLabel}.<br>
      Corners reached: ${cornerLabel}${noteLabel}.
    `;
    analysisSummary.style.display = "block";
  }

  drawBoard();
  renderGameTable(board);
}

function applyTutorialScenario(scenario) {
  if (!scenario) return;
  resetAnalysis();
  board = createInitialBoard(scenario.layers);
  Object.entries(scenario.placements || {}).forEach(([playerKey, coords]) => {
    const player = Number(playerKey);
    (coords || []).forEach(([x, y]) => {
      board[x][y] = player;
    });
  });
  currentPlayer = (scenario.startingPlayer || 1) - 1;
  hintMove = null;
  tutorialObjectiveEl.textContent = scenario.objective || "";
  setHelperMessage(scenario.helperText || scenario.objective || "");
  updateTimerLabels();
  updateCurrentTurn();
  drawBoard();
  renderGameTable(board);
}

function resetProgress() {
  if (!window.confirm("Reset all stored stats and achievements?")) {
    return;
  }
  stats.rounds = 0;
  stats.p1Wins = 0;
  stats.p2Wins = 0;
  stats.ties = 0;
  stats.fastest = null;
  Object.keys(achievements).forEach((key) => {
    achievements[key] = false;
  });
  consecutiveWins = 0;
  savePersistentProgress();
  renderStats();
  renderAchievements();
  setHelperMessage("Progress cleared. Start a new match to begin tracking again.");
}

function loadPersistentProgress() {
  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (!raw) return;
  try {
    const data = JSON.parse(raw);
    if (data.stats) {
      stats.rounds = data.stats.rounds || 0;
      stats.p1Wins = data.stats.p1Wins || 0;
      stats.p2Wins = data.stats.p2Wins || 0;
      stats.ties = data.stats.ties || 0;
      stats.fastest = data.stats.fastest || null;
    }
    if (data.achievements) {
      Object.keys(achievements).forEach((key) => {
        achievements[key] = Boolean(data.achievements[key]);
      });
    }
  } catch (err) {
    console.warn("Failed to read stored progress:", err);
  }
}

function savePersistentProgress() {
  try {
    const payload = {
      stats,
      achievements,
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch (err) {
    console.warn("Failed to save progress:", err);
  }
}

startGame();
