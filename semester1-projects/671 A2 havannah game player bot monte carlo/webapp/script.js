const gameBoard = document.getElementById("game-board");
const turnIndicator = document.getElementById("turn-indicator");
const player1ScoreEl = document.getElementById("player1-score");
const player2ScoreEl = document.getElementById("player2-score");
const boardSize = board.length;

let currentPlayer = 1;
let aiPlayer = new AI01(2);
let gameActive = true;
let player1Score = 0;
let player2Score = 0;
const startingLayout = board.map((row) => [...row]);

document.getElementById("ai-select").addEventListener("change", (event) => {
  const choice = event.target.value;
  aiPlayer = choice === "ai02" ? new AI02(2) : new AI01(2);
  if (!gameActive) {
    // Refresh indicator so the UI reflects the new opponent choice
    updateTurnIndicator();
  }
});

document.getElementById("reset-button").addEventListener("click", resetBoard);

function resetBoard() {
  for (let r = 0; r < boardSize; r++) {
    for (let c = 0; c < boardSize; c++) {
      board[r][c] = startingLayout[r][c];
    }
  }
  currentPlayer = 1;
  gameActive = true;
  updateTurnIndicator();
  renderBoard();
}

function renderBoard() {
  gameBoard.innerHTML = "";
  for (let i = 0; i < boardSize; i++) {
    for (let j = 0; j < boardSize; j++) {
      const value = board[i][j];
      const hex = document.createElement("div");
      hex.classList.add("hex");
      if (value === 0) {
        hex.classList.add("empty");
      } else if (value === 1) {
        hex.classList.add("player-one");
      } else if (value === 2) {
        hex.classList.add("player-two");
      } else if (value === 3) {
        hex.classList.add("blocked");
      }
      hex.addEventListener("click", () => handleCellClick(i, j));
      gameBoard.appendChild(hex);
    }
  }
}

function handleCellClick(row, col) {
  if (!gameActive || currentPlayer !== 1 || board[row][col] !== 0) return;
  placeStone(row, col, 1);
}

function placeStone(row, col, player) {
  board[row][col] = player;
  renderBoard();

  const { win, way } = Havannah.checkWin(board, [row, col], player);
  if (win) {
    finalizeRound(player, `${way}`);
    return;
  }

  if (Havannah.getValidActions(board).length === 0) {
    finalizeRound(null, "tie");
    return;
  }

  currentPlayer = player === 1 ? 2 : 1;
  updateTurnIndicator();
  if (currentPlayer === 2 && gameActive) {
    setTimeout(makeAiMove, 120);
  }
}

function makeAiMove() {
  if (!gameActive) return;
  const move = aiPlayer.getMove(board);
  if (!move) {
    finalizeRound(null, "tie");
    return;
  }
  placeStone(move[0], move[1], 2);
}

function finalizeRound(winner, detail) {
  gameActive = false;
  if (winner === 1) {
    player1Score += 1;
    player1ScoreEl.textContent = String(player1Score);
  } else if (winner === 2) {
    player2Score += 1;
    player2ScoreEl.textContent = String(player2Score);
  }

  const message =
    winner === 1 || winner === 2
      ? `Player ${winner} wins by ${detail}!`
      : "Round ended in a tie.";
  turnIndicator.textContent = message;
  setTimeout(() => alert(message), 30);
}

function updateTurnIndicator() {
  if (!gameActive) return;
  turnIndicator.textContent = `Current Turn: Player ${currentPlayer}`;
}

renderBoard();
updateTurnIndicator();
