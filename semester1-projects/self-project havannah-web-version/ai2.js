class AIPlayer2 {
  constructor(playerNumber, simulations = 150) {
    this.playerNumber = playerNumber;
    this.opponentNumber = playerNumber === 1 ? 2 : 1;
    this.simulations = simulations;
    this.moveCounter = 0;
    this.size = 0;
    this.criticalBlocks = this.getCriticalBlocks(); // Initialize critical blocks
  }

  // Define the critical blocks based on the game board size
  getCriticalBlocks() {
    return [
      // [(0, 0), (0, 3), (1, 2)] block
      [
        [0, 0],
        [0, 3],
        [1, 2],
      ],
      // [(3, 0), (6, 3), (4, 2)] block
      [
        [3, 0],
        [6, 3],
        [4, 2],
      ],
      // [(6, 3), (3, 6), (4, 4)] block
      [
        [6, 3],
        [3, 6],
        [4, 4],
      ],
      // [(3, 6), (0, 6), (2, 5)] block
      [
        [3, 6],
        [0, 6],
        [2, 5],
      ],
      // [(0, 6), (0, 3), (1, 4)] block
      [
        [0, 6],
        [0, 3],
        [1, 4],
      ],
      // [(0, 0), (3, 0), (2, 1)] block
      [
        [0, 0],
        [3, 0],
        [2, 1],
      ],
    ];
  }

  // Check if opponent is about to complete a critical block
  checkOpponentCriticalBlocks(state) {
    for (const block of this.criticalBlocks) {
      let opponentCount = 0;
      let emptyCount = 0;
      let emptyAction = null;

      // Iterate through each block
      for (const [row, col] of block) {
        if (state[row][col] === this.opponentNumber) {
          opponentCount++;
        } else if (state[row][col] === 0) {
          emptyCount++;
          emptyAction = [row, col];
        }
      }

      // If the opponent has two pieces and there's one empty space left, block it
      if (opponentCount === 2 && emptyCount === 1) {
        return emptyAction; // Block this critical block
      }
    }

    return null; // No critical block to block
  }

  // Check if AI is about to complete a critical block
  checkCriticalBlocks(state) {
    for (const block of this.criticalBlocks) {
      let playerCount = 0;
      let emptyCount = 0;
      let emptyAction = null;

      // Iterate through each block
      for (const [row, col] of block) {
        if (state[row][col] === this.playerNumber) {
          playerCount++;
        } else if (state[row][col] === 0) {
          emptyCount++;
          emptyAction = [row, col];
        }
      }

      // If the AI has two pieces and there's one empty space left, complete this block
      if (playerCount === 2 && emptyCount === 1) {
        return emptyAction; // Complete this critical block
      }
    }

    return null; // No critical block to complete
  }

  getMove(state) {
    if (this.moveCounter === 0) {
      this.size = (state.length + 1) / 2;
    }
    this.moveCounter++;
    console.log(state);
    const validActions = this.getValidActions(state);

    // Check for immediate winning move for AI
    for (const action of validActions) {
      state[action[0]][action[1]] = this.playerNumber;
      const [win] = checkWin(state, action, this.playerNumber);
      state[action[0]][action[1]] = 0;
      if (win) {
        return action;
      }
    }

    // Check for immediate winning move for opponent
    for (const action of validActions) {
      state[action[0]][action[1]] = this.opponentNumber;
      const [win] = checkWin(state, action, this.opponentNumber);
      state[action[0]][action[1]] = 0;
      if (win) {
        return action;
      }
    }

    // **If AI is Player 2 (defensive strategy)**:
    // Block opponent's critical block before forming own.
    if (this.playerNumber === 2) {
      const blockOpponent = this.checkOpponentCriticalBlocks(state);
      if (blockOpponent) {
        return blockOpponent; // Block the opponent's critical block
      }
    }

    // **If AI is Player 1 or no critical block needs blocking**:
    // Try to form own critical block.
    const criticalMove = this.checkCriticalBlocks(state);
    if (criticalMove) {
      return criticalMove; // Form own critical block
    }

    // Evaluate moves using heuristics if no critical blocks found
    let bestActions = null;
    let bestScore = -Infinity;

    for (const action of validActions) {
      state[action[0]][action[1]] = this.playerNumber;
      const opponentMoves = this.getValidActions(state);
      const maxOpponentThreatScore = this.evaluateOpponentThreat(
        state,
        opponentMoves
      );
      let currentScore = this.assignHeuristicScore(state, action);
      if (this.moveCounter > 1) {
        currentScore += this.runMonteCarloSimulations(state, action);
      }
      currentScore -= maxOpponentThreatScore;
      state[action[0]][action[1]] = 0;

      if (currentScore > bestScore) {
        bestScore = currentScore;
        bestActions = [action];
      } else if (currentScore === bestScore) {
        bestActions.push(action);
      }
    }

    return bestActions[Math.floor(Math.random() * bestActions.length)];
  }

  getValidActions(state) {
    const validMoves = [];
    for (let i = 0; i < state.length; i++) {
      for (let j = 0; j < state[i].length; j++) {
        if (state[i][j] === 0) {
          validMoves.push([i, j]);
        }
      }
    }
    return validMoves;
  }

  evaluateOpponentThreat(state, opponentMoves) {
    let maxOpponentThreatScore = 0;
    for (const opponentMove of opponentMoves) {
      state[opponentMove[0]][opponentMove[1]] = this.opponentNumber;
      const [win] = checkWin(state, opponentMove, this.opponentNumber);
      if (win) {
        maxOpponentThreatScore = Math.max(maxOpponentThreatScore, 100);
      }
      state[opponentMove[0]][opponentMove[1]] = 0;
    }
    return maxOpponentThreatScore;
  }

  assignHeuristicScore(state, action) {
    let score = 0;
    const dim = state.length;
    state[action[0]][action[1]] = this.playerNumber;

    if (getEdge(action, dim) !== -1) {
      score += 2;
    }
    if (getCorner(action, dim) !== -1) {
      score += 3;
    }

    const [win, structure] = checkWin(state, action, this.playerNumber);
    if (win) {
      score += structure === "fork" || structure === "ring" ? 200 : 150;
    }

    state[action[0]][action[1]] = 0;
    return score;
  }

  runMonteCarloSimulations(state, action) {
    let winCount = 0;
    let lossCount = 0;

    for (let i = 0; i < this.simulations; i++) {
      const simulationState = JSON.parse(JSON.stringify(state));
      simulationState[action[0]][action[1]] = this.playerNumber;
      const winner = this.simulateRandomGame(simulationState);
      if (winner === this.playerNumber) {
        winCount++;
      } else if (winner === this.opponentNumber) {
        lossCount++;
      }
    }

    return (winCount - lossCount) * 10;
  }

  simulateRandomGame(state) {
    let currentPlayer = this.opponentNumber;
    while (true) {
      const validActions = this.getValidActions(state);
      if (validActions.length === 0) return 0;
      const action =
        validActions[Math.floor(Math.random() * validActions.length)];
      state[action[0]][action[1]] = currentPlayer;
      const [win] = checkWin(state, action, currentPlayer);
      if (win) return currentPlayer;
      currentPlayer =
        currentPlayer === this.playerNumber
          ? this.opponentNumber
          : this.playerNumber;
    }
  }
}
