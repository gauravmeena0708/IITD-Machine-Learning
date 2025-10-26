class AIPlayer {
  constructor(playerNumber, simulations = 150) {
    this.playerNumber = playerNumber;
    this.opponentNumber = playerNumber === 1 ? 2 : 1;
    this.simulations = simulations;
    this.moveCounter = 0;
    this.size = 0;
  }

  getMove(state) {
    if (this.moveCounter === 0) {
      this.size = (state.length + 1) / 2;
    }
    this.moveCounter++;

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

    // Evaluate moves using heuristics
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
