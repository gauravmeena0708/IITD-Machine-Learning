class AI01 {
    constructor(playerNumber) {
        this.playerNumber = playerNumber;
        this.opponentNumber = playerNumber === 1 ? 2 : 1;
    }

    getMove(state) {
        const validActions = getValidActions(state);

        for (const action of validActions) {
            state[action[0]][action[1]] = this.playerNumber;
            const { win } = checkWin(state, action, this.playerNumber);
            state[action[0]][action[1]] = 0;
            if (win) {
                return action;
            }
        }

        let bestActions = null;
        let bestScore = -Infinity;

        for (const action of validActions) {
            state[action[0]][action[1]] = this.playerNumber;

            const opponentMoves = getValidActions(state);
            let maxOpponentThreatScore = 0;

            for (const opponentMove of opponentMoves) {
                state[opponentMove[0]][opponentMove[1]] = this.opponentNumber;
                const { win } = checkWin(state, opponentMove, this.opponentNumber);
                if (win) {
                    maxOpponentThreatScore = Math.max(maxOpponentThreatScore, 100);
                } else {
                    const secondStepValidActions = getValidActions(state);
                    for (const secondStepAction of secondStepValidActions) {
                        state[secondStepAction[0]][secondStepAction[1]] = this.playerNumber;
                        const { win: secondStepWin } = checkWin(state, secondStepAction, this.playerNumber);
                        if (secondStepWin) {
                            maxOpponentThreatScore = Math.max(maxOpponentThreatScore, -50);
                        }
                        state[secondStepAction[0]][secondStepAction[1]] = 0;
                    }
                }
                state[opponentMove[0]][opponentMove[1]] = 0;
            }

            const currentScore = this.evaluateAction(state, action) - maxOpponentThreatScore;
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

    evaluateAction(state, action) {
        let score = 0;
        const dim = state.length;

        state[action[0]][action[1]] = this.playerNumber;

        if (getEdge(action, dim) !== -1) {
            score += 2;
        }
        if (getCorner(action, dim) !== -1) {
            score += 3;
        }

        const { win, way } = checkWin(state, action, this.playerNumber);
        if (win) {
            if (way === "fork") {
                score += 200;
            } else if (way === "bridge") {
                score += 150;
            } else if (way === "ring") {
                score += 200;
            }
        }

        if (this.createsStrategicConnection(state, action, dim)) {
            score += 100;
        }

        state[action[0]][action[1]] = 0;

        return score;
    }

    createsStrategicConnection(state, action, dim) {
        const edgesConnected = new Set();
        const neighbors = getNeighbours(dim, action);
        for (const neighbor of neighbors) {
            if (state[neighbor[0]][neighbor[1]] === this.playerNumber) {
                const edge = getEdge(neighbor, dim);
                if (edge !== -1) {
                    edgesConnected.add(edge);
                }
            }
        }
        return edgesConnected.size > 1;
    }
}
