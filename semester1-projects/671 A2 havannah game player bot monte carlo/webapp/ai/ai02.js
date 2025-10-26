class AI02 {
    constructor(playerNumber) {
        this.playerNumber = playerNumber;
        this.opponentNumber = playerNumber === 1 ? 2 : 1;
        this.simulations = 200;
    }

    getWinProbabilities(state, n) {
        const probabilities = { bridge: 0, fork: 0, ring: 0 };
        const validActions = getValidActions(state);
        const totalSimulations = validActions.length * this.simulations;

        for (const action of validActions) {
            const winCounts = { bridge: 0, fork: 0, ring: 0 };
            for (let i = 0; i < this.simulations; i++) {
                const simulationState = JSON.parse(JSON.stringify(state));
                simulationState[action[0]][action[1]] = this.playerNumber;
                const winType = this.simulateNMoves(simulationState, n);
                if (winType) {
                    winCounts[winType]++;
                }
            }

            for (const winType in probabilities) {
                probabilities[winType] += winCounts[winType] / totalSimulations;
            }
        }

        return probabilities;
    }

    simulateNMoves(state, n) {
        let currentPlayer = this.playerNumber;
        for (let i = 0; i < n; i++) {
            const validActions = getValidActions(state);
            if (validActions.length === 0) {
                break;
            }

            const action = validActions[Math.floor(Math.random() * validActions.length)];
            state[action[0]][action[1]] = currentPlayer;

            const { win, way } = checkWin(state, action, currentPlayer);
            if (win) {
                return way;
            }

            currentPlayer = currentPlayer === this.playerNumber ? this.opponentNumber : this.playerNumber;
        }

        return null;
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

        for (const action of validActions) {
            state[action[0]][action[1]] = this.opponentNumber;
            const { win } = checkWin(state, action, this.opponentNumber);
            state[action[0]][action[1]] = 0;
            if (win) {
                return action;
            }
        }

        let bestActions = [];
        let bestScore = -Infinity;

        for (const action of validActions) {
            const probabilities = this.getWinProbabilities(state, 3);
            const currentScore = probabilities.bridge * 5 + probabilities.fork * 5 + probabilities.ring * 5;

            if (currentScore > bestScore) {
                bestScore = currentScore;
                bestActions = [action];
            } else if (currentScore === bestScore) {
                bestActions.push(action);
            }
        }

        if (bestActions.length === 0) {
            return validActions[Math.floor(Math.random() * validActions.length)];
        }

        return bestActions[Math.floor(Math.random() * bestActions.length)];
    }
}
