const board = [
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
  [3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
  [3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3],
  [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3],
  [3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3],
  [3, 3, 3, 3, 3, 3, 0, 0, 0, 3, 3, 3, 3, 3, 3],
  [3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3],
];

const havannahGlobals = {
  getValidActions: Havannah.getValidActions,
  getNeighbours: Havannah.getNeighbours,
  getEdge: Havannah.getEdge,
  getCorner: Havannah.getCorner,
  isValid: Havannah.isValid,
  moveCoordinates: Havannah.moveCoordinates,
  threeForwardMoves: Havannah.threeForwardMoves,
  checkRing: Havannah.checkRing,
  checkForkAndBridge: Havannah.checkForkAndBridge,
  checkWin: Havannah.checkWin,
  bfsReachable: Havannah.bfsReachable,
};

Object.assign(window, havannahGlobals);
