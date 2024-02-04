# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if len(newFood.asList()) == 0:
            foodScore = 0
        else:
            closestFoodDist = min(manhattanDistance(newPos,foodPos) for foodPos in newFood.asList())
            foodScore = 1/closestFoodDist

        closestGhostDist = min(manhattanDistance(newPos,ghostState.getPosition()) for ghostState in newGhostStates)
        if closestGhostDist == 0:
            ghostScore = 0
        else:
            ghostScore = 2 / closestGhostDist

        return successorGameState.getScore() + foodScore - ghostScore

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(state, agentIndex, depth):
            maxAgent = True if agentIndex == 0 else False
            operator = max if maxAgent else min
            opt_val = -float('inf') if maxAgent else float('inf')
            next_agent = (agentIndex+1) % state.getNumAgents()
            next_depth = depth - 1 if maxAgent else depth
            opt_act = None
            if next_depth == 0 or state.isLose() or state.isWin():
                return self.evaluationFunction(state), None
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                action_val = operator(opt_val,minimax(successor, next_agent, next_depth)[0])
                if action_val != opt_val:
                    opt_val = action_val
                    opt_act = action
            return opt_val, opt_act
        return minimax(gameState, self.index, self.depth + 1)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """


    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def ab(state, agentIndex, depth, alpha, beta):
            maxAgent = True if agentIndex == 0 else False
            operator = max if maxAgent else min
            opt_val = -float('inf') if maxAgent else float('inf')
            next_agent = (agentIndex+1) % state.getNumAgents()
            next_depth = depth - 1 if maxAgent else depth
            opt_act = None
            if next_depth == 0 or state.isLose() or state.isWin():
                return self.evaluationFunction(state), None
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                action_val = operator(opt_val,ab(successor, next_agent, next_depth, alpha, beta)[0])
                if action_val != opt_val:
                    opt_val = action_val
                    opt_act = action
                if maxAgent:
                    if action_val > beta:
                        return opt_val, opt_act
                    alpha = max(alpha, opt_val)
                else:
                    if action_val < alpha:
                        return opt_val, opt_act
                    beta = min(beta, opt_val)
            return opt_val, opt_act
        return ab(gameState, self.index, self.depth + 1,-float('inf'), float('inf'))[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, agentIndex, depth):
            maxAgent = True if agentIndex == 0 else False
            opt_val = -float('inf')
            next_agent = (agentIndex+1) % state.getNumAgents()
            next_depth = depth - 1 if maxAgent else depth
            opt_act = None
            if next_depth == 0 or state.isLose() or state.isWin():
                return self.evaluationFunction(state), None
            if not maxAgent:
                expect = 0.0
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    expect += expectimax(successor, next_agent,next_depth)[0]
                return expect / len(state.getLegalActions(agentIndex)), None
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                action_val = max(expectimax(successor,next_agent,next_depth)[0],opt_val)
                if action_val > opt_val:
                    opt_val = action_val
                    opt_act = action
            return opt_val, opt_act
        return expectimax(gameState, self.index, self.depth + 1)[1]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacPosi = currentGameState.getPacmanPosition()
    foodPosi= currentGameState.getFood().asList()
    ghostState = currentGameState.getGhostStates()
    scaredTimes = min([ghostState.scaredTimer for ghostState in ghostState])
    score = currentGameState.getScore()
    foods = len(foodPosi)
    # foodDist = [manhattanDistance(food, pacPosi) for food in foodPosi]
    minGhostDist = min([manhattanDistance(pacPosi,ghost.getPosition()) for ghost in ghostState])

    # We need to use diff strategy to evaluate
    if len(foodPosi) < 4:
        # award more coefficient of score for less foods
        # also give more score for more ghost dist
        totalScore = 3*score + 0.1*minGhostDist - 5 * foods
    else:
        # less score for more food
        if scaredTimes > 0 or minGhostDist >= 4:
            # dangerous
            totalScore = score - 3*foods
        else:
            # safe, give more score on safe dist, less penalty
            # compensate for loss
            totalScore = score + 2*minGhostDist - 2*foods
    return totalScore

# Abbreviation
better = betterEvaluationFunction
