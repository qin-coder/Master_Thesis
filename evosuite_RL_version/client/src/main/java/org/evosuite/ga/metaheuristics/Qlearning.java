package org.evosuite.ga.metaheuristics;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

import org.evosuite.Properties;
import org.evosuite.utils.LoggingUtils;
/**
 * Implementation of the QLearning algorithm
 *
 * @author Xuwei Qin
 */
class Environment {

    double previousCrossoverRate = 0.67; // Init crossover rate for the first chromosome
    double previousMutationRate = 0.75;
    int state;


    double penaltyFactor = 0.01;

    double rewardFactor = 0.02;

    double implementFactor = 0.01;

    double differenceFactor = 0.03;


    public double[] step(int action, double currentFitness,double parentFitness, double numCrossoverRate, double numMutationRate) {
        double fitnessDifference = parentFitness - currentFitness;
        double crossoverChange = numCrossoverRate - previousCrossoverRate;
        double mutationChange = numMutationRate - previousMutationRate;


        double reward = differenceFactor * fitnessDifference + rewardFactor * Math.abs(crossoverChange) - penaltyFactor * Math.abs(mutationChange);
        //The reward is calculated in the following equation: we are encouraged decrease the fitness value,the less the fitness value,
        //the better quality of the test case.So we are rewarding decrease scenarios,punishing decrease scenarios.
        //Our goal is maximum exploitation the algorithm,so we add additional reward:crossoverChange,additional punishment: mutationChange

        //set limits for the reward
        if (reward > 1.0) {
            reward = 1.0;
        } else if (reward < -1.0) {
            reward = -1.0;
        }


        if (action == 0) { // More exploitation
            if (reward > 0) {
                numCrossoverRate *= (1 + reward * implementFactor); // Increase crossover rate
                numMutationRate *= (1 - reward * implementFactor); // Decrease mutation rate

            } else {
                numCrossoverRate *= (1 + reward * implementFactor); // Decrease crossover rate
                numMutationRate *= (1 + Math.abs(reward) * implementFactor); // Increase mutation rate
            }
        } else if (action == 1) { // More exploration
            if (reward > 0) {
                numCrossoverRate *= (1 - reward * implementFactor); // Decrease crossover rate
                numMutationRate *= (1 + reward * implementFactor); // Increase mutation rate

            } else {
                numCrossoverRate *= (1 - reward * implementFactor); // Increase crossover rate
                numMutationRate *= (1 - Math.abs(reward) * implementFactor); // Decrease mutation rate

            }
        }

//        // Update state
        if (currentFitness == parentFitness) {
            state = 1;
        } else if (currentFitness < parentFitness) {
            state = 2;//better

        } else {
            state = 3; //worse

        }

        previousCrossoverRate = numCrossoverRate;
        previousMutationRate = numMutationRate;

        return new double[] { state, currentFitness, numCrossoverRate, numMutationRate, reward };
    }
}

class RLModel {
    int numStates;
    int numActions;
    double[][] qTable;
    private int state;
    double epsilon = 0.1;

    double learningRate = 0.02;
    double discountFactor = 0.95;

    public RLModel(int numStates, int numActions) {
        this.numStates = numStates;
        this.numActions = numActions;
        qTable = new double[numStates][numActions];
        this.state = 1;
    }

    public int chooseAction() {
        Random rand = new Random();
        if (rand.nextDouble() < epsilon) {
            // Explore: choose a random action
            return rand.nextInt(numActions);
        } else {
            // Exploit: choose the best action based on the Q-table
            return argMax(qTable[state - 1]);
        }
    }

    private int argMax(double[] array) {
        int bestAction = 0;
        double max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                bestAction = i;
            }
        }
        return bestAction;
    }
    private static final double MAX_Q_VALUE = 5.0;


    public double updateQTable(int state, int action, double reward, int nextState) {

        // Ensure state value is within bounds
        if (state < 1 || state > numStates) {
            // Handle out-of-bounds state value
            throw new IllegalArgumentException("State value is out of bounds: " + state);
        }

        // Adjust state value to match array index
        //The reason for using state - 1 in the updateQTable method in the RLModel class is that in array indexing, the array index starts at 0, while the state (state) starts at 1.
        // Therefore, in order to properly access the rows in a particular state in a Q table, the state value needs to be subtracted from 1 to get the correct array index.
        double currentQ = qTable[this.state -1][action];

        // Ensure nextState value is within bounds
        if (nextState < 1 || nextState > numStates) {
            // Handle out-of-bounds nextState value
            throw new IllegalArgumentException("NextState value is out of bounds: " + nextState);
        }

        // Adjust nextState value to match array index

        double maxNextQ = Double.NEGATIVE_INFINITY;

        for (double qValue : qTable[nextState-1]) {
            if (qValue > maxNextQ) {
                maxNextQ = qValue;
            }
        }
        double newQ = (1 - learningRate) * currentQ + learningRate * (reward + discountFactor * maxNextQ);

        if (newQ > MAX_Q_VALUE) {
            newQ = MAX_Q_VALUE;
        }

        qTable[this.state -1][action] = newQ;

        return newQ;

    }

    public double[][] getQTable() {

        return qTable;
    }

    public double[] runEpisode(Environment env, double currentFitness,double parentFitness,  double numCrossoverRate, double numMutationRate) {
        double[] result = new double[6];
        int action = chooseAction();

        double[] stepResult = env.step(action, currentFitness, parentFitness,numCrossoverRate, numMutationRate);

        int nextState = (int) stepResult[0];
        double newFitness = stepResult[1];
        double newCrossoverRate = stepResult[2];
        double newMutationRate = stepResult[3];
        double reward = stepResult[4];

        double newQ = updateQTable(state, action, reward, nextState);

        //updateQTable(state, action, reward, nextState);
        this.state = nextState;

        result[0] = stepResult[0]; // state
        result[1] = stepResult[1]; // numfts
        result[2] = newCrossoverRate; // new crossover rate
        result[3] = newMutationRate; // new mutation rate
        result[4] = reward;
        result[5] = newQ;

        return result;
    }

}

public class Qlearning {
    public static double[][] initialQTable  = {
            { 0.2, 0.2 },
            { 0.2, 0.2 },
            { 0.2, 0.2 } };

    private static int episodes = 0;


    public static void runQLearning(Environment env, RLModel rlModel, double currentFitness,double parentFitness, double numCrossoverRate,
                                    double numMutationRate) {

        int action = rlModel.chooseAction();

        rlModel.qTable = initialQTable;

        double[] episodeResult = rlModel.runEpisode(env, currentFitness, parentFitness,numCrossoverRate, numMutationRate);
        int state = env.state;

        Properties.CROSSOVER_RATE = episodeResult[2];
        Properties.MUTATION_RATE = episodeResult[3];

        // Apply constraints to crossover rate and mutation rate
        Properties.CROSSOVER_RATE = Math.max(0.3, Math.min(0.99, Properties.CROSSOVER_RATE));
        Properties.MUTATION_RATE = Math.max(0.3, Math.min(0.99, Properties.MUTATION_RATE));


        double reward = episodeResult[4];
        double newQ = episodeResult[5]; // 从结果数组中获取newQ值
        LoggingUtils.getEvoLogger().info("Successfully running experiment");

        // Write Q-table to file
        String qTableFilePath = "qtable.txt";
        writeQTableToFile(rlModel.qTable, qTableFilePath );

        // Write log

        try (FileWriter writer = new FileWriter("log.txt", true)) {
            writer.write("------------------------QLearning------------------------\n");
            writer.write("Episodes : "+ ++episodes + "\n");
            writer.write("Parent Fitness:" + parentFitness + "\n");
            writer.write("Current Fitness:" + currentFitness + "\n");
            writer.write("Difference Fitness:" + (parentFitness - currentFitness) + "\n");
            writer.write("Reward: " + reward + "\n");
            writer.write("newQ: " + newQ + "\n");
            writer.write("Current crossover rate: " + numCrossoverRate + "\n");
            writer.write("Current mutation rate: " + numMutationRate + "\n");
            writer.write("State: " + state + "\n");
            writer.write("Action: " + action + "\n");
            writer.write("0: " + episodeResult[0] + "\n");//state
            writer.write("1: " + episodeResult[1] + "\n");//ftns
            writer.write("2: " + episodeResult[2] + "\n");//cross
            writer.write("3: " + episodeResult[3] + "\n");//mutation
            writer.write("4: " + episodeResult[4] + "\n");//reward
            writer.write("-------------------------------------------------------\n");
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Save the current Q-table state for the next run
        initialQTable = rlModel.getQTable();
    }


    private static void writeQTableToFile(double[][] qTable, String fileName) {
        try (FileWriter writer = new FileWriter(fileName)) {
            writer.write("Q-Table:\n");
            for (double[] row : qTable) {
                for (double value : row) {
                    writer.write(value + " ");
                }
                writer.write("\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
