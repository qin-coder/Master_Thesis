package org.evosuite.ga.metaheuristics;

import org.evosuite.Properties;
import org.evosuite.ga.Chromosome;
import org.evosuite.utils.LoggingUtils;

import java.io.FileWriter;
import java.io.IOException;

/**
 * Listener that implement the QLearning algorithm into DynaMOSA (Many Objective Sorting Algorithm)
 *
 * @author Xuwei Qin
 */
public class RL_Listener<T extends Chromosome<T>> implements SearchListener<T> {

    private int iterations = 0;
    private boolean firstIteration = true;

    private final int RUN_QL_EVERY = 10;
    private final int LOG_EVERY = 100;

    public static double INITIAL_CROSSOVER_RATE = 0.67;
    public static double INITIAL_MUTATION_RATE = 0.75;



    @Override
    public void searchStarted(GeneticAlgorithm algorithm) {
        Properties.CROSSOVER_RATE = INITIAL_CROSSOVER_RATE;
        Properties.MUTATION_RATE = INITIAL_MUTATION_RATE;
        LoggingUtils.getEvoLogger().info("Search started.");
    }

    @Override
    public void iteration(GeneticAlgorithm algorithm) {
        if (iterations % LOG_EVERY == 0) {
            LoggingUtils.getEvoLogger().info("Current iteration: " + iterations);
        }

        if (iterations % RUN_QL_EVERY == 0 && !firstIteration) {
            // every 20 iterations running QLearning，and not the first iteration

            Chromosome chromosome = algorithm.getBestIndividual();

            double currentFtns = chromosome.getFitness();
            double parentFtns = chromosome.parentFitness;

            // check whether currentFtns is infinite
            if (Double.isInfinite(currentFtns)) {
                LoggingUtils.getEvoLogger().warn("Fitness is Infinity. Skipping QLearning for this iteration.");
                return;
            }

            if (Double.isInfinite(parentFtns)) {
                LoggingUtils.getEvoLogger().warn("Fitness is Infinity. Skipping QLearning for this iteration.");
                return;
            }

            // check whether parentFtns or currentFtns is 0.0，if it is,skip it
            if (parentFtns == 0.0 || currentFtns == 0.0) {
                LoggingUtils.getEvoLogger().warn("Parent Fitness is 0.0. Skipping QLearning for this iteration.");
                return;
            }

            // Log current and parent fitness values
            LoggingUtils.getEvoLogger().info("Current ftns: " + currentFtns);
            LoggingUtils.getEvoLogger().info("parentFtns: " + parentFtns);
            LoggingUtils.getEvoLogger().info("CROSSOVER_RATE: " + Properties.CROSSOVER_RATE);
            LoggingUtils.getEvoLogger().info("MUTATION_RATE: " + Properties.MUTATION_RATE);



            Environment env = new Environment();
            int numStates = 3;
            int numActions = 2;
            RLModel rlModel = new RLModel(numStates, numActions);

            Qlearning.runQLearning(env, rlModel, currentFtns, parentFtns, Properties.CROSSOVER_RATE, Properties.MUTATION_RATE);
        }

        if (firstIteration) {
            firstIteration = false;
        }

        iterations++;
    }

    @Override
    public void searchFinished(GeneticAlgorithm algorithm) {
        LoggingUtils.getEvoLogger().info("Search finished.");

    }

    @Override
    public void fitnessEvaluation(Chromosome individual) {
    }

    @Override
    public void modification(Chromosome individual) {
    }
}
