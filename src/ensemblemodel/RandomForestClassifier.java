package ensemblemodel;

import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

public class RandomForestClassifier {
    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();

        DataSource source = new DataSource("data\\InfoGain_data.arff");
        Instances dataset = source.getDataSet();

        dataset.setClassIndex(dataset.numAttributes() - 1);

        RandomForest randomForest = new RandomForest();
        randomForest.buildClassifier(dataset);
        System.out.println("RandomForest Parameters: " + String.join(" ", randomForest.getOptions()));

        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(randomForest, dataset, 10, new Random(42));

        System.out.println("Confusion Matrix:\n" + eval.toMatrixString());

        // Print additional evaluation metrics
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println("Precision = " + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("F-Measure = " + eval.fMeasure(1));
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println("\n" + eval.toClassDetailsString());

        SerializationHelper.write("models\\RandomForestClassifier.model", randomForest);

        long endTime = System.nanoTime();

        long duration = endTime - startTime;
        System.out.println("Runtime: " + duration + " nanoseconds");
    }
}