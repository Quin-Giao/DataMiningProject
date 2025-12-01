package basemodel;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import java.util.Random;
import weka.core.SerializationHelper;

public class J48Classifier {
    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();

        DataSource source = new DataSource("data\\InfoGain_data.arff");
        Instances dataset = source.getDataSet();

        dataset.setClassIndex(dataset.numAttributes() - 1);

        J48 j48 = new J48();
        j48.buildClassifier(dataset);

        System.out.println("J48 Parameters: " + String.join(" ", j48.getOptions()));

        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(j48, dataset, 10, new Random(42));

        // Print the confusion matrix
        System.out.println("\nConfusion Matrix:\n" + eval.toMatrixString());

        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println("Precision = " + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("F-Measure = " + eval.fMeasure(1));
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println("\n" + eval.toClassDetailsString());

        SerializationHelper.write("models\\J48BinaryModel.model", j48);

        long endTime = System.nanoTime();

        long duration = endTime - startTime;
        System.out.println("Runtime: " + duration + " nanoseconds");
    }
}