package basemodel;

import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import java.io.File;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.rules.ZeroR;
import java.util.Random;

public class ZeroRClassifier {
    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();
        DataSource source = new DataSource("data\\heart_diseasr_cleaned.arff");
        Instances dataset = source.getDataSet();

        dataset.setClassIndex(dataset.numAttributes() - 1);

        ZeroR zeroR = new ZeroR();
        zeroR.buildClassifier(dataset);

        System.out.println("ZeroR Parameters: " + String.join(" ", zeroR.getOptions()));

        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(zeroR, dataset, 10, new Random(42));

        // Print the confusion matrix
        System.out.println("Confusion Matrix:\n" + eval.toMatrixString());

        // Print additional evaluation metrics
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println("Precision = " + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("F-Measure = " + eval.fMeasure(1));
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println(eval.toClassDetailsString());

        SerializationHelper.write("models\\ZeroRBinaryModel.model", zeroR);

        long endTime = System.nanoTime();

        long duration = endTime - startTime;
        System.out.println("Runtime: " + duration + " nanoseconds");
    }
}