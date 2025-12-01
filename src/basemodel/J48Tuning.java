package basemodel;

import weka.classifiers.trees.J48;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.SerializationHelper;
import java.util.Random;

public class J48Tuning {
    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();

        DataSource trainSource = new DataSource("data\\InfoGain_data.arff");
        Instances trainDataset = trainSource.getDataSet();

        trainDataset.setClassIndex(trainDataset.numAttributes() - 1);

        String[] options = new String[4];
        options[0] = "-C"; options[1] = "0.25";
        options[2] = "-M"; options[3] = "1";

        J48 j48 = new J48();
        j48.setOptions(options);

        Evaluation eval = new Evaluation(trainDataset);
        eval.crossValidateModel(j48, trainDataset, 10, new Random(42)); // 42 for reproducibility

        System.out.println("J48 Selected Parameters: " + String.join(" ", j48.getOptions()));

        System.out.println("Confusion Matrix:\n" + eval.toMatrixString());

        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println("Precision = " + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("F-Measure = " + eval.fMeasure(1));
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println("\n" + eval.toClassDetailsString());

        SerializationHelper.write("models\\J48TuningBinaryModel.model", j48);

        long endTime = System.nanoTime();

        long duration = endTime - startTime;
        System.out.println("Runtime: " + duration + " nanoseconds");
    }
}