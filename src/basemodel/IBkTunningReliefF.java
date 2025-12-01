package basemodel;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;
import weka.core.SerializationHelper;

public class IBkTunningReliefF {
    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();

        DataSource source = new DataSource("data\\ReliefF_data.arff");
        Instances dataset = source.getDataSet();

        dataset.setClassIndex(dataset.numAttributes() - 1);

        String[] options = new String[6];
        options[0] = "-K"; options[1] = "3";
        options[2] = "-W"; options[3] = "0";
        options[4] = "-A"; options[5] = "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.ManhattanDistance -R first-last\"";

        IBk ibk = new IBk();
        ibk.setOptions(options);
        ibk.buildClassifier(dataset);

        System.out.println("IBk Selected Parameters: " + String.join(" ", ibk.getOptions()));

        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(ibk, dataset, 10, new Random(42));

        System.out.println("Confusion Matrix:\n" + eval.toMatrixString());

        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println("Precision = " + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("F-Measure = " + eval.fMeasure(1));
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println(eval.toClassDetailsString());

        SerializationHelper.write("models\\IBkTuningReliefF.model", ibk);

        long endTime = System.nanoTime();

        long duration = endTime - startTime;
        System.out.println("Runtime: " + duration + " nanoseconds");
    }     
}
