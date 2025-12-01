package basemodel;

import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.rules.OneR;
import java.util.Random;
import java.io.File;
import weka.core.SerializationHelper;

public class OneRTuningInfoGain {
    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();
        DataSource source = new DataSource("data\\InfoGain_data.arff");
        Instances dataset = source.getDataSet();

        dataset.setClassIndex(dataset.numAttributes() - 1);

        String[] options = new String[2];
        options[0] = "-B"; options[1] = "12";

        OneR oner = new OneR();
        oner.setOptions(options);
        oner.buildClassifier(dataset);
        System.out.println("OneR Selected Parameters: " + String.join(" ", oner.getOptions()));

        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(oner, dataset, 10, new Random(42));

        System.out.println("Confusion Matrix:\n" + eval.toMatrixString());

        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println("Precision = " + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("F-Measure = " + eval.fMeasure(1));
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println(eval.toClassDetailsString());

        SerializationHelper.write("models\\OneRInfoGain.model", oner);

        long endTime = System.nanoTime();

        long duration = endTime - startTime;
        System.out.println("Runtime: " + duration + " nanoseconds");
    }
}