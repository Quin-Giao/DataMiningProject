package basemodel;

import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.Logistic;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import java.util.Random;
import weka.core.SerializationHelper;

public class LogisticClassifier {
    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();

        DataSource source = new DataSource("data\\InfoGain_data.arff");
        Instances data = source.getDataSet();

        data.setClassIndex(data.numAttributes() - 1);

        NominalToBinary nominalFilter = new NominalToBinary();
        nominalFilter.setInputFormat(data);
        Instances dataset = Filter.useFilter(data, nominalFilter);        

        // Create and build the classifier
        Logistic log = new Logistic();
        log.buildClassifier(dataset);

        System.out.println("Logistic Parameters: " + String.join(" ", log.getOptions()));

        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(log, dataset, 10, new Random(1));

        System.out.println("Confusion Matrix:\n" + eval.toMatrixString());

        // Print additional evaluation metrics
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println("Precision = " + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("F-Measure = " + eval.fMeasure(1));
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println(eval.toClassDetailsString());

        SerializationHelper.write("models\\LogisticClassifier.model", log);

        long endTime = System.nanoTime();

        long duration = endTime - startTime;
        System.out.println("Runtime: " + duration + " nanoseconds");
    }
}