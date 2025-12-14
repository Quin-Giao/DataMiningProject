package basemodel;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.classifiers.functions.SMO;
import java.util.Random;
import weka.core.SerializationHelper;

public class SVMTuning {
    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();

        DataSource source = new DataSource("data\\ReliefF_data.arff");
        Instances data = source.getDataSet();

        data.setClassIndex(data.numAttributes() - 1);

        NominalToBinary nominalFilter = new NominalToBinary();
        nominalFilter.setInputFormat(data);
        Instances dataset = Filter.useFilter(data, nominalFilter);

        String[] options = new String[6];
        options[0] = "-C"; options[1] = "0.5";
        options[2] = "-N"; options[3] = "2";
        options[4] = "-L"; options[5] = "0.01";

        SMO svm = new SMO();
        svm.setOptions(options);
        svm.buildClassifier(dataset);
        System.out.println("SVM Selected Parameters: " + String.join(" ", svm.getOptions()));


        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(svm, dataset, 10, new Random(42));

        // Print the confusion matrix
        System.out.println("Confusion Matrix:\n" + eval.toMatrixString());

        // Print additional evaluation metrics
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println("Precision = " + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("F-Measure = " + eval.fMeasure(1));
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println(eval.toClassDetailsString());

        SerializationHelper.write("models\\SVMTuning.model", svm);

        long endTime = System.nanoTime();

        long duration = endTime - startTime;
        System.out.println("Runtime: " + duration + " nanoseconds");
    }
}