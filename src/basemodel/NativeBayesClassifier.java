package basemodel;


import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.NaiveBayes;
import java.util.Random;

public class NativeBayesClassifier {
   public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();

        DataSource source = new DataSource("data\\heart_disease_cleaned.arff");
        Instances dataset = source.getDataSet();

        dataset.setClassIndex(dataset.numAttributes() - 1);

        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(dataset);

        System.out.println("NaiveBayes Parameters: " + String.join(" ", nb.getOptions()));

        // Evaluate the model using 10-fold cross-validation
        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(nb, dataset, 10, new Random(42));

        // Print confusion matrix
        System.out.println("Confusion Matrix:\n" + eval.toMatrixString());

        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println("Precision = " + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("F-Measure = " + eval.fMeasure(1));
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println(eval.toClassDetailsString());

        SerializationHelper.write("models\\NaiveBayes.model", nb);

        long endTime = System.nanoTime();

        long duration = endTime - startTime;
        System.out.println("Runtime: " + duration + " nanoseconds");
    }    
}
