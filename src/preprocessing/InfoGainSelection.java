package preprocessing;

import weka.attributeSelection.AttributeSelection;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.attributeSelection.Ranker;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.attributeSelection.InfoGainAttributeEval;
import java.io.File;

public class InfoGainSelection {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("data\\heart_disease_balanced.arff");
        Instances dataset = source.getDataSet();
        dataset.setClassIndex(dataset.numAttributes() - 1);

        // Evaluate feature importance using information gain
        AttributeSelection attrSelection = new AttributeSelection();
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker ranker = new Ranker();
        ranker.setNumToSelect(15);
        attrSelection.setEvaluator(eval);
        attrSelection.setSearch(ranker);
        attrSelection.SelectAttributes(dataset);

        // Get selected attributes
        int[] selectedAttributes = attrSelection.selectedAttributes();

        // Create a Remove filter to keep only the selected attributes
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(selectedAttributes);
        remove.setInvertSelection(true);
        remove.setInputFormat(dataset);

        // Apply the filter
        Instances selectedData = Filter.useFilter(dataset, remove);

        // Save the selected attributes to a new ARFF file
        ArffSaver saver = new ArffSaver();
        saver.setInstances(selectedData);
        saver.setFile(new File("data\\InfoGain_data.arff"));
        saver.writeBatch();

        // Print results with InfoGain values
        System.out.println("\n=== InfoGain Attribute Ranking ===\n");
        System.out.println("Rank\tInfoGain\tAttribute");
        System.out.println("----\t--------\t---------");
        
        double[][] rankedAttributes = attrSelection.rankedAttributes();
        for (int i = 0; i < rankedAttributes.length; i++) {
            int attrIndex = (int) rankedAttributes[i][0];
            double infoGainValue = rankedAttributes[i][1];
            String attrName = dataset.attribute(attrIndex).name();
            System.out.printf("%d\t%.6f\t%s\n", (i + 1), infoGainValue, attrName);
        }
        
        System.out.println("\n=== Top 15 Selected Attributes Saved to InfoGain_data.arff ===");
    }
}