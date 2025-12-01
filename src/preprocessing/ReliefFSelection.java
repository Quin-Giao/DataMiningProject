package preprocessing;

import weka.attributeSelection.AttributeSelection;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.attributeSelection.Ranker;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.attributeSelection.ReliefFAttributeEval;
import java.io.File;

public class ReliefFSelection {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("data\\heart_disease_cleaned.arff");
        Instances dataset = source.getDataSet();
        dataset.setClassIndex(dataset.numAttributes() - 1);

        // Evaluate feature importance using ReliefF
        AttributeSelection attrSelection = new AttributeSelection();
        ReliefFAttributeEval eval = new ReliefFAttributeEval();
        Ranker ranker = new Ranker();
        ranker.setNumToSelect(10);
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
        saver.setFile(new File("data\\ReliefF_data.arff "));
        saver.writeBatch();

        // Print results
        System.out.println("Selected Attributes:");
        for (int attr : selectedAttributes) {
            System.out.println(dataset.attribute(attr).name());
        } 
    }
}
