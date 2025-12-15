package preprocessing;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.File;

public class ResampleSmoteBalance {

    public static void main(String[] args) throws Exception {

        DataSource source = new DataSource("data\\heart_disease_feature_engineered.arff");
        Instances dataset = source.getDataSet();
        dataset.setClassIndex(dataset.numAttributes() - 1);

        // Convert class attribute to nominal if it's numeric
        if (dataset.classAttribute().isNumeric()) {
            NumericToNominal convert = new NumericToNominal();
            String indices = String.valueOf(dataset.classIndex() + 1);
            convert.setAttributeIndices(indices);
            convert.setInputFormat(dataset);
            dataset = Filter.useFilter(dataset, convert);
            System.out.println("✓ Converted class attribute to nominal");
        }

        Resample resample = new Resample();
        resample.setBiasToUniformClass(1.0);
        resample.setNoReplacement(false);
        resample.setSampleSizePercent(80.0);
        resample.setRandomSeed(1);
        resample.setInputFormat(dataset);

        Instances resampledData = Filter.useFilter(dataset, resample);

        SMOTE smote = new SMOTE();
        smote.setClassValue("1");
        smote.setPercentage(0);
        smote.setNearestNeighbors(5);
        smote.setRandomSeed(1);
        smote.setInputFormat(resampledData);

        Instances balancedData = Filter.useFilter(resampledData, smote);

        ArffSaver saver = new ArffSaver();
        saver.setInstances(balancedData);
        saver.setFile(new File("data\\heart_disease_balanced.arff"));
        saver.writeBatch();

        System.out.println("Original: " + dataset.numInstances());
        System.out.println("After Resample: " + resampledData.numInstances());
        System.out.println("Final: " + balancedData.numInstances());
    }
}
