import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class LoadData {

    public static Instances loadArff(String path) throws Exception {
        DataSource source = new DataSource(path);
        Instances data = source.getDataSet();
        if (data.classIndex() == -1 && data.numAttributes() > 0) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.err.println("Usage: java LoadData <data.arff>");
            System.exit(1);
        }

        Instances data = loadArff(args[0]);

        System.out.println("Relation: " + data.relationName());
        System.out.println("Instances: " + data.numInstances());
        System.out.println("Attributes: " + data.numAttributes());
        System.out.println("Class attribute: " + data.classAttribute().name());
    }
}
